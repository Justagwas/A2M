from __future__ import annotations
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from . import gpu_runtime_service, runtime_service
from .gpu_helper import EXIT_OK, EXIT_PROVIDER_LOAD_FAILED, EXIT_PROVIDER_UNAVAILABLE, EXIT_RUNTIME_PACKAGE_INVALID, EXIT_SESSION_CREATION_FAILED, REASON_HELPER_PROCESS_FAILED, REASON_MODEL_MISSING_FOR_GPU_CHECK, REASON_PROVIDER_LOAD_FAILED, REASON_PROVIDER_NOT_EXPOSED, REASON_RUNTIME_NOT_INSTALLED, REASON_RUNTIME_PACKAGE_INVALID, REASON_SESSION_CREATION_FAILED, REASON_UNKNOWN_GPU_ERROR
from .paths import app_dir


@dataclass(slots=True)
class RuntimeDecision:
    device_mode: str
    active_provider: str
    gpu_model_name: str
    reason_code: str
    reason_text: str


class GpuRuntimeManager:

    def __init__(self) -> None:
        self._helper_timeout_seconds = 35

    @staticmethod
    def _normalize_provider(provider_pref: str | None) -> str:
        normalized = str(provider_pref or 'auto').strip().lower()
        if normalized in {'cuda', 'dml'}:
            return normalized
        return 'auto'

    @staticmethod
    def _normalize_device(device_mode: str | None) -> str:
        normalized = str(device_mode or 'cpu').strip().lower()
        if normalized == 'gpu':
            return 'gpu'
        return 'cpu'

    @staticmethod
    def _provider_from_helper_payload(payload: dict[str, Any]) -> str:
        provider_ep = str(payload.get('active_provider') or payload.get('provider_ep') or '').strip()
        if provider_ep == 'CUDAExecutionProvider':
            return 'cuda'
        if provider_ep == 'DmlExecutionProvider':
            return 'dml'
        provider = str(payload.get('provider') or '').strip().lower()
        if provider in {'cuda', 'dml'}:
            return provider
        return 'cpu'

    @staticmethod
    def _short_reason(reason_text: str) -> str:
        text = str(reason_text or '').strip()
        if not text:
            return 'GPU check failed.'
        first_line = text.splitlines()[0].strip()
        return first_line or text

    @staticmethod
    def _helper_command(task: str, runtime_path: str, provider: str='', model_path: str='') -> list[str]:
        runtime = str(runtime_path or '').strip()
        args = ['--gpu-helper', str(task or '').strip(), '--runtime-path', runtime]
        if provider:
            args.extend(['--provider', str(provider).strip().lower()])
        if model_path:
            args.extend(['--model', str(model_path).strip()])
        if getattr(sys, 'frozen', False):
            return [str(sys.executable)] + args
        return [str(sys.executable), str(app_dir() / 'A2M.py')] + args

    def _run_helper(self, task: str, *, runtime_path: str, provider: str='', model_path: str='') -> tuple[int, dict[str, Any]]:
        cmd = self._helper_command(task, runtime_path, provider=provider, model_path=model_path)
        kwargs: dict[str, Any] = {
            'capture_output': True,
            'text': True,
            'timeout': self._helper_timeout_seconds,
            'check': False,
        }
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            kwargs['startupinfo'] = startupinfo
            kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        try:
            proc = subprocess.run(cmd, **kwargs)
        except Exception as exc:
            return (1, {'ok': False, 'reason_code': REASON_HELPER_PROCESS_FAILED, 'reason_text': 'GPU helper process failed to run.', 'details': str(exc)})
        payload: dict[str, Any] = {}
        raw_stdout = str(proc.stdout or '').strip()
        if raw_stdout:
            for line in reversed(raw_stdout.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                except Exception:
                    continue
                if isinstance(candidate, dict):
                    payload = candidate
                    break
        if not payload:
            reason = f'GPU helper returned invalid output (exit {proc.returncode}).'
            details = str(proc.stderr or raw_stdout or '').strip()
            payload = {'ok': False, 'reason_code': REASON_HELPER_PROCESS_FAILED, 'reason_text': reason, 'details': details}
        return (int(proc.returncode), payload)

    @staticmethod
    def _reason_code_for_exit(exit_code: int, default: str='') -> str:
        if exit_code == EXIT_PROVIDER_UNAVAILABLE:
            return REASON_PROVIDER_NOT_EXPOSED
        if exit_code == EXIT_PROVIDER_LOAD_FAILED:
            return REASON_PROVIDER_LOAD_FAILED
        if exit_code == EXIT_SESSION_CREATION_FAILED:
            return REASON_SESSION_CREATION_FAILED
        if exit_code == EXIT_RUNTIME_PACKAGE_INVALID:
            return REASON_RUNTIME_PACKAGE_INVALID
        return default or REASON_UNKNOWN_GPU_ERROR

    @staticmethod
    def _runtime_path() -> str:
        return str(runtime_service.get_effective_runtime_root_for_gpu_checks() or '').strip()

    def probe_runtime_path(self, runtime_path: str | Path | None) -> dict[str, Any]:
        normalized = str(runtime_path or '').strip()
        code, payload = self._run_helper('probe', runtime_path=normalized)
        if code == EXIT_OK and bool(payload.get('ok', False)):
            providers = tuple((str(p) for p in payload.get('available_providers', ())))
            return {'runtime_available': bool(payload.get('runtime_available', True)), 'cuda_available': bool(payload.get('cuda_available', False)), 'dml_available': bool(payload.get('dml_available', False)), 'available_providers': providers, 'reason_code': '', 'reason_text': ''}
        reason_code = str(payload.get('reason_code') or '').strip()
        reason_text = self._short_reason(str(payload.get('reason_text') or '').strip())
        if not reason_code:
            reason_code = self._reason_code_for_exit(code)
        return {'runtime_available': False, 'cuda_available': False, 'dml_available': False, 'available_providers': tuple(), 'reason_code': reason_code, 'reason_text': reason_text}

    def probe_runtime_support(self) -> dict[str, Any]:
        runtime_path = self._runtime_path()
        if not runtime_path:
            return {'runtime_available': False, 'cuda_available': False, 'dml_available': False, 'available_providers': tuple(), 'reason_code': REASON_RUNTIME_NOT_INSTALLED, 'reason_text': 'GPU runtime is not installed.'}
        return self.probe_runtime_path(runtime_path)

    def validate_runtime_provider(self, provider_pref: str) -> RuntimeDecision:
        runtime_path = self._runtime_path()
        provider = self._normalize_provider(provider_pref)
        if provider == 'auto':
            detected = gpu_runtime_service.detect_runtime_provider(runtime_path) if runtime_path else ''
            provider = detected if detected in {'cuda', 'dml'} else gpu_runtime_service.resolve_provider_for_install('auto')
        code, payload = self._run_helper('validate-provider', runtime_path=runtime_path, provider=provider)
        if code == EXIT_OK and bool(payload.get('ok', False)):
            active_provider = self._provider_from_helper_payload(payload)
            model = gpu_runtime_service.get_gpu_model_name(active_provider)
            return RuntimeDecision(device_mode='gpu', active_provider=active_provider, gpu_model_name=model, reason_code='', reason_text='')
        reason_code = str(payload.get('reason_code') or '').strip()
        reason_text = self._short_reason(str(payload.get('reason_text') or '').strip())
        if not reason_code:
            reason_code = self._reason_code_for_exit(code)
        return RuntimeDecision(device_mode='cpu', active_provider='cpu', gpu_model_name='', reason_code=reason_code, reason_text=reason_text)

    def activate_gpu_or_fallback(self, device_pref: str, provider_pref: str, model_path: Path | str | None) -> RuntimeDecision:
        if self._normalize_device(device_pref) != 'gpu':
            return RuntimeDecision(device_mode='cpu', active_provider='cpu', gpu_model_name='', reason_code='', reason_text='')
        provider_decision = self.validate_runtime_provider(provider_pref)
        if provider_decision.device_mode != 'gpu':
            return provider_decision
        model = Path(model_path) if model_path else None
        if model is None:
            return provider_decision
        if not model.exists():
            return RuntimeDecision(device_mode='cpu', active_provider='cpu', gpu_model_name='', reason_code=REASON_MODEL_MISSING_FOR_GPU_CHECK, reason_text='Model file is missing for GPU validation.')
        runtime_path = self._runtime_path()
        code, payload = self._run_helper('create-session', runtime_path=runtime_path, provider=provider_decision.active_provider, model_path=str(model))
        if code == EXIT_OK and bool(payload.get('ok', False)):
            active_provider = self._provider_from_helper_payload(payload)
            model_name = gpu_runtime_service.get_gpu_model_name(active_provider)
            return RuntimeDecision(device_mode='gpu', active_provider=active_provider, gpu_model_name=model_name, reason_code='', reason_text='')
        reason_code = str(payload.get('reason_code') or '').strip()
        reason_text = self._short_reason(str(payload.get('reason_text') or '').strip())
        if not reason_code:
            reason_code = self._reason_code_for_exit(code)
        return RuntimeDecision(device_mode='cpu', active_provider='cpu', gpu_model_name='', reason_code=reason_code, reason_text=reason_text or 'GPU activation failed.')

    def prepare_gpu_runtime(self, provider_pref: str, model_path: Path | str | None) -> RuntimeDecision:
        return self.activate_gpu_or_fallback('gpu', provider_pref, model_path)

    def get_runtime_status(self) -> dict[str, Any]:
        runtime_path = self._runtime_path()
        provider = gpu_runtime_service.detect_runtime_provider(runtime_path) if runtime_path else ''
        probe = self.probe_runtime_support()
        runtime_available = bool(probe.get('runtime_available', False))
        return {'runtime_installed': bool(runtime_path) or runtime_available, 'provider': provider, 'runtime_path': runtime_path, 'runtime_available': runtime_available, 'cuda_available': bool(probe.get('cuda_available', False)), 'dml_available': bool(probe.get('dml_available', False)), 'reason_code': str(probe.get('reason_code', '') or ''), 'reason_text': str(probe.get('reason_text', '') or '')}
