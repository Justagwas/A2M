from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .runtime_artifacts import runtime_root_from_path as _shared_runtime_root_from_path
from .runtime_path_service import prime_runtime_path as _shared_prime_runtime_path

EXIT_OK = 0
EXIT_PROVIDER_UNAVAILABLE = 10
EXIT_PROVIDER_LOAD_FAILED = 11
EXIT_SESSION_CREATION_FAILED = 12
EXIT_RUNTIME_PACKAGE_INVALID = 13

REASON_RUNTIME_NOT_INSTALLED = 'RUNTIME_NOT_INSTALLED'
REASON_RUNTIME_PACKAGE_INVALID = 'RUNTIME_PACKAGE_INVALID'
REASON_PROVIDER_NOT_EXPOSED = 'PROVIDER_NOT_EXPOSED'
REASON_PROVIDER_LOAD_FAILED = 'PROVIDER_LOAD_FAILED'
REASON_SESSION_CREATION_FAILED = 'SESSION_CREATION_FAILED'
REASON_MODEL_MISSING_FOR_GPU_CHECK = 'MODEL_MISSING_FOR_GPU_CHECK'
REASON_HELPER_PROCESS_FAILED = 'HELPER_PROCESS_FAILED'
REASON_UNKNOWN_GPU_ERROR = 'UNKNOWN_GPU_ERROR'


@dataclass(slots=True)
class HelperResult:
    exit_code: int
    payload: dict[str, Any]


_DLL_DIR_HANDLES: list[object] = []


def _provider_ep(provider: str) -> str:
    normalized = str(provider or '').strip().lower()
    if normalized == 'cuda':
        return 'CUDAExecutionProvider'
    if normalized == 'dml':
        return 'DmlExecutionProvider'
    return ''


def _runtime_root(runtime_path: str) -> Path:
    base = Path(str(runtime_path or '').strip())
    return _shared_runtime_root_from_path(base)


def _prime_runtime_path(runtime_root: Path) -> None:
    _shared_prime_runtime_path(runtime_root, dll_dir_handles=_DLL_DIR_HANDLES)


def _import_ort(runtime_path: str):
    runtime_str = str(runtime_path or '').strip()
    runtime_root = _runtime_root(runtime_str) if runtime_str else None
    if runtime_root is not None:
        package_dir = runtime_root / 'onnxruntime'
        capi_dir = package_dir / 'capi'
        if runtime_root.exists() and package_dir.is_dir() and (package_dir / '__init__.py').exists() and capi_dir.is_dir():
            _prime_runtime_path(runtime_root)
            for name in [n for n in list(sys.modules) if n == 'onnxruntime' or n.startswith('onnxruntime.')]:
                sys.modules.pop(name, None)
            init_file = package_dir / '__init__.py'
            spec = importlib.util.spec_from_file_location('onnxruntime', str(init_file), submodule_search_locations=[str(package_dir)])
            if spec is None or spec.loader is None:
                raise RuntimeError('Failed to load ONNX runtime package spec from runtime path.')
            module = importlib.util.module_from_spec(spec)
            sys.modules['onnxruntime'] = module
            spec.loader.exec_module(module)
            return module
    try:
        import onnxruntime as ort
        return ort
    except Exception as exc:
        if runtime_str and runtime_root is not None and not runtime_root.exists():
            raise RuntimeError('Runtime path does not exist.') from exc
        raise RuntimeError('Runtime package is missing required onnxruntime files.') from exc


def _error_payload(reason_code: str, reason_text: str, *, details: str='') -> dict[str, Any]:
    payload: dict[str, Any] = {
        'ok': False,
        'reason_code': str(reason_code or REASON_UNKNOWN_GPU_ERROR),
        'reason_text': str(reason_text or 'GPU helper request failed.'),
    }
    if details:
        payload['details'] = str(details)
    return payload


def _success_payload(**kwargs: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {'ok': True, 'reason_code': '', 'reason_text': ''}
    payload.update(kwargs)
    return payload


def _probe_runtime(runtime_path: str) -> HelperResult:
    try:
        ort = _import_ort(runtime_path)
    except Exception as exc:
        return HelperResult(EXIT_RUNTIME_PACKAGE_INVALID, _error_payload(REASON_RUNTIME_PACKAGE_INVALID, 'Runtime package is invalid or incomplete.', details=str(exc)))
    try:
        providers = tuple((str(p) for p in ort.get_available_providers()))
        if not providers:
            providers = ('CPUExecutionProvider',)
        return HelperResult(EXIT_OK, _success_payload(runtime_available=True, available_providers=providers, cuda_available='CUDAExecutionProvider' in providers, dml_available='DmlExecutionProvider' in providers))
    except Exception as exc:
        return HelperResult(EXIT_PROVIDER_LOAD_FAILED, _error_payload(REASON_PROVIDER_LOAD_FAILED, 'Failed to query ONNX providers from runtime package.', details=str(exc)))


def _validate_provider(runtime_path: str, provider: str) -> HelperResult:
    ep = _provider_ep(provider)
    if not ep:
        return HelperResult(EXIT_PROVIDER_UNAVAILABLE, _error_payload(REASON_PROVIDER_NOT_EXPOSED, 'Requested provider is invalid.'))
    probe = _probe_runtime(runtime_path)
    if probe.exit_code != EXIT_OK:
        return probe
    providers = tuple((str(p) for p in probe.payload.get('available_providers', ())))
    if ep not in providers:
        return HelperResult(EXIT_PROVIDER_UNAVAILABLE, _error_payload(REASON_PROVIDER_NOT_EXPOSED, f'{provider.upper()} execution provider is not exposed by this runtime package.', details=f'Available providers: {", ".join(providers) if providers else "none"}'))
    return HelperResult(EXIT_OK, _success_payload(provider=provider, provider_ep=ep, available_providers=providers))


def _create_session(runtime_path: str, provider: str, model_path: str) -> HelperResult:
    model = Path(str(model_path or '').strip())
    if not model.exists():
        return HelperResult(EXIT_SESSION_CREATION_FAILED, _error_payload(REASON_MODEL_MISSING_FOR_GPU_CHECK, 'Model file is missing for GPU session check.', details=str(model)))
    provider_result = _validate_provider(runtime_path, provider)
    if provider_result.exit_code != EXIT_OK:
        return provider_result
    ep = str(provider_result.payload.get('provider_ep', '') or '')
    if not ep:
        return HelperResult(EXIT_PROVIDER_UNAVAILABLE, _error_payload(REASON_PROVIDER_NOT_EXPOSED, 'GPU provider is unavailable.'))
    try:
        ort = _import_ort(runtime_path)
    except Exception as exc:
        return HelperResult(EXIT_RUNTIME_PACKAGE_INVALID, _error_payload(REASON_RUNTIME_PACKAGE_INVALID, 'Runtime package is invalid or incomplete.', details=str(exc)))
    try:
        options = ort.SessionOptions()
        using_dml = ep == 'DmlExecutionProvider'
        options.enable_mem_pattern = False if using_dml else True
        if using_dml:
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(model), sess_options=options, providers=[ep])
        active_providers = tuple((str(p) for p in session.get_providers()))
        active = active_providers[0] if active_providers else 'CPUExecutionProvider'
        if active != ep:
            if ep == 'CUDAExecutionProvider':
                return HelperResult(EXIT_PROVIDER_LOAD_FAILED, _error_payload(REASON_PROVIDER_LOAD_FAILED, 'CUDA provider failed to initialize and session started on CPU.', details=f'Active provider: {active}'))
            return HelperResult(EXIT_PROVIDER_UNAVAILABLE, _error_payload(REASON_PROVIDER_NOT_EXPOSED, f'{provider.upper()} session started on CPU.', details=f'Active provider: {active}'))
        return HelperResult(EXIT_OK, _success_payload(provider=provider, provider_ep=ep, active_provider=active, available_providers=active_providers))
    except Exception as exc:
        message = str(exc)
        lower = message.lower()
        if 'loadlibrary' in lower or 'failed to load' in lower or 'provider' in lower:
            return HelperResult(EXIT_PROVIDER_LOAD_FAILED, _error_payload(REASON_PROVIDER_LOAD_FAILED, f'Failed to load {provider.upper()} provider.', details=message))
        return HelperResult(EXIT_SESSION_CREATION_FAILED, _error_payload(REASON_SESSION_CREATION_FAILED, f'Failed to create ONNX session on {provider.upper()}.', details=message))


def run_gpu_helper(task: str, *, runtime_path: str, provider: str='', model_path: str='') -> HelperResult:
    if str(provider or '').strip().lower() == 'cuda':
        try:
            from . import cuda_dependency_service
            cuda_dependency_service.ensure_cuda_runtime_bins_in_process_path()
        except Exception:
            pass
    normalized = str(task or '').strip().lower()
    if normalized == 'probe':
        return _probe_runtime(runtime_path)
    if normalized == 'validate-provider':
        return _validate_provider(runtime_path, provider)
    if normalized == 'create-session':
        return _create_session(runtime_path, provider, model_path)
    return HelperResult(EXIT_RUNTIME_PACKAGE_INVALID, _error_payload(REASON_UNKNOWN_GPU_ERROR, f'Unknown helper task: {task}'))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu-helper', dest='task', required=True)
    parser.add_argument('--runtime-path', dest='runtime_path', default='')
    parser.add_argument('--provider', dest='provider', default='')
    parser.add_argument('--model', dest='model_path', default='')
    return parser


def run_gpu_helper_cli(argv: list[str]) -> int:
    parser = _build_parser()
    try:
        args, _ = parser.parse_known_args(argv)
    except Exception as exc:
        print(json.dumps(_error_payload(REASON_HELPER_PROCESS_FAILED, 'Failed to parse helper arguments.', details=str(exc)), separators=(',', ':')))
        return EXIT_RUNTIME_PACKAGE_INVALID
    result = run_gpu_helper(str(args.task or ''), runtime_path=str(args.runtime_path or ''), provider=str(args.provider or ''), model_path=str(args.model_path or ''))
    print(json.dumps(result.payload, separators=(',', ':')))
    return int(result.exit_code)
