from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import model_service, runtime_pack_service, runtime_service
from .gpu_helper import REASON_RUNTIME_NOT_INSTALLED
from .gpu_runtime_manager import GpuRuntimeManager, RuntimeDecision
from .paths import normalized_path_key

_PROVIDERS = ('dml', 'cuda')
StatusCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ProviderRuntimeStatus:
    provider: str
    runtime_path: str = ''
    runtime_installed: bool = False
    available: bool = False
    decision: RuntimeDecision | None = None

    @property
    def reason_code(self) -> str:
        return str(self.decision.reason_code if self.decision else '')

    @property
    def reason_text(self) -> str:
        return str(self.decision.reason_text if self.decision else '')


@dataclass(slots=True)
class RuntimeInventory:
    providers: dict[str, ProviderRuntimeStatus] = field(default_factory=dict)
    probes: dict[str, dict[str, object]] = field(default_factory=dict)
    model_path: str = ''
    cpu_available: bool = False
    cpu_runtime_path: str = ''
    cpu_reason_text: str = ''

    def provider_status(self, provider: str | None) -> ProviderRuntimeStatus:
        normalized = runtime_pack_service.resolve_provider_for_preference(provider)
        return self.providers.get(normalized, ProviderRuntimeStatus(provider=normalized))

def _append_candidate(candidates: list[str], runtime_path: str | Path | None) -> None:
    candidate = str(runtime_path or '').strip()
    if not candidate:
        return
    key = normalized_path_key(candidate)
    if any(normalized_path_key(existing) == key for existing in candidates):
        return
    candidates.append(candidate)


def inspect_runtime_inventory(
    *,
    configured_provider: str | None = None,
    configured_runtime_path: str | Path | None = None,
    status_callback: StatusCallback | None = None,
) -> RuntimeInventory:
    """Inspect both GPU backends without importing ONNX Runtime into the UI process."""

    manager = GpuRuntimeManager()
    inventory = RuntimeInventory()
    existing_model = model_service.get_existing_model_path()
    inventory.model_path = str(existing_model or '')
    configured = runtime_pack_service.resolve_provider_for_preference(configured_provider)
    packaged_path = str(runtime_service.get_packaged_runtime_root() or '').strip()
    probe_cache: dict[str, dict[str, object]] = {}
    installed_paths = {
        provider: runtime_pack_service.resolve_installed_pack(provider)[1]
        for provider in _PROVIDERS
    }

    def report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    def probe_path(runtime_path: str) -> dict[str, object]:
        key = normalized_path_key(runtime_path)
        cached = probe_cache.get(key)
        if cached is None:
            cached = dict(manager.probe_runtime_path(runtime_path))
            probe_cache[key] = cached
        return cached

    report('Checking CPU transcription support...')
    cpu_candidates: list[str] = []
    _append_candidate(cpu_candidates, packaged_path)
    _append_candidate(cpu_candidates, configured_runtime_path)
    for installed_path in installed_paths.values():
        _append_candidate(cpu_candidates, installed_path)
    for candidate in cpu_candidates:
        probe = probe_path(candidate)
        if bool(probe.get('runtime_available', False)):
            inventory.cpu_available = True
            inventory.cpu_runtime_path = candidate
            break
    if not inventory.cpu_available:
        inventory.cpu_reason_text = 'ONNX Runtime CPU support is not installed.'

    for provider in _PROVIDERS:
        label = runtime_pack_service.provider_display_name(provider)
        report(f'Checking {label} GPU support...')
        candidates: list[str] = []
        _append_candidate(candidates, installed_paths[provider])
        if provider == configured:
            _append_candidate(candidates, configured_runtime_path)
        _append_candidate(candidates, packaged_path)

        selected_path = ''
        selected_probe: dict[str, object] | None = None
        for candidate in candidates:
            if (
                installed_paths[provider]
                and normalized_path_key(candidate) == normalized_path_key(installed_paths[provider])
            ):
                probe = {
                    'runtime_available': True,
                    'cuda_available': provider == 'cuda',
                    'dml_available': provider == 'dml',
                    'available_providers': (
                        'CUDAExecutionProvider' if provider == 'cuda' else 'DmlExecutionProvider',
                        'CPUExecutionProvider',
                    ),
                    'reason_code': '',
                    'reason_text': '',
                }
            else:
                probe = probe_path(candidate)
            provider_available = bool(
                probe.get('dml_available' if provider == 'dml' else 'cuda_available', False)
            )
            if bool(probe.get('runtime_available', False)) and provider_available:
                selected_path = candidate
                selected_probe = probe
                break

        if not selected_path:
            decision = RuntimeDecision(
                device_mode='cpu',
                active_provider='cpu',
                gpu_model_name='',
                reason_code=REASON_RUNTIME_NOT_INSTALLED,
                reason_text=f'{label} runtime is not installed.',
            )
            inventory.providers[provider] = ProviderRuntimeStatus(
                provider=provider,
                decision=decision,
            )
            continue

        decision = manager.prepare_gpu_runtime_at(selected_path, provider, existing_model)
        inventory.providers[provider] = ProviderRuntimeStatus(
            provider=provider,
            runtime_path=selected_path,
            runtime_installed=True,
            available=decision.device_mode == 'gpu',
            decision=decision,
        )
        if selected_probe is not None:
            probe_cache[normalized_path_key(selected_path)] = selected_probe

    inventory.probes.update(probe_cache)
    return inventory
