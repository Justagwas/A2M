from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CONFIG_SCHEMA_VERSION, ENGINE_UNIFORM_VELOCITY_DEFAULT, ENGINE_UNIFORM_VELOCITY_MAX, ENGINE_UNIFORM_VELOCITY_MIN, GPU_BATCH_SIZE_MAX, GPU_BATCH_SIZE_MIN, OUTPUT_MIDI_DIR, UI_SCALE_PERCENT_MAX, UI_SCALE_PERCENT_MIN
from .resource_service import normalize_gpu_memory_max_batch, normalize_performance_mode
from .paths import config_path, legacy_config_paths


@dataclass(slots=True)
class AppConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    device_preference: str = 'cpu'
    gpu_batch_size: int = 2
    gpu_memory_max_batch: int = 4
    gpu_provider_preference: str = 'dml'
    gpu_runtime_enabled: bool = False
    gpu_runtime_path: str = ''
    gpu_last_reason_code: str = ''
    gpu_last_reason_text: str = ''
    gpu_last_validated_provider: str = ''
    gpu_provider_rollback: dict[str, object] | None = None
    engine_pedals_enabled: bool = True
    engine_velocity_mode: str = 'expressive'
    engine_uniform_velocity: int = ENGINE_UNIFORM_VELOCITY_DEFAULT
    cpu_performance_mode: str = 'balanced'
    gpu_performance_mode: str = 'balanced'
    hardware_defaults_initialized: bool = False
    auto_check_updates: bool = True
    theme_mode: str = 'dark'
    ui_scale_percent: int = 100
    download_location: str = str(OUTPUT_MIDI_DIR)
    window_geometry: str = ''

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'AppConfig':
        if not isinstance(payload, dict):
            return default_config()
        preference = str(payload.get('device_preference', 'cpu')).strip().lower()
        if preference not in {'cpu', 'gpu'}:
            preference = 'cpu'
        try:
            batch = int(payload.get('gpu_batch_size', 2))
        except Exception:
            batch = 2
        batch = max(GPU_BATCH_SIZE_MIN, min(GPU_BATCH_SIZE_MAX, batch))
        memory_max_batch = normalize_gpu_memory_max_batch(payload.get('gpu_memory_max_batch', 4))
        batch = min(batch, memory_max_batch)
        provider = str(payload.get('gpu_provider_preference', 'dml') or '').strip().lower()
        if provider not in {'cuda', 'dml'}:
            provider = 'dml'
        runtime_path = str(payload.get('gpu_runtime_path', '') or '').strip()
        runtime_enabled = _coerce_bool(payload.get('gpu_runtime_enabled', False), default=False) and bool(runtime_path)
        validated_provider = str(payload.get('gpu_last_validated_provider', '') or '').strip().lower()
        if validated_provider not in {'', 'cuda', 'dml', 'cpu'}:
            validated_provider = ''
        velocity_mode = str(payload.get('engine_velocity_mode', 'expressive') or '').strip().lower()
        if velocity_mode not in {'expressive', 'uniform'}:
            velocity_mode = 'expressive'
        try:
            uniform_velocity = int(payload.get('engine_uniform_velocity', ENGINE_UNIFORM_VELOCITY_DEFAULT))
        except Exception:
            uniform_velocity = ENGINE_UNIFORM_VELOCITY_DEFAULT
        uniform_velocity = max(ENGINE_UNIFORM_VELOCITY_MIN, min(ENGINE_UNIFORM_VELOCITY_MAX, uniform_velocity))
        theme = str(payload.get('theme_mode', 'dark')).strip().lower()
        if theme not in {'dark', 'light'}:
            theme = 'dark'
        try:
            scale = int(payload.get('ui_scale_percent', 100))
        except Exception:
            scale = 100
        scale = max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, scale))
        download_location = str(payload.get('download_location', OUTPUT_MIDI_DIR) or '').strip() or str(OUTPUT_MIDI_DIR)
        legacy_performance_mode = normalize_performance_mode(payload.get('transcription_performance_mode', 'balanced'))
        return cls(
            schema_version=CONFIG_SCHEMA_VERSION,
            device_preference=preference,
            gpu_batch_size=batch,
            gpu_memory_max_batch=memory_max_batch,
            gpu_provider_preference=provider,
            gpu_runtime_enabled=runtime_enabled,
            gpu_runtime_path=runtime_path,
            gpu_last_reason_code=str(payload.get('gpu_last_reason_code', '') or '').strip(),
            gpu_last_reason_text=str(payload.get('gpu_last_reason_text', '') or '').strip(),
            gpu_last_validated_provider=validated_provider,
            gpu_provider_rollback=_normalize_gpu_provider_rollback(payload.get('gpu_provider_rollback')),
            engine_pedals_enabled=_coerce_bool(payload.get('engine_pedals_enabled', True), default=True),
            engine_velocity_mode=velocity_mode,
            engine_uniform_velocity=uniform_velocity,
            cpu_performance_mode=normalize_performance_mode(payload.get('cpu_performance_mode', legacy_performance_mode)),
            gpu_performance_mode=normalize_performance_mode(payload.get('gpu_performance_mode', legacy_performance_mode)),
            hardware_defaults_initialized=_coerce_bool(
                payload.get('hardware_defaults_initialized', False),
                default=False,
            ),
            auto_check_updates=_coerce_bool(payload.get('auto_check_updates', True), default=True),
            theme_mode=theme,
            ui_scale_percent=scale,
            download_location=download_location,
            window_geometry=str(payload.get('window_geometry', '') or ''),
        )

    def to_dict(self) -> dict[str, Any]:
        provider = str(self.gpu_provider_preference or 'dml').strip().lower()
        if provider not in {'cuda', 'dml'}:
            provider = 'dml'
        try:
            batch = int(self.gpu_batch_size)
        except Exception:
            batch = 2
        batch = max(GPU_BATCH_SIZE_MIN, min(GPU_BATCH_SIZE_MAX, batch))
        memory_max_batch = normalize_gpu_memory_max_batch(self.gpu_memory_max_batch)
        batch = min(batch, memory_max_batch)
        velocity_mode = str(self.engine_velocity_mode or 'expressive').strip().lower()
        if velocity_mode not in {'expressive', 'uniform'}:
            velocity_mode = 'expressive'
        try:
            uniform_velocity = int(self.engine_uniform_velocity)
        except Exception:
            uniform_velocity = ENGINE_UNIFORM_VELOCITY_DEFAULT
        uniform_velocity = max(
            ENGINE_UNIFORM_VELOCITY_MIN,
            min(ENGINE_UNIFORM_VELOCITY_MAX, uniform_velocity),
        )
        return {
            'schema_version': CONFIG_SCHEMA_VERSION,
            'device_preference': 'gpu' if self.device_preference == 'gpu' else 'cpu',
            'gpu_batch_size': batch,
            'gpu_memory_max_batch': memory_max_batch,
            'gpu_provider_preference': provider,
            'gpu_runtime_enabled': bool(self.gpu_runtime_enabled),
            'gpu_runtime_path': self.gpu_runtime_path or '',
            'gpu_last_reason_code': self.gpu_last_reason_code or '',
            'gpu_last_reason_text': self.gpu_last_reason_text or '',
            'gpu_last_validated_provider': self.gpu_last_validated_provider or '',
            'gpu_provider_rollback': _normalize_gpu_provider_rollback(self.gpu_provider_rollback),
            'engine_pedals_enabled': bool(self.engine_pedals_enabled),
            'engine_velocity_mode': velocity_mode,
            'engine_uniform_velocity': uniform_velocity,
            'cpu_performance_mode': normalize_performance_mode(self.cpu_performance_mode),
            'gpu_performance_mode': normalize_performance_mode(self.gpu_performance_mode),
            'hardware_defaults_initialized': bool(self.hardware_defaults_initialized),
            'auto_check_updates': bool(self.auto_check_updates),
            'theme_mode': self.theme_mode if self.theme_mode in {'dark', 'light'} else 'dark',
            'ui_scale_percent': max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, int(self.ui_scale_percent))),
            'download_location': str(self.download_location or '').strip() or str(OUTPUT_MIDI_DIR),
            'window_geometry': self.window_geometry or '',
        }


def default_config() -> AppConfig:
    return AppConfig()


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    return default


def _normalize_gpu_provider_rollback(value: Any) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    provider = str(value.get('provider', '') or '').strip().lower()
    if provider not in {'cuda', 'dml'}:
        return None
    device = str(value.get('device', 'cpu') or '').strip().lower()
    if device not in {'cpu', 'gpu'}:
        device = 'cpu'
    runtime_path = str(value.get('runtime_path', '') or '').strip()
    runtime_enabled = _coerce_bool(value.get('runtime_enabled', False), default=False) and bool(runtime_path)
    try:
        memory_max_batch = normalize_gpu_memory_max_batch(value.get('gpu_memory_max_batch', 4))
        batch_size = int(value.get('gpu_batch_size', 2))
    except Exception:
        batch_size = 2
        memory_max_batch = 4
    batch_size = max(GPU_BATCH_SIZE_MIN, min(GPU_BATCH_SIZE_MAX, batch_size, memory_max_batch))
    return {
        'provider': provider,
        'runtime_enabled': runtime_enabled,
        'runtime_path': runtime_path if runtime_enabled else '',
        'device': device,
        'gpu_batch_size': batch_size,
        'gpu_memory_max_batch': memory_max_batch,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    temporary = path.with_name(f'{path.name}.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def load_config() -> AppConfig:
    path = config_path()
    if path.exists():
        payload = _read_json(path)
        if payload is not None:
            return AppConfig.from_dict(payload)

    for legacy_path in legacy_config_paths():
        if not legacy_path.exists():
            continue
        payload = _read_json(legacy_path)
        if payload is None:
            continue
        config = AppConfig.from_dict(payload)
        save_config(config)
        return config

    config = default_config()
    save_config(config)
    return config


def save_config(config: AppConfig) -> str | None:
    payload = config.to_dict()
    path = config_path()
    if _write_json(path, payload):
        return str(path)
    print(f"[A2M] Warning: failed to write config: {path}", file=sys.stderr)
    return None
