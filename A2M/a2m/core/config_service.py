from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .constants import CONFIG_SCHEMA_VERSION, GPU_BATCH_SIZE_MAX, GPU_BATCH_SIZE_MIN, OUTPUT_MIDI_DIR, UI_SCALE_PERCENT_MAX, UI_SCALE_PERCENT_MIN
from .constants import MODERN_FRAME_THRESHOLD_DEFAULT, MODERN_FRAME_THRESHOLD_MAX, MODERN_FRAME_THRESHOLD_MIN
from .constants import MODERN_OFFSET_THRESHOLD_DEFAULT, MODERN_OFFSET_THRESHOLD_MAX, MODERN_OFFSET_THRESHOLD_MIN
from .constants import MODERN_ONSET_THRESHOLD_DEFAULT, MODERN_ONSET_THRESHOLD_MAX, MODERN_ONSET_THRESHOLD_MIN
from .constants import MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT, MODERN_PEDAL_OFFSET_THRESHOLD_MAX, MODERN_PEDAL_OFFSET_THRESHOLD_MIN
from .paths import config_read_paths, config_write_paths

@dataclass(slots=True)
class AppConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    device_preference: str = 'cpu'
    conversion_method: str = 'legacy_v1'
    modern_adaptive_thresholds_enabled: bool = True
    modern_input_normalization_enabled: bool = True
    modern_smart_overlap_stitching_enabled: bool = True
    modern_auto_calibration_enabled: bool = True
    modern_cleanup_scale: float = 1.0
    modern_pedal_cluster_scale: float = 1.0
    modern_alignment_gate_scale: float = 1.0
    modern_manual_onset_threshold: float = MODERN_ONSET_THRESHOLD_DEFAULT
    modern_manual_offset_threshold: float = MODERN_OFFSET_THRESHOLD_DEFAULT
    modern_manual_frame_threshold: float = MODERN_FRAME_THRESHOLD_DEFAULT
    modern_manual_pedal_offset_threshold: float = MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT
    gpu_batch_size: int = 4
    gpu_provider_preference: str = 'auto'
    gpu_runtime_enabled: bool = False
    gpu_runtime_path: str = ''
    gpu_last_reason_code: str = ''
    gpu_last_reason_text: str = ''
    gpu_last_validated_provider: str = ''
    auto_check_updates: bool = True
    theme_mode: str = 'dark'
    ui_scale_percent: int = 100
    download_location: str = str(OUTPUT_MIDI_DIR)
    window_geometry: str = ''

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'AppConfig':
        if not isinstance(payload, dict):
            return default_config()
        try:
            schema_version = int(payload.get('schema_version', 0) or 0)
        except Exception:
            schema_version = 0
        if schema_version not in {1, CONFIG_SCHEMA_VERSION}:
            return default_config()
        pref = str(payload.get('device_preference', 'cpu')).strip().lower()
        if pref not in {'cpu', 'gpu'}:
            pref = 'cpu'
        conversion_method = normalize_conversion_method(payload.get('conversion_method', 'legacy_v1'))
        modern_adaptive_thresholds_enabled = _coerce_bool(payload.get('modern_adaptive_thresholds_enabled', True), default=True)
        modern_input_normalization_enabled = _coerce_bool(payload.get('modern_input_normalization_enabled', True), default=True)
        modern_smart_overlap_stitching_enabled = _coerce_bool(payload.get('modern_smart_overlap_stitching_enabled', True), default=True)
        modern_auto_calibration_enabled = _coerce_bool(payload.get('modern_auto_calibration_enabled', True), default=True)
        modern_cleanup_scale = _normalize_modern_scale(payload.get('modern_cleanup_scale', 1.0), default=1.0)
        modern_pedal_cluster_scale = _normalize_modern_scale(payload.get('modern_pedal_cluster_scale', 1.0), default=1.0)
        modern_alignment_gate_scale = _normalize_modern_scale(payload.get('modern_alignment_gate_scale', 1.0), default=1.0)
        modern_manual_onset_threshold = _normalize_modern_threshold(payload.get('modern_manual_onset_threshold', MODERN_ONSET_THRESHOLD_DEFAULT), lower=MODERN_ONSET_THRESHOLD_MIN, upper=MODERN_ONSET_THRESHOLD_MAX, default=MODERN_ONSET_THRESHOLD_DEFAULT)
        modern_manual_offset_threshold = _normalize_modern_threshold(payload.get('modern_manual_offset_threshold', MODERN_OFFSET_THRESHOLD_DEFAULT), lower=MODERN_OFFSET_THRESHOLD_MIN, upper=MODERN_OFFSET_THRESHOLD_MAX, default=MODERN_OFFSET_THRESHOLD_DEFAULT)
        modern_manual_frame_threshold = _normalize_modern_threshold(payload.get('modern_manual_frame_threshold', MODERN_FRAME_THRESHOLD_DEFAULT), lower=MODERN_FRAME_THRESHOLD_MIN, upper=MODERN_FRAME_THRESHOLD_MAX, default=MODERN_FRAME_THRESHOLD_DEFAULT)
        modern_manual_pedal_offset_threshold = _normalize_modern_threshold(payload.get('modern_manual_pedal_offset_threshold', MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT), lower=MODERN_PEDAL_OFFSET_THRESHOLD_MIN, upper=MODERN_PEDAL_OFFSET_THRESHOLD_MAX, default=MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT)
        try:
            batch = int(payload.get('gpu_batch_size', 4))
        except Exception:
            batch = 4
        batch = max(GPU_BATCH_SIZE_MIN, min(GPU_BATCH_SIZE_MAX, batch))
        provider_pref = str(payload.get('gpu_provider_preference', 'auto') or '').strip().lower()
        if provider_pref not in {'auto', 'cuda', 'dml'}:
            provider_pref = 'auto'
        runtime_enabled = _coerce_bool(payload.get('gpu_runtime_enabled', False), default=False)
        runtime_path = str(payload.get('gpu_runtime_path', '') or '').strip()
        gpu_last_reason_code = str(payload.get('gpu_last_reason_code', '') or '').strip()
        gpu_last_reason_text = str(payload.get('gpu_last_reason_text', '') or '').strip()
        gpu_last_validated_provider = str(payload.get('gpu_last_validated_provider', '') or '').strip().lower()
        if gpu_last_validated_provider not in {'', 'cuda', 'dml', 'cpu'}:
            gpu_last_validated_provider = ''
        if not runtime_path:
            runtime_enabled = False
        theme_mode = str(payload.get('theme_mode', 'dark')).strip().lower()
        if theme_mode not in {'dark', 'light'}:
            theme_mode = 'dark'
        try:
            ui_scale_percent = int(payload.get('ui_scale_percent', 100))
        except Exception:
            ui_scale_percent = 100
        ui_scale_percent = max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, ui_scale_percent))
        default_download_location = str(OUTPUT_MIDI_DIR)
        download_location = str(payload.get('download_location', default_download_location) or '').strip()
        if not download_location:
            download_location = default_download_location
        return cls(schema_version=CONFIG_SCHEMA_VERSION, device_preference=pref, conversion_method=conversion_method, modern_adaptive_thresholds_enabled=modern_adaptive_thresholds_enabled, modern_input_normalization_enabled=modern_input_normalization_enabled, modern_smart_overlap_stitching_enabled=modern_smart_overlap_stitching_enabled, modern_auto_calibration_enabled=modern_auto_calibration_enabled, modern_cleanup_scale=modern_cleanup_scale, modern_pedal_cluster_scale=modern_pedal_cluster_scale, modern_alignment_gate_scale=modern_alignment_gate_scale, modern_manual_onset_threshold=modern_manual_onset_threshold, modern_manual_offset_threshold=modern_manual_offset_threshold, modern_manual_frame_threshold=modern_manual_frame_threshold, modern_manual_pedal_offset_threshold=modern_manual_pedal_offset_threshold, gpu_batch_size=batch, gpu_provider_preference=provider_pref, gpu_runtime_enabled=runtime_enabled, gpu_runtime_path=runtime_path, gpu_last_reason_code=gpu_last_reason_code, gpu_last_reason_text=gpu_last_reason_text, gpu_last_validated_provider=gpu_last_validated_provider, auto_check_updates=_coerce_bool(payload.get('auto_check_updates', True), default=True), theme_mode=theme_mode, ui_scale_percent=ui_scale_percent, download_location=download_location, window_geometry=str(payload.get('window_geometry', '') or ''))

    def to_dict(self) -> dict[str, Any]:
        provider_pref = str(self.gpu_provider_preference or 'auto').strip().lower()
        if provider_pref not in {'auto', 'cuda', 'dml'}:
            provider_pref = 'auto'
        return {'schema_version': CONFIG_SCHEMA_VERSION, 'device_preference': self.device_preference, 'conversion_method': normalize_conversion_method(self.conversion_method), 'modern_adaptive_thresholds_enabled': bool(self.modern_adaptive_thresholds_enabled), 'modern_input_normalization_enabled': bool(self.modern_input_normalization_enabled), 'modern_smart_overlap_stitching_enabled': bool(self.modern_smart_overlap_stitching_enabled), 'modern_auto_calibration_enabled': bool(self.modern_auto_calibration_enabled), 'modern_cleanup_scale': _normalize_modern_scale(self.modern_cleanup_scale, default=1.0), 'modern_pedal_cluster_scale': _normalize_modern_scale(self.modern_pedal_cluster_scale, default=1.0), 'modern_alignment_gate_scale': _normalize_modern_scale(self.modern_alignment_gate_scale, default=1.0), 'modern_manual_onset_threshold': _normalize_modern_threshold(self.modern_manual_onset_threshold, lower=MODERN_ONSET_THRESHOLD_MIN, upper=MODERN_ONSET_THRESHOLD_MAX, default=MODERN_ONSET_THRESHOLD_DEFAULT), 'modern_manual_offset_threshold': _normalize_modern_threshold(self.modern_manual_offset_threshold, lower=MODERN_OFFSET_THRESHOLD_MIN, upper=MODERN_OFFSET_THRESHOLD_MAX, default=MODERN_OFFSET_THRESHOLD_DEFAULT), 'modern_manual_frame_threshold': _normalize_modern_threshold(self.modern_manual_frame_threshold, lower=MODERN_FRAME_THRESHOLD_MIN, upper=MODERN_FRAME_THRESHOLD_MAX, default=MODERN_FRAME_THRESHOLD_DEFAULT), 'modern_manual_pedal_offset_threshold': _normalize_modern_threshold(self.modern_manual_pedal_offset_threshold, lower=MODERN_PEDAL_OFFSET_THRESHOLD_MIN, upper=MODERN_PEDAL_OFFSET_THRESHOLD_MAX, default=MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT), 'gpu_batch_size': int(self.gpu_batch_size), 'gpu_provider_preference': provider_pref, 'gpu_runtime_enabled': bool(self.gpu_runtime_enabled), 'gpu_runtime_path': self.gpu_runtime_path or '', 'gpu_last_reason_code': self.gpu_last_reason_code or '', 'gpu_last_reason_text': self.gpu_last_reason_text or '', 'gpu_last_validated_provider': self.gpu_last_validated_provider or '', 'auto_check_updates': bool(self.auto_check_updates), 'theme_mode': self.theme_mode, 'ui_scale_percent': int(self.ui_scale_percent), 'download_location': str(self.download_location or '').strip() or str(OUTPUT_MIDI_DIR), 'window_geometry': self.window_geometry or ''}


def normalize_conversion_method(value: Any) -> str:
    normalized = str(value or 'legacy_v1').strip().lower()
    if normalized in {'legacy_v1', 'modern'}:
        return normalized
    return 'legacy_v1'


def _normalize_modern_threshold(value: Any, *, lower: float, upper: float, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not (parsed == parsed):
        parsed = float(default)
    return max(float(lower), min(float(upper), float(parsed)))


def _normalize_modern_scale(value: Any, *, default: float=1.0, lower: float=0.6, upper: float=2.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not (parsed == parsed):
        parsed = float(default)
    return max(float(lower), min(float(upper), float(parsed)))


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

def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None

def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        return True
    except Exception:
        return False

def load_config() -> AppConfig:
    paths = config_read_paths()
    config = default_config()
    candidates: list[tuple[int, int, Path, AppConfig]] = []
    for index, path in enumerate(paths):
        if not path.exists():
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        try:
            mtime_ns = path.stat().st_mtime_ns
        except Exception:
            mtime_ns = 0
        candidates.append((mtime_ns, -index, path, AppConfig.from_dict(payload)))
    loaded_from: Path | None = None
    if candidates:
        _, _, loaded_from, config = max(candidates, key=lambda item: (item[0], item[1]))
    if loaded_from is None:
        save_config(config)
        return config
    preferred = paths[0] if paths else None
    if preferred is not None and preferred != loaded_from:
        save_config(config)
    return config

def save_config(config: AppConfig) -> str | None:
    payload = config.to_dict()
    attempted: list[Path] = []
    for path in config_write_paths():
        attempted.append(path)
        if _write_json(path, payload):
            return str(path)
    if attempted:
        joined = ', '.join((str(p) for p in attempted))
        print(f'[A2M] Warning: failed to write config to any location: {joined}', file=sys.stderr)
    return None
