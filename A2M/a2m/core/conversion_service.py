from __future__ import annotations
import importlib
import os
import re
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, Iterator
import numpy as np
from piano_transcription_inference import config as pti_config
from piano_transcription_inference.utilities import RegressionPostProcessor, write_events_to_midi
from .constants import MODERN_FRAME_THRESHOLD_DEFAULT, MODERN_FRAME_THRESHOLD_MAX, MODERN_FRAME_THRESHOLD_MIN
from .constants import MODERN_OFFSET_THRESHOLD_DEFAULT, MODERN_OFFSET_THRESHOLD_MAX, MODERN_OFFSET_THRESHOLD_MIN
from .constants import MODERN_ONSET_THRESHOLD_DEFAULT, MODERN_ONSET_THRESHOLD_MAX, MODERN_ONSET_THRESHOLD_MIN
from .constants import MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT, MODERN_PEDAL_OFFSET_THRESHOLD_MAX, MODERN_PEDAL_OFFSET_THRESHOLD_MIN
from .constants import OUTPUT_MIDI_DIR
from .model_service import get_existing_model_path
from . import runtime_service
try:
    from pathvalidate import sanitize_filename
except Exception:

    def sanitize_filename(name: str) -> str:
        return re.sub('[<>:"/\\\\|?*]', '', name)
WINDOWS_RESERVED_NAMES = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
SegmentProgressCallback = Callable[[int, int], None]
_LIBROSA_MODULE = None
_PRETTY_MIDI_MODULE = None
_PRETTY_MIDI_IMPORT_ERROR: str | None = None
_PRETTY_MIDI_WARNING_EMITTED = False
_ONNX_SESSION = None
_ONNX_SESSION_SIGNATURE: tuple[str, str, str, str] | None = None
_ONNX_OUTPUT_MAP: dict[str, str] = {}
_ACTIVE_PROVIDER_LABEL = 'CPU'
_LAST_MODERN_DIAGNOSTICS_TEXT = ''
_OUTPUT_MIDI_DIR = Path(OUTPUT_MIDI_DIR)
_STOPPED_ERROR = 'Transcription stopped by user.'
_MODEL_MISSING_ERROR = 'ONNX model file is missing. Please download it first.'
_SEGMENT_SECONDS = 10
_LOW_REGISTER_MAX_MIDI = 48
_HIGH_REGISTER_MIN_MIDI = 73
_MID_HIGH_REGISTER_MIN_MIDI = 53
_MID_HIGH_REGISTER_MAX_MIDI = 76
_LEGACY_ONSET_THRESHOLD = 0.3
_LEGACY_OFFSET_THRESHOLD = 0.3
_LEGACY_FRAME_THRESHOLD = 0.1
_LEGACY_PEDAL_OFFSET_THRESHOLD = 0.2
_MODERN_ALIGNMENT_MAX_SHIFT_SECONDS = 0.12
_MODERN_ALIGNMENT_MIN_APPLY_SECONDS = 0.008
_MODERN_ALIGNMENT_MIN_MEDIAN_IMPROVEMENT_SECONDS = 0.010
_MODERN_ALIGNMENT_MAX_P90_DEGRADATION_SECONDS = 0.015
_MODERN_ALIGNMENT_MIN_CONFIDENCE = 0.08
_MODERN_ALIGNMENT_MAX_DRIFT_WORSEN_SECONDS_PER_MIN = 0.012
_MODERN_ALIGNMENT_MAX_WINDOW_MEDIAN_WORSEN_SECONDS = 0.005
_MODERN_ALIGNMENT_MAX_ABS_DRIFT_SECONDS_PER_MIN = 0.080


@dataclass(slots=True, frozen=True)
class ConversionOptions:
    conversion_method: str = 'legacy_v1'
    modern_adaptive_thresholds_enabled: bool = True
    modern_input_normalization_enabled: bool = True
    modern_smart_overlap_stitching_enabled: bool = True
    modern_auto_calibration_enabled: bool = True
    modern_threshold_bias_scale: float = 1.0
    modern_cleanup_scale: float = 1.0
    modern_pedal_cluster_scale: float = 1.0
    modern_alignment_gate_scale: float = 1.0
    modern_manual_onset_threshold: float = MODERN_ONSET_THRESHOLD_DEFAULT
    modern_manual_offset_threshold: float = MODERN_OFFSET_THRESHOLD_DEFAULT
    modern_manual_frame_threshold: float = MODERN_FRAME_THRESHOLD_DEFAULT
    modern_manual_pedal_offset_threshold: float = MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT


def _normalize_conversion_method(value: str | None) -> str:
    normalized = str(value or 'legacy_v1').strip().lower()
    if normalized in {'legacy_v1', 'modern'}:
        return normalized
    return 'legacy_v1'


def _normalize_conversion_options(conversion_options: ConversionOptions | None) -> ConversionOptions:
    if isinstance(conversion_options, ConversionOptions):
        return ConversionOptions(
            conversion_method=_normalize_conversion_method(conversion_options.conversion_method),
            modern_adaptive_thresholds_enabled=bool(conversion_options.modern_adaptive_thresholds_enabled),
            modern_input_normalization_enabled=bool(conversion_options.modern_input_normalization_enabled),
            modern_smart_overlap_stitching_enabled=bool(conversion_options.modern_smart_overlap_stitching_enabled),
            modern_auto_calibration_enabled=bool(conversion_options.modern_auto_calibration_enabled),
            modern_threshold_bias_scale=_normalize_modern_scale(conversion_options.modern_threshold_bias_scale),
            modern_cleanup_scale=_normalize_modern_scale(conversion_options.modern_cleanup_scale),
            modern_pedal_cluster_scale=_normalize_modern_scale(conversion_options.modern_pedal_cluster_scale),
            modern_alignment_gate_scale=_normalize_modern_scale(conversion_options.modern_alignment_gate_scale),
            modern_manual_onset_threshold=_normalize_modern_threshold_value(conversion_options.modern_manual_onset_threshold, lower=MODERN_ONSET_THRESHOLD_MIN, upper=MODERN_ONSET_THRESHOLD_MAX, default=MODERN_ONSET_THRESHOLD_DEFAULT),
            modern_manual_offset_threshold=_normalize_modern_threshold_value(conversion_options.modern_manual_offset_threshold, lower=MODERN_OFFSET_THRESHOLD_MIN, upper=MODERN_OFFSET_THRESHOLD_MAX, default=MODERN_OFFSET_THRESHOLD_DEFAULT),
            modern_manual_frame_threshold=_normalize_modern_threshold_value(conversion_options.modern_manual_frame_threshold, lower=MODERN_FRAME_THRESHOLD_MIN, upper=MODERN_FRAME_THRESHOLD_MAX, default=MODERN_FRAME_THRESHOLD_DEFAULT),
            modern_manual_pedal_offset_threshold=_normalize_modern_threshold_value(conversion_options.modern_manual_pedal_offset_threshold, lower=MODERN_PEDAL_OFFSET_THRESHOLD_MIN, upper=MODERN_PEDAL_OFFSET_THRESHOLD_MAX, default=MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT),
        )
    return ConversionOptions()


def _is_modern_mode(conversion_options: ConversionOptions) -> bool:
    return _normalize_conversion_method(conversion_options.conversion_method) == 'modern'


def _modern_input_processing_enabled(conversion_options: ConversionOptions) -> bool:
    return _is_modern_mode(conversion_options) and bool(conversion_options.modern_input_normalization_enabled)


def _modern_overlap_stitching_enabled(conversion_options: ConversionOptions) -> bool:
    return _is_modern_mode(conversion_options) and bool(conversion_options.modern_smart_overlap_stitching_enabled)


def _modern_adaptive_thresholds_enabled(conversion_options: ConversionOptions) -> bool:
    return _is_modern_mode(conversion_options) and bool(conversion_options.modern_adaptive_thresholds_enabled)


def _modern_auto_calibration_enabled(conversion_options: ConversionOptions) -> bool:
    return _is_modern_mode(conversion_options) and bool(conversion_options.modern_auto_calibration_enabled)


def _normalize_modern_threshold_value(value: object, *, lower: float, upper: float, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not np.isfinite(parsed):
        parsed = float(default)
    return max(float(lower), min(float(upper), float(parsed)))


def _normalize_modern_scale(value: object, *, default: float=1.0, lower: float=0.6, upper: float=2.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not np.isfinite(parsed):
        parsed = float(default)
    return max(float(lower), min(float(upper), float(parsed)))


def _set_last_modern_diagnostics_text(text: str) -> None:
    global _LAST_MODERN_DIAGNOSTICS_TEXT
    _LAST_MODERN_DIAGNOSTICS_TEXT = str(text or '').strip()


def get_last_modern_diagnostics_text() -> str:
    return str(_LAST_MODERN_DIAGNOSTICS_TEXT or '')

def get_librosa_module():
    global _LIBROSA_MODULE
    if _LIBROSA_MODULE is None:
        _LIBROSA_MODULE = importlib.import_module('librosa')
    return _LIBROSA_MODULE

def get_pretty_midi_module():
    global _PRETTY_MIDI_MODULE, _PRETTY_MIDI_IMPORT_ERROR
    if _PRETTY_MIDI_MODULE is None:
        try:
            original_add_dll_directory = getattr(os, 'add_dll_directory', None)
            if callable(original_add_dll_directory):

                class _NoopDllHandle:
                    def close(self) -> None:
                        return

                def _safe_add_dll_directory(path: str):
                    try:
                        return original_add_dll_directory(path)
                    except FileNotFoundError:
                        return _NoopDllHandle()

                setattr(os, 'add_dll_directory', _safe_add_dll_directory)
            try:
                _PRETTY_MIDI_MODULE = importlib.import_module('pretty_midi')
                _PRETTY_MIDI_IMPORT_ERROR = None
            finally:
                if callable(original_add_dll_directory):
                    setattr(os, 'add_dll_directory', original_add_dll_directory)
        except Exception as exc:
            _PRETTY_MIDI_MODULE = False
            _PRETTY_MIDI_IMPORT_ERROR = str(exc)
    if _PRETTY_MIDI_MODULE is False:
        return None
    return _PRETTY_MIDI_MODULE

def _ensure_not_stopped(stop_event: Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError(_STOPPED_ERROR)

def reset_transcriptor() -> None:
    global _ONNX_SESSION, _ONNX_SESSION_SIGNATURE, _ONNX_OUTPUT_MAP, _ACTIVE_PROVIDER_LABEL, _LAST_MODERN_DIAGNOSTICS_TEXT
    _ONNX_SESSION = None
    _ONNX_SESSION_SIGNATURE = None
    _ONNX_OUTPUT_MAP = {}
    _ACTIVE_PROVIDER_LABEL = 'CPU'
    _LAST_MODERN_DIAGNOSTICS_TEXT = ''

def get_active_provider_label() -> str:
    return str(_ACTIVE_PROVIDER_LABEL or 'CPU')

def safe_filename(name: str, fallback: str='audio', max_length: int=180) -> str:
    cleaned = sanitize_filename(name or '').strip().strip('.')
    if not cleaned or cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = fallback
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned or fallback

def ensure_unique_path(path: Path | str) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    base = path.with_suffix('')
    ext = path.suffix
    for i in range(2, 1000):
        candidate = Path(f'{base}_{i}{ext}')
        if not candidate.exists():
            return candidate
    raise RuntimeError(f'Unable to find a unique filename for {path}.')

def _normalize_output_midi_dir(path: Path | str | None) -> Path:
    candidate = str(path or '').strip()
    if not candidate:
        return Path(OUTPUT_MIDI_DIR)
    return Path(candidate).expanduser()

def set_output_midi_dir(path: Path | str | None) -> Path:
    global _OUTPUT_MIDI_DIR
    _OUTPUT_MIDI_DIR = _normalize_output_midi_dir(path)
    return Path(_OUTPUT_MIDI_DIR)

def get_output_midi_dir() -> Path:
    target_dir = Path(_OUTPUT_MIDI_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

def _resolve_model_path(model_path_override: Path | str | None) -> Path:
    model_path = Path(model_path_override) if model_path_override else get_existing_model_path()
    if not model_path:
        raise RuntimeError(_MODEL_MISSING_ERROR)
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(f'ONNX model file was not found: {model_path}')
    return model_path

def _resolve_transcription_runtime(*, model_path_override: Path | str | None, device_override: str | None):
    model_path = _resolve_model_path(model_path_override)
    device_preference, gpu_provider_preference = _resolve_device_overrides(device_override)
    session, output_map, _provider = _get_or_create_session(model_path, device_preference=device_preference, gpu_provider_preference=gpu_provider_preference)
    return (model_path, session, output_map, device_preference)

def _resolve_device_overrides(device_override: str | None) -> tuple[str, str]:
    normalized = str(device_override or '').strip().lower()
    if not normalized or normalized == 'auto':
        return (runtime_service.get_device_preference(), runtime_service.get_gpu_provider_preference())
    if normalized == 'cpu':
        return ('cpu', runtime_service.get_gpu_provider_preference())
    if normalized == 'gpu':
        return ('gpu', runtime_service.get_gpu_provider_preference())
    if normalized == 'cuda':
        return ('gpu', 'cuda')
    if normalized in {'dml', 'directml'}:
        return ('gpu', 'dml')
    return (runtime_service.get_device_preference(), runtime_service.get_gpu_provider_preference())

def _session_signature(model_path: Path, *, device_preference: str, gpu_provider_preference: str) -> tuple[str, str, str, str]:
    runtime_path = runtime_service.get_gpu_runtime_path() if runtime_service.is_gpu_runtime_enabled() else ''
    return (str(model_path.resolve()), str(device_preference), str(gpu_provider_preference), runtime_path)

def _normalize_output_name(name: str) -> str:
    return ''.join((ch for ch in str(name).lower() if ch.isalnum()))

def _match_output_name(normalized_names: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        normalized = _normalize_output_name(candidate)
        for output_name, output_norm in normalized_names.items():
            if output_norm == normalized:
                return output_name
        for output_name, output_norm in normalized_names.items():
            if normalized in output_norm:
                return output_name
    return None

def _build_output_map(session) -> dict[str, str]:
    output_names = [str(meta.name) for meta in session.get_outputs()]
    normalized = {name: _normalize_output_name(name) for name in output_names}
    mapping: dict[str, str] = {}
    expected = {'reg_onset_output': ['reg_onset_output', 'regonsetoutput', 'onsetregression'], 'reg_offset_output': ['reg_offset_output', 'regoffsetoutput', 'offsetregression'], 'frame_output': ['frame_output', 'frameoutput'], 'velocity_output': ['velocity_output', 'velocityoutput'], 'reg_pedal_onset_output': ['reg_pedal_onset_output', 'regpedalonsetoutput'], 'reg_pedal_offset_output': ['reg_pedal_offset_output', 'regpedaloffsetoutput'], 'pedal_frame_output': ['pedal_frame_output', 'pedalframeoutput']}
    for key, aliases in expected.items():
        matched = _match_output_name(normalized, aliases)
        if matched:
            mapping[key] = matched
    required = ['reg_onset_output', 'reg_offset_output', 'frame_output', 'velocity_output']
    missing = [key for key in required if key not in mapping]
    if missing:
        raise RuntimeError(f"ONNX model outputs are incompatible. Missing required outputs: {', '.join(missing)}. Available outputs: {', '.join(output_names)}")
    return mapping

def _get_or_create_session(model_path: Path, *, device_preference: str, gpu_provider_preference: str):
    global _ONNX_SESSION, _ONNX_SESSION_SIGNATURE, _ONNX_OUTPUT_MAP, _ACTIVE_PROVIDER_LABEL
    signature = _session_signature(model_path, device_preference=device_preference, gpu_provider_preference=gpu_provider_preference)
    if _ONNX_SESSION is None or _ONNX_SESSION_SIGNATURE != signature:
        _ONNX_SESSION, _ACTIVE_PROVIDER_LABEL = runtime_service.create_session(model_path, device_preference=device_preference, gpu_provider_preference=gpu_provider_preference)
        _ONNX_OUTPUT_MAP = _build_output_map(_ONNX_SESSION)
        _ONNX_SESSION_SIGNATURE = signature
    return (_ONNX_SESSION, _ONNX_OUTPUT_MAP, _ACTIVE_PROVIDER_LABEL)

def _count_overlapping_segments(total_samples: int, segment_samples: int) -> int:
    if segment_samples <= 0:
        raise RuntimeError('Segment size must be greater than zero.')
    normalized_total = max(segment_samples, int(total_samples))
    hop_samples = max(1, segment_samples // 2)
    return 1 + max(0, (normalized_total - segment_samples) // hop_samples)

def _iter_overlapping_segments(audio: np.ndarray, segment_samples: int) -> Iterator[np.ndarray]:
    if segment_samples <= 0:
        raise RuntimeError('Segment size must be greater than zero.')
    if audio.ndim != 2:
        raise RuntimeError('Audio batch must be a 2D array.')
    if audio.shape[1] % segment_samples != 0:
        raise RuntimeError('Audio length must align with segment size before enframe.')
    hop_samples = max(1, segment_samples // 2)
    pointer = 0
    while pointer + segment_samples <= audio.shape[1]:
        yield audio[:, pointer:pointer + segment_samples]
        pointer += hop_samples

def _target_frames_for_audio_length(audio_len_samples: int) -> int:
    return int(np.ceil(int(audio_len_samples) / float(pti_config.sample_rate) * float(pti_config.frames_per_second)))

def _trim_outputs_to_target_frames(outputs: dict[str, np.ndarray], target_frames: int, *, smart_overlap_stitching: bool=False) -> dict[str, np.ndarray]:
    onset_weight_map = None
    if smart_overlap_stitching:
        onset_weight_map = _build_overlap_weight_map_from_onset(outputs.get('reg_onset_output'))
    for key in list(outputs.keys()):
        if smart_overlap_stitching:
            outputs[key] = _deframe_overlap_weighted(outputs[key], weight_map=onset_weight_map)[0:target_frames]
        else:
            outputs[key] = _deframe(outputs[key])[0:target_frames]
    return outputs


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _to_finite_float_array(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.asarray([], dtype=np.float32)
    return finite


def _percentile_or_default(values: np.ndarray | list[float], percentile: float, default: float) -> float:
    finite = _to_finite_float_array(values)
    if finite.size == 0:
        return float(default)
    return float(np.percentile(finite, float(percentile)))


def _smooth_per_key(values: np.ndarray, passes: int=1) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 2:
        return arr
    smoothed = arr
    for _ in range(max(1, int(passes))):
        padded = np.pad(smoothed, (1, 1), mode='edge')
        smoothed = padded[:-2] * 0.25 + padded[1:-1] * 0.5 + padded[2:] * 0.25
    return smoothed.astype(np.float32, copy=False)


def _quantile_per_key(values: np.ndarray | None, quantile: float, default: float) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 0:
        return None
    finite = np.where(np.isfinite(arr), arr, np.nan)
    try:
        quant = np.nanpercentile(finite, float(quantile), axis=0)
    except Exception:
        return None
    quant = np.nan_to_num(quant, nan=float(default), posinf=float(default), neginf=float(default))
    return np.asarray(quant, dtype=np.float32).reshape(-1)


def _as_per_key_bool_mask(mask: np.ndarray | None, size: int) -> np.ndarray:
    out = np.zeros((max(0, int(size)),), dtype=bool)
    if mask is None:
        return out
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size <= 0:
        return out
    copy_len = min(out.size, arr.size)
    out[:copy_len] = arr[:copy_len]
    return out


def _high_register_low_energy_mask(frame_values: np.ndarray | None, size: int) -> np.ndarray:
    if size <= 0:
        return np.zeros((0,), dtype=bool)
    out = np.zeros((int(size),), dtype=bool)
    if frame_values is None:
        return out
    peak = _quantile_per_key(frame_values, 92.0, 0.0)
    if peak is None or peak.size <= 0:
        return out
    begin_note = int(getattr(pti_config, 'begin_note', 21))
    midi_notes = begin_note + np.arange(int(size), dtype=np.int32)
    high_mask = midi_notes >= int(_HIGH_REGISTER_MIN_MIDI)
    aligned_peak = np.asarray(peak, dtype=np.float32).reshape(-1)
    if aligned_peak.size < int(size):
        aligned = np.pad(aligned_peak, (0, int(size) - aligned_peak.size), mode='edge')
    else:
        aligned = aligned_peak[:int(size)]
    finite_high = aligned[high_mask]
    finite_high = finite_high[np.isfinite(finite_high)]
    if finite_high.size <= 0:
        return out
    cutoff = float(np.percentile(finite_high, 55.0))
    out = high_mask & np.isfinite(aligned) & (aligned <= cutoff)
    return np.asarray(out, dtype=bool)


def _apply_masked_boost(values: np.ndarray, mask: np.ndarray, delta: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    if arr.size <= 0:
        return arr
    bool_mask = _as_per_key_bool_mask(mask, arr.size)
    if np.any(bool_mask):
        arr[bool_mask] += float(delta)
    return arr.astype(np.float32, copy=False)


def _estimate_frame_noise_pressure(frame_values: np.ndarray | None) -> float:
    if frame_values is None:
        return 0.0
    arr = np.asarray(frame_values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 4 or arr.shape[1] <= 0:
        return 0.0
    floor_per_key = _quantile_per_key(arr, 20.0, 0.0)
    peak_per_key = _quantile_per_key(arr, 92.0, 0.0)
    if floor_per_key is None or peak_per_key is None:
        return 0.0
    floor = np.asarray(floor_per_key, dtype=np.float32).reshape(-1)
    peak = np.asarray(peak_per_key, dtype=np.float32).reshape(-1)
    if floor.size <= 0 or peak.size <= 0:
        return 0.0
    size = min(floor.size, peak.size)
    floor = floor[:size]
    peak = peak[:size]
    dyn = np.maximum(0.0, peak - floor)
    finite_floor = floor[np.isfinite(floor)]
    finite_dyn = dyn[np.isfinite(dyn)]
    if finite_floor.size <= 0 or finite_dyn.size <= 0:
        return 0.0
    floor_med = float(np.median(finite_floor))
    dyn_med = float(np.median(finite_dyn))
    floor_score = _clamp((floor_med - 0.045) / 0.08, 0.0, 1.0)
    dyn_score = _clamp((0.17 - dyn_med) / 0.12, 0.0, 1.0)
    return float(_clamp(floor_score + (0.8 * dyn_score), 0.0, 1.4))


def _resolve_post_processor_thresholds(outputs: dict[str, np.ndarray], conversion_options: ConversionOptions) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray, float]:
    onset_threshold = float(MODERN_ONSET_THRESHOLD_DEFAULT)
    offset_threshold = float(MODERN_OFFSET_THRESHOLD_DEFAULT)
    frame_threshold = float(MODERN_FRAME_THRESHOLD_DEFAULT)
    pedal_offset_threshold = float(MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT)
    manual_onset_threshold = float(conversion_options.modern_manual_onset_threshold)
    manual_offset_threshold = float(conversion_options.modern_manual_offset_threshold)
    manual_frame_threshold = float(conversion_options.modern_manual_frame_threshold)
    manual_pedal_offset_threshold = float(conversion_options.modern_manual_pedal_offset_threshold)
    if not _is_modern_mode(conversion_options):
        return (
            float(_LEGACY_ONSET_THRESHOLD),
            float(_LEGACY_OFFSET_THRESHOLD),
            float(_LEGACY_FRAME_THRESHOLD),
            float(_LEGACY_PEDAL_OFFSET_THRESHOLD),
        )
    if not _modern_auto_calibration_enabled(conversion_options):
        return (
            manual_onset_threshold,
            manual_offset_threshold,
            manual_frame_threshold,
            manual_pedal_offset_threshold,
        )
    if not _modern_adaptive_thresholds_enabled(conversion_options):
        return (onset_threshold, offset_threshold, frame_threshold, pedal_offset_threshold)
    profile_scale = _normalize_modern_scale(conversion_options.modern_threshold_bias_scale)
    onset_vals = outputs.get('reg_onset_output')
    offset_vals = outputs.get('reg_offset_output')
    frame_vals = outputs.get('frame_output')
    pedal_vals = outputs.get('reg_pedal_offset_output')
    noise_pressure = _estimate_frame_noise_pressure(frame_vals)
    onset_base = _quantile_per_key(onset_vals, 90.0, onset_threshold)
    onset_peak = _quantile_per_key(onset_vals, 98.5, onset_threshold)
    onset_per_key = None
    if onset_base is not None:
        onset_per_key = np.asarray(onset_base, dtype=np.float32)
        if onset_peak is not None:
            peak_aligned = np.asarray(onset_peak, dtype=np.float32)
            if peak_aligned.size == onset_per_key.size:
                onset_per_key = onset_per_key + np.maximum(0.0, peak_aligned - onset_per_key) * 0.45
    if onset_per_key is not None:
        onset_per_key = _smooth_per_key(onset_per_key, passes=1)
        onset_per_key = _apply_register_threshold_bias(
            onset_per_key,
            low_delta=float(0.035 * profile_scale),
            high_delta=float(0.028 * profile_scale),
            mid_delta=float(0.014 * profile_scale),
        )
        if float(np.median(onset_per_key)) <= 0.185:
            onset_per_key += 0.03
        onset_threshold = np.clip(onset_per_key, 0.16, 0.68).astype(np.float32, copy=False)
    offset_base = _quantile_per_key(offset_vals, 88.0, offset_threshold)
    offset_peak = _quantile_per_key(offset_vals, 97.0, offset_threshold)
    offset_per_key = None
    if offset_base is not None:
        offset_per_key = np.asarray(offset_base, dtype=np.float32)
        if offset_peak is not None:
            peak_aligned = np.asarray(offset_peak, dtype=np.float32)
            if peak_aligned.size == offset_per_key.size:
                offset_per_key = offset_per_key + np.maximum(0.0, peak_aligned - offset_per_key) * 0.4
    if offset_per_key is not None:
        offset_per_key = _smooth_per_key(offset_per_key, passes=1)
        if float(np.median(offset_per_key)) <= 0.19:
            offset_per_key += 0.02
        if noise_pressure > 0.0:
            offset_per_key += float(0.018 * noise_pressure * profile_scale)
        offset_threshold = np.clip(offset_per_key, 0.16, 0.62).astype(np.float32, copy=False)
    if frame_vals is not None and np.asarray(frame_vals).ndim == 2:
        frame_low = _quantile_per_key(frame_vals, 55.0, 0.05)
        frame_high = _quantile_per_key(frame_vals, 96.0, 0.35)
        if frame_low is not None and frame_high is not None:
            frame_per_key = frame_low + (frame_high - frame_low) * 0.22
            frame_per_key = _smooth_per_key(frame_per_key, passes=1)
            frame_per_key = _apply_register_threshold_bias(
                frame_per_key,
                low_delta=float(0.012 * profile_scale),
                high_delta=float(0.01 * profile_scale),
                mid_delta=float(0.008 * profile_scale),
            )
            if noise_pressure > 0.0:
                frame_per_key += float(0.012 * noise_pressure * profile_scale)
            frame_threshold = np.clip(frame_per_key, 0.04, 0.32).astype(np.float32, copy=False)
    elif noise_pressure > 0.0:
        frame_threshold = float(np.clip(float(frame_threshold) + (0.01 * noise_pressure * profile_scale), 0.04, 0.32))
    if isinstance(onset_threshold, np.ndarray):
        high_low_energy = _high_register_low_energy_mask(frame_vals, int(onset_threshold.size))
        onset_threshold = np.clip(_apply_masked_boost(onset_threshold, high_low_energy, 0.04), 0.16, 0.70).astype(np.float32, copy=False)
        if noise_pressure > 0.0:
            onset_threshold = _apply_register_threshold_bias(
                onset_threshold,
                low_delta=0.0,
                high_delta=float(0.018 * profile_scale * _clamp(noise_pressure, 0.0, 1.4)),
                mid_delta=float(0.008 * profile_scale * _clamp(noise_pressure, 0.0, 1.4)),
            )
            onset_threshold = np.clip(onset_threshold, 0.16, 0.72).astype(np.float32, copy=False)
    if isinstance(offset_threshold, np.ndarray):
        high_low_energy = _high_register_low_energy_mask(frame_vals, int(offset_threshold.size))
        offset_threshold = np.clip(_apply_masked_boost(offset_threshold, high_low_energy, 0.025), 0.16, 0.64).astype(np.float32, copy=False)
        if noise_pressure > 0.0:
            offset_threshold = _apply_register_threshold_bias(
                offset_threshold,
                low_delta=0.0,
                high_delta=float(0.012 * profile_scale * _clamp(noise_pressure, 0.0, 1.4)),
                mid_delta=float(0.006 * profile_scale * _clamp(noise_pressure, 0.0, 1.4)),
            )
            offset_threshold = np.clip(offset_threshold, 0.16, 0.66).astype(np.float32, copy=False)
    elif noise_pressure > 0.0:
        offset_threshold = float(np.clip(float(offset_threshold) + (0.014 * noise_pressure * profile_scale), 0.16, 0.64))
    if isinstance(frame_threshold, np.ndarray) and noise_pressure > 0.0:
        frame_threshold = _apply_register_threshold_bias(
            frame_threshold,
            low_delta=0.0,
            high_delta=float(0.008 * profile_scale * _clamp(noise_pressure, 0.0, 1.2)),
            mid_delta=float(0.004 * profile_scale * _clamp(noise_pressure, 0.0, 1.2)),
        )
        frame_threshold = np.clip(frame_threshold, 0.04, 0.34).astype(np.float32, copy=False)
    if pedal_vals is not None:
        pedal_offset_threshold = _clamp(_percentile_or_default(pedal_vals, 90.0, pedal_offset_threshold), 0.15, 0.35)
    return (onset_threshold, offset_threshold, frame_threshold, pedal_offset_threshold)


def _create_post_processor(*, thresholds: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray, float] | None=None, max_note_frames: int | None=600) -> RegressionPostProcessor:
    if thresholds is None:
        thresholds = (
            float(_LEGACY_ONSET_THRESHOLD),
            float(_LEGACY_OFFSET_THRESHOLD),
            float(_LEGACY_FRAME_THRESHOLD),
            float(_LEGACY_PEDAL_OFFSET_THRESHOLD),
        )
    onset_threshold, offset_threshold, frame_threshold, pedal_offset_threshold = thresholds
    return RegressionPostProcessor(
        frames_per_second=pti_config.frames_per_second,
        classes_num=pti_config.classes_num,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        frame_threshold=frame_threshold,
        pedal_offset_threshold=float(pedal_offset_threshold),
        max_note_frames=max_note_frames,
    )


def _apply_register_threshold_bias(values: np.ndarray, *, low_delta: float, high_delta: float, mid_delta: float=0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    if arr.size <= 0:
        return arr
    begin_note = int(getattr(pti_config, 'begin_note', 21))
    midi_notes = begin_note + np.arange(arr.size, dtype=np.int32)
    low_mask = midi_notes <= int(_LOW_REGISTER_MAX_MIDI)
    mid_mask = (midi_notes >= int(_MID_HIGH_REGISTER_MIN_MIDI)) & (midi_notes <= int(_MID_HIGH_REGISTER_MAX_MIDI))
    high_mask = midi_notes >= int(_HIGH_REGISTER_MIN_MIDI)
    if np.any(low_mask):
        arr[low_mask] += float(low_delta)
    if float(mid_delta) > 0.0 and np.any(mid_mask):
        arr[mid_mask] += float(mid_delta)
    if np.any(high_mask):
        arr[high_mask] += float(high_delta)
    return arr.astype(np.float32, copy=False)

def _deframe(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    if x.shape[0] == 1:
        return x[0]
    x = x[:, 0:-1, :]
    _, segment_frames, _ = x.shape
    if segment_frames % 4 != 0:
        return x.reshape(-1, x.shape[-1])
    parts = [x[0, 0:int(segment_frames * 0.75)]]
    for i in range(1, x.shape[0] - 1):
        parts.append(x[i, int(segment_frames * 0.25):int(segment_frames * 0.75)])
    parts.append(x[-1, int(segment_frames * 0.25):])
    return np.concatenate(parts, axis=0)


def _base_overlap_window(length: int, *, min_weight: float=0.05) -> np.ndarray:
    window = np.hanning(max(2, int(length))).astype(np.float32)
    if not np.any(window > 0):
        window = np.ones((int(length),), dtype=np.float32)
    return np.clip(window, float(min_weight), 1.0)


def _build_overlap_weight_map_from_onset(reg_onset_output: np.ndarray | None) -> np.ndarray | None:
    if reg_onset_output is None:
        return None
    onset = np.asarray(reg_onset_output, dtype=np.float32)
    if onset.ndim != 3 or onset.shape[0] <= 1:
        return None
    onset = onset[:, 0:-1, :]
    if onset.ndim != 3 or onset.shape[1] <= 1:
        return None
    confidence = np.max(onset, axis=2)
    if confidence.size == 0:
        return None
    scale = _percentile_or_default(confidence, 95.0, 1.0)
    if scale <= 1e-6:
        scale = 1.0
    confidence_norm = np.clip(confidence / scale, 0.0, 1.0)
    confidence_curve = np.power(confidence_norm, 1.8)
    noise_floor = _percentile_or_default(confidence, 60.0, 0.0)
    support_gate = np.where(confidence >= noise_floor, 1.0, 0.65).astype(np.float32)
    segment_frames = int(confidence.shape[1])
    edge_pos = np.linspace(0.0, 1.0, segment_frames, dtype=np.float32)
    edge_distance = np.abs(edge_pos - 0.5) * 2.0
    edge_gate = np.clip(1.0 - np.power(edge_distance, 0.9), 0.0, 1.0)[None, :]
    low_conf_gate = (0.15 + 0.85 * np.power(confidence_norm, 1.2)).astype(np.float32)
    base_window = _base_overlap_window(confidence.shape[1], min_weight=0.01)[None, :]
    weight_map = base_window * support_gate * low_conf_gate * (0.25 + 1.35 * confidence_curve) * (0.2 + 0.8 * edge_gate)
    return np.asarray(weight_map, dtype=np.float32)


def _deframe_overlap_weighted(x: np.ndarray, *, weight_map: np.ndarray | None=None) -> np.ndarray:
    if x.ndim == 2:
        return x
    if x.shape[0] == 1:
        return x[0]
    x = x[:, 0:-1, :]
    segments = int(x.shape[0])
    segment_frames = int(x.shape[1])
    if segments <= 1 or segment_frames <= 1:
        return x.reshape(-1, x.shape[-1])
    hop = max(1, segment_frames // 2)
    total_frames = hop * (segments - 1) + segment_frames
    features = int(x.shape[-1])
    accum = np.zeros((total_frames, features), dtype=np.float32)
    weights = np.zeros((total_frames, 1), dtype=np.float32)
    base_window_col = _base_overlap_window(segment_frames, min_weight=0.01)[:, None]
    use_map = (
        weight_map is not None
        and isinstance(weight_map, np.ndarray)
        and weight_map.ndim == 2
        and int(weight_map.shape[0]) == segments
        and int(weight_map.shape[1]) == segment_frames
    )
    for idx in range(segments):
        start = idx * hop
        end = start + segment_frames
        chunk = np.asarray(x[idx], dtype=np.float32)
        if use_map:
            seg_weight = np.asarray(weight_map[idx], dtype=np.float32)[:, None]
        else:
            seg_weight = base_window_col
        accum[start:end] += chunk * seg_weight
        weights[start:end] += seg_weight
    return accum / np.maximum(weights, 1e-6)

def _run_session_batches(session, output_map: dict[str, str], segments: Iterable[np.ndarray], *, total_segments: int | None=None, progress_callback: SegmentProgressCallback | None=None, stop_event: Event | None=None, batch_size_override: int | None=None, device_preference: str | None=None) -> dict[str, np.ndarray]:
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError('ONNX model has no input tensor.')
    input_name = str(inputs[0].name)
    output_keys = list(output_map.keys())
    requested_output_names = [output_map[key] for key in output_keys]
    outputs_acc: dict[str, list[np.ndarray]] = {key: [] for key in output_keys}
    if batch_size_override is not None:
        batch_size = max(1, int(batch_size_override))
    else:
        use_gpu_batches = str(device_preference or '').strip().lower() == 'gpu'
        batch_size = runtime_service.get_gpu_batch_size() if use_gpu_batches else 1
    batch_size = max(1, int(batch_size))
    total_batches = 0
    if total_segments is not None:
        total_batches = int(math.ceil(max(0, int(total_segments)) / float(batch_size)))
    pending: list[np.ndarray] = []
    batch_index = 0
    processed_segments = 0

    def flush_pending() -> None:
        nonlocal batch_index, processed_segments
        if not pending:
            return
        pending_count = len(pending)
        _ensure_not_stopped(stop_event)
        batch_index += 1
        if len(pending) == 1:
            batch = pending[0]
        else:
            batch = np.concatenate(pending, axis=0)
        if batch.dtype != np.float32 or (not batch.flags.c_contiguous):
            batch = np.ascontiguousarray(batch, dtype=np.float32)
        pending.clear()
        raw_outputs = session.run(requested_output_names, {input_name: batch})
        _ensure_not_stopped(stop_event)
        for key, value in zip(output_keys, raw_outputs):
            outputs_acc[key].append(np.asarray(value))
        previous_processed = processed_segments
        processed_segments += pending_count
        if progress_callback:
            if total_segments is not None and int(total_segments) > 0:
                for step in range(1, pending_count + 1):
                    progressed = previous_processed + step
                    clamped = min(progressed, int(total_segments))
                    progress_callback(clamped, int(total_segments))
            else:
                progress_callback(batch_index, max(1, total_batches))

    saw_segments = False
    for segment in segments:
        _ensure_not_stopped(stop_event)
        segment_arr = np.asarray(segment, dtype=np.float32)
        if segment_arr.ndim == 1:
            segment_arr = segment_arr[None, :]
        if segment_arr.ndim != 2:
            raise RuntimeError('Audio segment must be a 1D/2D numpy array.')
        if segment_arr.dtype != np.float32 or (not segment_arr.flags.c_contiguous):
            segment_arr = np.ascontiguousarray(segment_arr, dtype=np.float32)
        pending.append(segment_arr)
        saw_segments = True
        if len(pending) >= batch_size:
            flush_pending()

    flush_pending()
    if not saw_segments:
        raise RuntimeError('No audio segments were generated for transcription.')
    return {key: np.concatenate(values, axis=0) for key, values in outputs_acc.items() if values}


def _resolve_modern_post_min_velocity(note_events: list[dict], base_min_velocity: int) -> int:
    try:
        base = max(0, int(base_min_velocity))
    except Exception:
        base = 20
    velocities = []
    for event in note_events:
        try:
            velocities.append(float(event.get('velocity', 0)))
        except Exception:
            continue
    if not velocities:
        return max(base, 12)
    vel_arr = np.asarray(velocities, dtype=np.float32)
    quiet = _percentile_or_default(vel_arr, 15.0, 12.0)
    median = _percentile_or_default(vel_arr, 50.0, 24.0)
    dynamic_floor = int(round(_clamp(quiet + max(0.0, median - quiet) * 0.25, 12.0, 36.0)))
    return max(base, dynamic_floor)


def _build_modern_cleanup_profile(notes: list, *, base_min_duration: float, base_min_velocity: int, cleanup_scale: float=1.0) -> dict[str, float]:
    durations = []
    velocities = []
    pitches = []
    for note in notes:
        start_time = float(getattr(note, 'start', 0.0))
        end_time = float(getattr(note, 'end', start_time))
        durations.append(max(0.0, end_time - start_time))
        velocities.append(max(0, int(getattr(note, 'velocity', 0))))
        pitches.append(int(getattr(note, 'pitch', 0)))
    dur_arr = np.asarray(durations, dtype=np.float32)
    vel_arr = np.asarray(velocities, dtype=np.float32)
    pitch_arr = np.asarray(pitches, dtype=np.int16)
    total_notes = max(1, int(dur_arr.size))
    short_ratio_80 = float(np.sum(dur_arr <= 0.080)) / float(total_notes)
    short_ratio_50 = float(np.sum(dur_arr <= 0.050)) / float(total_notes)
    fragmentation_pressure = float(_clamp(((short_ratio_80 - 0.10) * 4.0) + ((short_ratio_50 - 0.04) * 6.0), 0.0, 1.5))
    mid_high_ratio = float(np.sum(pitch_arr >= int(_MID_HIGH_REGISTER_MIN_MIDI))) / float(total_notes) if pitch_arr.size > 0 else 0.0
    cluster_pressure = float(_clamp(((short_ratio_80 - 0.14) * 3.8) + ((mid_high_ratio - 0.52) * 1.4), 0.0, 1.5))
    micro_noise_pressure = float(_clamp(max(fragmentation_pressure, cluster_pressure), 0.0, 1.7))
    weak_duration = _clamp(
        _percentile_or_default(dur_arr, 22.0, max(0.05, float(base_min_duration) * 1.2)),
        max(0.035, float(base_min_duration) * 1.15),
        0.12,
    )
    high_short_duration = _clamp(
        _percentile_or_default(dur_arr, 34.0, max(0.06, float(base_min_duration) * 2.0)),
        max(0.05, float(base_min_duration) * 1.8),
        0.16,
    )
    weak_velocity = int(round(_clamp(
        _percentile_or_default(vel_arr, 24.0, float(base_min_velocity) + 3.0),
        float(base_min_velocity),
        52.0,
    )))
    high_weak_velocity = int(round(_clamp(
        _percentile_or_default(vel_arr, 33.0, float(base_min_velocity) + 7.0),
        float(base_min_velocity) + 2.0,
        58.0,
    )))
    mid_short_duration = float(_clamp(
        _percentile_or_default(dur_arr, 30.0, max(0.05, float(base_min_duration) * 1.7)),
        max(0.045, float(base_min_duration) * 1.4),
        0.15,
    ))
    mid_weak_velocity = int(round(_clamp(
        _percentile_or_default(vel_arr, 30.0, float(base_min_velocity) + 5.0),
        float(base_min_velocity) + 1.0,
        56.0,
    )))
    profile_scale = _normalize_modern_scale(cleanup_scale)
    if micro_noise_pressure > 0.0:
        weak_duration = float(_clamp(weak_duration * (1.0 + (0.38 * micro_noise_pressure * profile_scale)), max(0.035, float(base_min_duration) * 1.2), 0.16))
        mid_short_duration = float(_clamp(mid_short_duration * (1.0 + (0.33 * micro_noise_pressure * profile_scale)), max(0.045, float(base_min_duration) * 1.4), 0.18))
        high_short_duration = float(_clamp(high_short_duration * (1.0 + (0.36 * micro_noise_pressure * profile_scale)), max(0.05, float(base_min_duration) * 1.8), 0.22))
        weak_velocity = int(round(_clamp(float(weak_velocity) + (3.2 * micro_noise_pressure * profile_scale), float(base_min_velocity), 62.0)))
        mid_weak_velocity = int(round(_clamp(float(mid_weak_velocity) + (4.0 * micro_noise_pressure * profile_scale), float(base_min_velocity) + 1.0, 64.0)))
        high_weak_velocity = int(round(_clamp(float(high_weak_velocity) + (5.5 * micro_noise_pressure * profile_scale), float(base_min_velocity) + 2.0, 68.0)))
    return {
        'weak_duration': float(weak_duration),
        'mid_short_duration': float(mid_short_duration),
        'high_short_duration': float(high_short_duration),
        'weak_velocity': int(weak_velocity),
        'mid_weak_velocity': int(mid_weak_velocity),
        'high_weak_velocity': int(high_weak_velocity),
        'fragmentation_pressure': float(fragmentation_pressure),
        'cluster_pressure': float(cluster_pressure),
        'micro_noise_pressure': float(micro_noise_pressure),
        'mid_high_ratio': float(mid_high_ratio),
        'short_ratio_80': float(short_ratio_80),
        'short_ratio_50': float(short_ratio_50),
        'profile_scale': float(profile_scale),
    }


def _estimate_modern_onset_shift_seconds(audio_file_path: Path, midi_file_path: Path, *, max_shift_seconds: float=_MODERN_ALIGNMENT_MAX_SHIFT_SECONDS) -> tuple[float, float]:
    pretty_midi = get_pretty_midi_module()
    if pretty_midi is None:
        return (0.0, 0.0)
    librosa = get_librosa_module()
    try:
        y, sr = librosa.load(str(audio_file_path), sr=8000, mono=True, dtype=np.float32)
    except Exception:
        return (0.0, 0.0)
    if y.size <= 2048:
        return (0.0, 0.0)
    hop = 256
    try:
        onset_env = np.asarray(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), dtype=np.float32).reshape(-1)
    except Exception:
        return (0.0, 0.0)
    if onset_env.size <= 8:
        return (0.0, 0.0)
    onset_env = (onset_env - float(np.mean(onset_env))) / (float(np.std(onset_env)) + 1e-8)
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_file_path))
    except Exception:
        return (0.0, 0.0)
    starts: list[float] = []
    for inst in midi_data.instruments:
        if bool(getattr(inst, 'is_drum', False)):
            continue
        for note in inst.notes:
            starts.append(float(getattr(note, 'start', 0.0)))
    if len(starts) < 10:
        return (0.0, 0.0)
    starts_arr = np.asarray(starts, dtype=np.float64)
    impulse = np.zeros((onset_env.size,), dtype=np.float32)
    frames = np.round(starts_arr * float(sr) / float(hop)).astype(np.int64)
    frames = frames[(frames >= 0) & (frames < impulse.size)]
    if frames.size <= 0:
        return (0.0, 0.0)
    np.add.at(impulse, frames, 1.0)
    impulse = (impulse - float(np.mean(impulse))) / (float(np.std(impulse)) + 1e-8)
    max_lag = max(1, int(round(float(max_shift_seconds) * float(sr) / float(hop))))
    best_lag = 0
    best_score = -float('inf')
    second_score = -float('inf')
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = onset_env[lag:]
            b = impulse[:onset_env.size - lag]
        else:
            a = onset_env[:onset_env.size + lag]
            b = impulse[-lag:]
        if a.size <= 0 or b.size <= 0:
            continue
        score = float(np.dot(a, b))
        if score > best_score:
            second_score = best_score
            best_score = score
            best_lag = lag
        elif score > second_score:
            second_score = score
    if not np.isfinite(best_score):
        return (0.0, 0.0)
    confidence = float(_clamp((best_score - second_score) / (abs(best_score) + 1e-8), 0.0, 1.0))
    return (float(best_lag) * float(hop) / float(sr), confidence)


def _nearest_distances_to_sorted_reference(query: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    ref = np.asarray(sorted_reference, dtype=np.float64).reshape(-1)
    if q.size <= 0 or ref.size <= 0:
        return np.full(q.shape, np.inf, dtype=np.float64)
    idx = np.searchsorted(ref, q)
    left_idx = np.clip(idx - 1, 0, ref.size - 1)
    right_idx = np.clip(idx, 0, ref.size - 1)
    left = np.abs(q - ref[left_idx])
    right = np.abs(q - ref[right_idx])
    return np.minimum(left, right)


def _load_alignment_onset_series(audio_file_path: Path, midi_file_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    pretty_midi = get_pretty_midi_module()
    if pretty_midi is None:
        return (None, None)
    librosa = get_librosa_module()
    try:
        y, sr = librosa.load(str(audio_file_path), sr=8000, mono=True, dtype=np.float32)
    except Exception:
        return (None, None)
    if y.size <= 2048:
        return (None, None)
    hop = 256
    try:
        onset_env = np.asarray(librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), dtype=np.float32).reshape(-1)
        onset_frames = np.asarray(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False), dtype=np.int64).reshape(-1)
    except Exception:
        return (None, None)
    if onset_frames.size <= 0:
        return (None, None)
    audio_onsets = onset_frames.astype(np.float64) * float(hop) / float(sr)
    audio_onsets.sort()
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_file_path))
    except Exception:
        return (None, None)
    starts: list[float] = []
    for inst in midi_data.instruments:
        if bool(getattr(inst, 'is_drum', False)):
            continue
        for note in inst.notes:
            starts.append(float(getattr(note, 'start', 0.0)))
    if len(starts) < 10:
        return (None, None)
    starts_arr = np.asarray(starts, dtype=np.float64).reshape(-1)
    starts_arr = starts_arr[np.isfinite(starts_arr)]
    if starts_arr.size < 10:
        return (None, None)
    starts_arr.sort()
    return (audio_onsets, starts_arr)


def _estimate_alignment_drift_seconds_per_min(starts: np.ndarray, nearest_errors: np.ndarray, *, window_seconds: float=20.0, step_seconds: float=10.0, min_notes: int=12) -> tuple[float, float]:
    starts_arr = np.asarray(starts, dtype=np.float64).reshape(-1)
    err_arr = np.asarray(nearest_errors, dtype=np.float64).reshape(-1)
    if starts_arr.size <= 0 or err_arr.size <= 0:
        return (0.0, float('inf'))
    if starts_arr.size != err_arr.size:
        size = min(starts_arr.size, err_arr.size)
        starts_arr = starts_arr[:size]
        err_arr = err_arr[:size]
    max_time = float(np.max(starts_arr)) if starts_arr.size > 0 else 0.0
    if max_time <= 0.0:
        return (0.0, float(np.median(np.abs(err_arr))) if err_arr.size > 0 else float('inf'))
    half = max(2.0, float(window_seconds) * 0.5)
    centers = np.arange(0.0, max_time + max(1.0, float(step_seconds)), max(1.0, float(step_seconds)), dtype=np.float64)
    xs: list[float] = []
    ys: list[float] = []
    for center in centers:
        mask = (starts_arr >= (center - half)) & (starts_arr <= (center + half))
        if int(np.sum(mask)) < int(min_notes):
            continue
        window_vals = np.abs(err_arr[mask])
        finite_vals = window_vals[np.isfinite(window_vals)]
        if finite_vals.size <= 0:
            continue
        xs.append(float(center))
        ys.append(float(np.median(finite_vals)))
    if len(xs) < 2:
        fallback = np.abs(err_arr[np.isfinite(err_arr)])
        return (0.0, float(np.median(fallback)) if fallback.size > 0 else float('inf'))
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    slope_sec_per_sec = float(np.polyfit(x_arr, y_arr, 1)[0])
    return (float(slope_sec_per_sec) * 60.0, float(np.median(y_arr)))


def _evaluate_alignment_error_metrics(audio_onsets: np.ndarray, midi_starts: np.ndarray, *, shift_seconds: float=0.0) -> dict[str, float]:
    shifted = np.asarray(midi_starts, dtype=np.float64).reshape(-1) + float(shift_seconds)
    shifted = np.maximum(0.0, shifted)
    nearest = _nearest_distances_to_sorted_reference(shifted, audio_onsets)
    finite = nearest[np.isfinite(nearest)]
    if finite.size <= 0:
        return {
            'median': float('inf'),
            'p90': float('inf'),
            'drift_slope_sec_per_min': float('inf'),
            'window_median_abs_sec': float('inf'),
        }
    drift_slope, window_median_abs = _estimate_alignment_drift_seconds_per_min(shifted, nearest)
    return {
        'median': float(np.median(finite)),
        'p90': float(np.percentile(finite, 90.0)),
        'drift_slope_sec_per_min': float(drift_slope),
        'window_median_abs_sec': float(window_median_abs),
    }


def _estimate_midi_to_audio_onset_error_seconds(audio_file_path: Path, midi_file_path: Path, *, shift_seconds: float=0.0) -> tuple[float, float]:
    audio_onsets, midi_starts = _load_alignment_onset_series(audio_file_path, midi_file_path)
    if audio_onsets is None or midi_starts is None:
        return (float('inf'), float('inf'))
    metrics = _evaluate_alignment_error_metrics(audio_onsets, midi_starts, shift_seconds=float(shift_seconds))
    return (float(metrics.get('median', float('inf'))), float(metrics.get('p90', float('inf'))))


def _should_apply_modern_alignment_shift(audio_file_path: Path, midi_file_path: Path, proposed_shift_seconds: float, *, confidence: float, alignment_gate_scale: float=1.0) -> bool:
    gate_scale = _normalize_modern_scale(alignment_gate_scale)
    strictness = gate_scale
    min_confidence = float(_MODERN_ALIGNMENT_MIN_CONFIDENCE + (0.03 * strictness))
    min_median_improvement = float(_MODERN_ALIGNMENT_MIN_MEDIAN_IMPROVEMENT_SECONDS + (0.004 * strictness))
    max_p90_degradation = float(max(0.003, _MODERN_ALIGNMENT_MAX_P90_DEGRADATION_SECONDS - (0.004 * strictness)))
    max_drift_worsen = float(max(0.003, _MODERN_ALIGNMENT_MAX_DRIFT_WORSEN_SECONDS_PER_MIN - (0.003 * strictness)))
    max_window_worsen = float(max(0.0015, _MODERN_ALIGNMENT_MAX_WINDOW_MEDIAN_WORSEN_SECONDS - (0.001 * strictness)))
    max_abs_drift = float(max(0.025, _MODERN_ALIGNMENT_MAX_ABS_DRIFT_SECONDS_PER_MIN - (0.015 * strictness)))
    if abs(float(proposed_shift_seconds)) < float(_MODERN_ALIGNMENT_MIN_APPLY_SECONDS):
        return False
    if float(confidence) < min_confidence:
        return False
    audio_onsets, midi_starts = _load_alignment_onset_series(audio_file_path, midi_file_path)
    if audio_onsets is None or midi_starts is None:
        return False
    before_metrics = _evaluate_alignment_error_metrics(audio_onsets, midi_starts, shift_seconds=0.0)
    after_metrics = _evaluate_alignment_error_metrics(audio_onsets, midi_starts, shift_seconds=float(proposed_shift_seconds))
    before_med = float(before_metrics.get('median', float('inf')))
    before_p90 = float(before_metrics.get('p90', float('inf')))
    after_med = float(after_metrics.get('median', float('inf')))
    after_p90 = float(after_metrics.get('p90', float('inf')))
    before_drift = float(before_metrics.get('drift_slope_sec_per_min', float('inf')))
    after_drift = float(after_metrics.get('drift_slope_sec_per_min', float('inf')))
    before_window = float(before_metrics.get('window_median_abs_sec', float('inf')))
    after_window = float(after_metrics.get('window_median_abs_sec', float('inf')))
    if not (
        np.isfinite(before_med)
        and np.isfinite(after_med)
        and np.isfinite(before_p90)
        and np.isfinite(after_p90)
        and np.isfinite(before_drift)
        and np.isfinite(after_drift)
        and np.isfinite(before_window)
        and np.isfinite(after_window)
    ):
        return False
    median_gain = float(before_med - after_med)
    p90_delta = float(after_p90 - before_p90)
    if median_gain < min_median_improvement:
        return False
    if p90_delta > max_p90_degradation:
        return False
    drift_worsen = abs(after_drift) - abs(before_drift)
    window_worsen = after_window - before_window
    if (
        drift_worsen > max_drift_worsen
        and window_worsen > max_window_worsen
    ):
        return False
    if (
        abs(after_drift) > max_abs_drift
        and median_gain < float(min_median_improvement + 0.006)
    ):
        return False
    return True


def _apply_modern_global_midi_shift(midi_file_path: Path, shift_seconds: float, *, stop_event: Event | None=None) -> float:
    if abs(float(shift_seconds)) < float(_MODERN_ALIGNMENT_MIN_APPLY_SECONDS):
        return 0.0
    pretty_midi = get_pretty_midi_module()
    if pretty_midi is None:
        return 0.0
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_file_path))
    except Exception:
        return 0.0
    applied = float(shift_seconds)
    for instrument in midi_data.instruments:
        _ensure_not_stopped(stop_event)
        for note in instrument.notes:
            start = max(0.0, float(getattr(note, 'start', 0.0)) + applied)
            end = max(start + 1e-4, float(getattr(note, 'end', start)) + applied)
            note.start = start
            note.end = end
        for cc in getattr(instrument, 'control_changes', []):
            cc.time = max(0.0, float(getattr(cc, 'time', 0.0)) + applied)
        for pb in getattr(instrument, 'pitch_bends', []):
            pb.time = max(0.0, float(getattr(pb, 'time', 0.0)) + applied)
    _ensure_not_stopped(stop_event)
    midi_data.write(str(midi_file_path))
    return applied


def _extract_pedal_intervals(instrument) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    controls = [cc for cc in getattr(instrument, 'control_changes', []) if int(getattr(cc, 'number', -1)) == 64]
    if not controls:
        return intervals
    controls.sort(key=lambda cc: float(getattr(cc, 'time', 0.0)))
    pedal_down_time: float | None = None
    for cc in controls:
        time_value = float(getattr(cc, 'time', 0.0))
        value = int(getattr(cc, 'value', 0))
        if value >= 64:
            if pedal_down_time is None:
                pedal_down_time = time_value
        elif pedal_down_time is not None:
            intervals.append((pedal_down_time, max(time_value, pedal_down_time)))
            pedal_down_time = None
    if pedal_down_time is not None:
        last_note_end = max((float(getattr(note, 'end', pedal_down_time)) for note in getattr(instrument, 'notes', [])), default=pedal_down_time)
        intervals.append((pedal_down_time, max(last_note_end, pedal_down_time)))
    return intervals


def _is_pedal_active_at_time(pedal_intervals: list[tuple[float, float]], time_value: float) -> bool:
    for start_time, end_time in pedal_intervals:
        if start_time <= time_value <= end_time:
            return True
    return False


def _register_bucket_for_pitch(pitch: int) -> str:
    p = int(pitch)
    if p <= int(_LOW_REGISTER_MAX_MIDI):
        return 'low'
    if p >= int(_HIGH_REGISTER_MIN_MIDI):
        return 'high'
    return 'mid'


def _is_near_overlap_boundary(time_value: float, *, tolerance: float=0.06) -> bool:
    hop_seconds = max(1e-3, float(_SEGMENT_SECONDS) * 0.5)
    phase = float(time_value) % hop_seconds
    distance = min(phase, hop_seconds - phase)
    return distance <= max(0.0, float(tolerance))


def _prune_pedal_burst_notes(notes: list, *, pedal_intervals: list[tuple[float, float]], low_pitch_max: int=_LOW_REGISTER_MAX_MIDI, bucket_seconds: float=0.04, max_notes_per_bucket: int=4) -> tuple[list, int]:
    if (not notes) or (not pedal_intervals):
        return (notes, 0)
    kept = list(notes)
    removed = 0
    bucket_map: dict[int, list] = {}
    for note in kept:
        pitch = int(getattr(note, 'pitch', -1))
        if pitch > int(low_pitch_max):
            continue
        start_time = float(getattr(note, 'start', 0.0))
        if not _is_pedal_active_at_time(pedal_intervals, start_time):
            continue
        bucket = int(round(start_time / max(1e-3, float(bucket_seconds))))
        bucket_map.setdefault(bucket, []).append(note)
    drop_ids: set[int] = set()
    for bucket_notes in bucket_map.values():
        if len(bucket_notes) <= int(max_notes_per_bucket):
            continue
        ranked = sorted(
            bucket_notes,
            key=lambda n: (
                int(getattr(n, 'velocity', 0)),
                float(getattr(n, 'end', 0.0)) - float(getattr(n, 'start', 0.0)),
                -int(getattr(n, 'pitch', 0)),
            ),
            reverse=True,
        )
        for note in ranked[int(max_notes_per_bucket):]:
            drop_ids.add(id(note))
    if not drop_ids:
        return (kept, 0)
    filtered = []
    for note in kept:
        if id(note) in drop_ids:
            removed += 1
            continue
        filtered.append(note)
    return (filtered, removed)


def _prune_pedal_near_pitch_cooldown(notes: list, *, pedal_intervals: list[tuple[float, float]], window_seconds: float=0.07) -> tuple[list, int, int]:
    if (not notes) or (not pedal_intervals):
        return (notes, 0, 0)
    sorted_notes = sorted(notes, key=lambda n: (float(getattr(n, 'start', 0.0)), int(getattr(n, 'pitch', 0))))
    kept: list = []
    removed = 0
    boundary_removed = 0
    recent: list[dict[str, float | int | object]] = []
    for note in sorted_notes:
        start_time = float(getattr(note, 'start', 0.0))
        pitch = int(getattr(note, 'pitch', 0))
        duration = max(0.0, float(getattr(note, 'end', start_time)) - start_time)
        velocity = int(getattr(note, 'velocity', 0))
        pedal_active = _is_pedal_active_at_time(pedal_intervals, start_time)
        if (not pedal_active) or (pitch < int(_MID_HIGH_REGISTER_MIN_MIDI)):
            kept.append(note)
            continue
        cutoff = start_time - max(0.01, float(window_seconds))
        recent = [item for item in recent if float(item['start']) >= cutoff]
        near_match = None
        for item in recent:
            if abs(int(item['pitch']) - pitch) <= 1:
                near_match = item
                break
        if near_match is None:
            kept.append(note)
            recent.append({'note': note, 'start': start_time, 'pitch': pitch, 'velocity': velocity, 'duration': duration})
            continue
        prev_velocity = int(near_match['velocity'])
        prev_duration = float(near_match['duration'])
        current_strength = velocity + duration * 120.0
        prev_strength = prev_velocity + prev_duration * 120.0
        if current_strength > (prev_strength + 1.5):
            prev_note = near_match['note']
            kept = [k for k in kept if k is not prev_note]
            kept.append(note)
            near_match['note'] = note
            near_match['start'] = start_time
            near_match['pitch'] = pitch
            near_match['velocity'] = velocity
            near_match['duration'] = duration
            removed += 1
            if _is_near_overlap_boundary(start_time):
                boundary_removed += 1
            continue
        removed += 1
        if _is_near_overlap_boundary(start_time):
            boundary_removed += 1
    kept.sort(key=lambda n: (int(getattr(n, 'pitch', 0)), float(getattr(n, 'start', 0.0))))
    return (kept, removed, boundary_removed)


def _prune_pedal_suspicious_clusters(
    notes: list,
    *,
    pedal_intervals: list[tuple[float, float]],
    window_seconds: float=0.18,
    min_cluster_notes: int=6,
    keep_top_strength: int=3,
    pitch_min: int=_MID_HIGH_REGISTER_MIN_MIDI,
    weak_duration: float=0.12,
    weak_velocity: int=52,
) -> tuple[list, int, int]:
    if (not notes) or (not pedal_intervals):
        return (notes, 0, 0)
    sorted_notes = sorted(notes, key=lambda n: float(getattr(n, 'start', 0.0)))
    removed = 0
    boundary_removed = 0
    dropped_ids: set[int] = set()
    candidates: list[dict[str, float | int | object]] = []
    window = max(0.06, float(window_seconds))
    weak_duration_cap = max(0.04, float(weak_duration))
    weak_velocity_cap = max(1, int(weak_velocity))
    for note in sorted_notes:
        note_id = id(note)
        if note_id in dropped_ids:
            continue
        start_time = float(getattr(note, 'start', 0.0))
        pitch = int(getattr(note, 'pitch', 0))
        if pitch < int(pitch_min):
            continue
        if not _is_pedal_active_at_time(pedal_intervals, start_time):
            continue
        duration = max(0.0, float(getattr(note, 'end', start_time)) - start_time)
        velocity = int(getattr(note, 'velocity', 0))
        strength = float(velocity) + (duration * 120.0) + (0.15 * float(pitch - int(pitch_min)))
        cutoff = start_time - window
        candidates = [item for item in candidates if float(item['start']) >= cutoff and int(item['id']) not in dropped_ids]
        candidates.append({'id': note_id, 'note': note, 'start': start_time, 'duration': duration, 'velocity': velocity, 'strength': strength})
        if len(candidates) < int(min_cluster_notes):
            continue
        ranked = sorted(candidates, key=lambda item: float(item['strength']), reverse=True)
        keep_ids = {int(item['id']) for item in ranked[:max(1, int(keep_top_strength))]}
        for item in ranked[max(1, int(keep_top_strength)):]:
            iid = int(item['id'])
            if iid in dropped_ids or iid in keep_ids:
                continue
            is_weak = (
                float(item['duration']) <= weak_duration_cap
                and int(item['velocity']) <= weak_velocity_cap
            )
            if not is_weak:
                continue
            dropped_ids.add(iid)
            removed += 1
            if _is_near_overlap_boundary(float(item['start'])):
                boundary_removed += 1
        candidates = [item for item in candidates if int(item['id']) not in dropped_ids]
    if not dropped_ids:
        return (notes, 0, 0)
    filtered = [note for note in sorted_notes if id(note) not in dropped_ids]
    filtered.sort(key=lambda n: (int(getattr(n, 'pitch', 0)), float(getattr(n, 'start', 0.0))))
    return (filtered, removed, boundary_removed)


def _describe_threshold(value: float | np.ndarray) -> str:
    if isinstance(value, (int, float, np.floating)):
        return f'{float(value):.3f}'
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return 'n/a'
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return 'n/a'
    return f'min={float(np.min(finite)):.3f}, med={float(np.median(finite)):.3f}, max={float(np.max(finite)):.3f}'


def _format_modern_diagnostics(*, conversion_options: ConversionOptions, thresholds: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray, float], total_segments: int | None) -> str:
    onset_threshold, offset_threshold, frame_threshold, pedal_offset_threshold = thresholds
    lines = [
        'Modern diagnostics:',
        f'  provider: {get_active_provider_label()}',
        f'  threshold_bias_scale: {_normalize_modern_scale(conversion_options.modern_threshold_bias_scale):.3f}',
        f'  cleanup_scale: {_normalize_modern_scale(conversion_options.modern_cleanup_scale):.3f}',
        f'  pedal_cluster_scale: {_normalize_modern_scale(conversion_options.modern_pedal_cluster_scale):.3f}',
        f'  alignment_gate_scale: {_normalize_modern_scale(conversion_options.modern_alignment_gate_scale):.3f}',
        f'  auto_calibration: {bool(conversion_options.modern_auto_calibration_enabled)}',
        f'  adaptive_thresholds: {bool(conversion_options.modern_adaptive_thresholds_enabled)}',
        f'  input_norm_denoise: {bool(conversion_options.modern_input_normalization_enabled)}',
        f'  smart_overlap_stitching: {bool(conversion_options.modern_smart_overlap_stitching_enabled)}',
        f'  manual_onset_threshold: {float(conversion_options.modern_manual_onset_threshold):.3f}',
        f'  manual_offset_threshold: {float(conversion_options.modern_manual_offset_threshold):.3f}',
        f'  manual_frame_threshold: {float(conversion_options.modern_manual_frame_threshold):.3f}',
        f'  manual_pedal_offset_threshold: {float(conversion_options.modern_manual_pedal_offset_threshold):.3f}',
        f'  onset_threshold: {_describe_threshold(onset_threshold)}',
        f'  offset_threshold: {_describe_threshold(offset_threshold)}',
        f'  frame_threshold: {_describe_threshold(frame_threshold)}',
        f'  pedal_offset_threshold: {float(pedal_offset_threshold):.3f}',
    ]
    if total_segments is not None:
        lines.append(f'  processed_segments: {int(total_segments)}')
    return '\n'.join(lines)


def post_process_midi(midi_file_path: Path | str, min_duration: float=0.05, min_velocity: int=20, stop_event: Event | None=None, *, dedupe_same_pitch_overlap: bool=True, overlap_tolerance_seconds: float=0.0, pedal_aware_cleanup: bool=False, modern_strict_cleanup: bool=False, modern_cleanup_scale: float=1.0, modern_pedal_cluster_scale: float=1.0) -> dict[str, int]:
    global _PRETTY_MIDI_WARNING_EMITTED
    _ensure_not_stopped(stop_event)
    pretty_midi = get_pretty_midi_module()
    if pretty_midi is None:
        if not _PRETTY_MIDI_WARNING_EMITTED:
            _PRETTY_MIDI_WARNING_EMITTED = True
            detail = f': {_PRETTY_MIDI_IMPORT_ERROR}' if _PRETTY_MIDI_IMPORT_ERROR else ''
            print(f'[A2M] Warning: MIDI post-processing skipped{detail}', file=sys.stderr)
        return {
            'removed_duration_velocity': 0,
            'removed_overlap': 0,
            'removed_near_repeat': 0,
            'removed_pedal_burst': 0,
            'removed_pedal_cluster': 0,
            'removed_cooldown': 0,
            'removed_boundary_zone': 0,
            'removed_pedal_tail_proxy': 0,
            'removed_pedal_mid_high': 0,
            'removed_weak_short': 0,
            'removed_mid_low_conf': 0,
            'removed_high_low_conf': 0,
            'removed_fragmented_noise': 0,
            'kept_low': 0,
            'kept_mid': 0,
            'kept_high': 0,
        }
    midi_data = pretty_midi.PrettyMIDI(str(midi_file_path))
    overlap_tolerance = max(0.0, float(overlap_tolerance_seconds))
    stats = {
        'removed_duration_velocity': 0,
        'removed_overlap': 0,
        'removed_near_repeat': 0,
        'removed_pedal_burst': 0,
        'removed_pedal_cluster': 0,
        'removed_cooldown': 0,
        'removed_boundary_zone': 0,
        'removed_pedal_tail_proxy': 0,
        'removed_pedal_mid_high': 0,
        'removed_weak_short': 0,
        'removed_mid_low_conf': 0,
        'removed_high_low_conf': 0,
        'removed_fragmented_noise': 0,
        'kept_low': 0,
        'kept_mid': 0,
        'kept_high': 0,
    }
    for instrument in midi_data.instruments:
        _ensure_not_stopped(stop_event)
        pedal_intervals = _extract_pedal_intervals(instrument) if pedal_aware_cleanup else []
        cleanup_profile = _build_modern_cleanup_profile(
            instrument.notes,
            base_min_duration=float(min_duration),
            base_min_velocity=int(min_velocity),
            cleanup_scale=modern_cleanup_scale,
        ) if modern_strict_cleanup else None
        notes = []
        for note in instrument.notes:
            note_start = float(getattr(note, 'start', 0.0))
            note_end = float(getattr(note, 'end', note_start))
            note_duration = max(0.0, note_end - note_start)
            note_pitch = int(getattr(note, 'pitch', 0))
            note_velocity = int(getattr(note, 'velocity', 0))
            register_bucket = _register_bucket_for_pitch(note_pitch)
            pedal_active = pedal_aware_cleanup and _is_pedal_active_at_time(pedal_intervals, note_start)
            min_duration_for_note = float(min_duration)
            min_velocity_for_note = int(min_velocity)
            weak_short_note = False
            mid_low_conf_note = False
            high_low_conf_note = False
            if pedal_active:
                if modern_strict_cleanup:
                    if register_bucket == 'high':
                        dur_mult = 1.9
                        vel_bonus = 12
                    elif note_pitch >= int(_MID_HIGH_REGISTER_MIN_MIDI):
                        dur_mult = 1.7
                        vel_bonus = 10
                    else:
                        dur_mult = 1.45
                        vel_bonus = 7
                    strong_signal = (
                        note_duration >= max(0.16, float(min_duration) * 3.0)
                        or note_velocity >= (int(min_velocity) + 24)
                    )
                    if strong_signal:
                        dur_mult = max(1.05, dur_mult * 0.7)
                        vel_bonus = int(round(vel_bonus * 0.4))
                    min_duration_for_note *= dur_mult
                    min_velocity_for_note = min(127, min_velocity_for_note + vel_bonus)
                else:
                    min_duration_for_note *= 0.6
            if modern_strict_cleanup and cleanup_profile is not None:
                micro_noise_pressure = float(cleanup_profile.get('micro_noise_pressure', cleanup_profile.get('fragmentation_pressure', 0.0)))
                weak_short_note = (
                    note_duration <= float(cleanup_profile['weak_duration'])
                    and note_velocity <= int(cleanup_profile['weak_velocity'])
                )
                mid_low_conf_note = (
                    register_bucket == 'mid'
                    and note_duration <= float(cleanup_profile.get('mid_short_duration', cleanup_profile['weak_duration']))
                    and note_velocity <= int(cleanup_profile.get('mid_weak_velocity', cleanup_profile['weak_velocity']))
                )
                high_low_conf_note = (
                    register_bucket == 'high'
                    and note_duration <= float(cleanup_profile['high_short_duration'])
                    and note_velocity <= int(cleanup_profile['high_weak_velocity'])
                )
                if weak_short_note:
                    min_duration_for_note = max(min_duration_for_note, float(cleanup_profile['weak_duration']) * 1.15)
                    min_velocity_for_note = min(127, max(min_velocity_for_note, int(cleanup_profile['weak_velocity']) + 2))
                if mid_low_conf_note:
                    min_duration_for_note = max(min_duration_for_note, float(cleanup_profile.get('mid_short_duration', cleanup_profile['weak_duration'])) * 1.25)
                    min_velocity_for_note = min(127, max(min_velocity_for_note, int(cleanup_profile.get('mid_weak_velocity', cleanup_profile['weak_velocity'])) + 3))
                if high_low_conf_note:
                    min_duration_for_note = max(min_duration_for_note, float(cleanup_profile['high_short_duration']) * 1.35)
                    min_velocity_for_note = min(127, max(min_velocity_for_note, int(cleanup_profile['high_weak_velocity']) + 5))
                if micro_noise_pressure > 0.0 and (weak_short_note or mid_low_conf_note or high_low_conf_note):
                    min_duration_for_note = max(min_duration_for_note, float(min_duration_for_note) * (1.0 + (0.12 * micro_noise_pressure)))
                    min_velocity_for_note = min(127, int(round(float(min_velocity_for_note) + (1.8 * micro_noise_pressure))))
            if modern_strict_cleanup and note_pitch <= int(_LOW_REGISTER_MAX_MIDI):
                min_duration_for_note = max(min_duration_for_note, float(min_duration) * 1.8)
                min_velocity_for_note = min(127, min_velocity_for_note + 5)
            if note_duration >= min_duration_for_note and note_velocity >= min_velocity_for_note:
                notes.append(note)
            else:
                stats['removed_duration_velocity'] += 1
                if _is_near_overlap_boundary(note_start):
                    stats['removed_boundary_zone'] += 1
                if pedal_active and note_duration <= 0.12:
                    stats['removed_pedal_tail_proxy'] += 1
                if pedal_active and note_pitch >= int(_MID_HIGH_REGISTER_MIN_MIDI):
                    stats['removed_pedal_mid_high'] += 1
                if weak_short_note:
                    stats['removed_weak_short'] += 1
                if mid_low_conf_note:
                    stats['removed_mid_low_conf'] += 1
                if high_low_conf_note:
                    stats['removed_high_low_conf'] += 1
                if modern_strict_cleanup and cleanup_profile is not None and float(cleanup_profile.get('micro_noise_pressure', cleanup_profile.get('fragmentation_pressure', 0.0))) > 0.0 and (weak_short_note or mid_low_conf_note or high_low_conf_note):
                    stats['removed_fragmented_noise'] += 1
        notes.sort(key=lambda n: (n.pitch, n.start))
        if dedupe_same_pitch_overlap:
            filtered_notes = []
            last_end_by_pitch = {}
            last_start_by_pitch = {}
            last_velocity_by_pitch = {}
            for note in notes:
                _ensure_not_stopped(stop_event)
                last_end = last_end_by_pitch.get(note.pitch, -float('inf'))
                pedal_active = pedal_aware_cleanup and _is_pedal_active_at_time(pedal_intervals, float(note.start))
                if (note.start + overlap_tolerance < last_end) and (modern_strict_cleanup or (not pedal_active)):
                    stats['removed_overlap'] += 1
                    if _is_near_overlap_boundary(float(note.start)):
                        stats['removed_boundary_zone'] += 1
                    if pedal_active and int(getattr(note, 'pitch', 0)) >= int(_MID_HIGH_REGISTER_MIN_MIDI):
                        stats['removed_pedal_mid_high'] += 1
                    continue
                if modern_strict_cleanup:
                    last_start = float(last_start_by_pitch.get(note.pitch, -float('inf')))
                    last_velocity = int(last_velocity_by_pitch.get(note.pitch, 127))
                    near_repeat_window = 0.06
                    if (float(note.start) - last_start) < near_repeat_window and int(getattr(note, 'velocity', 0)) <= (last_velocity + 4):
                        stats['removed_near_repeat'] += 1
                        if _is_near_overlap_boundary(float(note.start)):
                            stats['removed_boundary_zone'] += 1
                        if pedal_active and int(getattr(note, 'pitch', 0)) >= int(_MID_HIGH_REGISTER_MIN_MIDI):
                            stats['removed_pedal_mid_high'] += 1
                        continue
                filtered_notes.append(note)
                last_end_by_pitch[note.pitch] = note.end
                last_start_by_pitch[note.pitch] = note.start
                last_velocity_by_pitch[note.pitch] = int(getattr(note, 'velocity', 0))
        else:
            filtered_notes = notes
        if modern_strict_cleanup and pedal_aware_cleanup:
            filtered_notes, pruned = _prune_pedal_burst_notes(filtered_notes, pedal_intervals=pedal_intervals)
            stats['removed_pedal_burst'] += int(pruned)
            cleanup_profile = cleanup_profile or {}
            cluster_scale = _normalize_modern_scale(modern_pedal_cluster_scale)
            baseline_min_cluster = max(4, min(8, int(round(6.0 - ((cluster_scale - 1.0) * 8.0)))))
            keep_top_strength = 2 if cluster_scale >= 1.10 else 3
            base_mid_weak_velocity = int(min_velocity) + (8 if cluster_scale >= 1.10 else 6)
            filtered_notes, cluster_removed, cluster_boundary_removed = _prune_pedal_suspicious_clusters(
                filtered_notes,
                pedal_intervals=pedal_intervals,
                window_seconds=0.18 / cluster_scale,
                min_cluster_notes=baseline_min_cluster if float(cleanup_profile.get('micro_noise_pressure', 0.0)) < 0.6 else max(4, int(round(5 + (0.5 / max(0.6, cluster_scale))))),
                keep_top_strength=keep_top_strength,
                pitch_min=int(_MID_HIGH_REGISTER_MIN_MIDI),
                weak_duration=float(cleanup_profile.get('mid_short_duration', 0.11)),
                weak_velocity=int(cleanup_profile.get('mid_weak_velocity', base_mid_weak_velocity) + max(0, int(round((cluster_scale - 1.0) * 4.0)))),
            )
            stats['removed_pedal_cluster'] += int(cluster_removed)
            stats['removed_boundary_zone'] += int(cluster_boundary_removed)
            filtered_notes, cooldown_removed, cooldown_boundary_removed = _prune_pedal_near_pitch_cooldown(
                filtered_notes,
                pedal_intervals=pedal_intervals,
                window_seconds=0.07,
            )
            stats['removed_cooldown'] += int(cooldown_removed)
            stats['removed_boundary_zone'] += int(cooldown_boundary_removed)
        for note in filtered_notes:
            bucket = _register_bucket_for_pitch(int(getattr(note, 'pitch', 0)))
            if bucket == 'low':
                stats['kept_low'] += 1
            elif bucket == 'high':
                stats['kept_high'] += 1
            else:
                stats['kept_mid'] += 1
        instrument.notes = filtered_notes
    _ensure_not_stopped(stop_event)
    midi_data.write(str(midi_file_path))
    return stats


def _append_modern_post_cleanup_diagnostics(*, post_stats: dict[str, int], min_duration: float, min_velocity: int, alignment_shift_seconds: float=0.0, alignment_confidence: float=0.0, alignment_applied: bool=False) -> None:
    if not post_stats:
        return
    kept_low = int(post_stats.get('kept_low', 0))
    kept_mid = int(post_stats.get('kept_mid', 0))
    kept_high = int(post_stats.get('kept_high', 0))
    kept_total = max(1, kept_low + kept_mid + kept_high)
    low_rate = 100.0 * kept_low / kept_total
    mid_rate = 100.0 * kept_mid / kept_total
    high_rate = 100.0 * kept_high / kept_total
    lines = [line for line in str(_LAST_MODERN_DIAGNOSTICS_TEXT or '').splitlines() if line.strip()]
    lines.extend(
        [
            '  post_cleanup: strict',
            f'  post_min_duration: {float(min_duration):.3f}s',
            f'  post_min_velocity: {int(min_velocity)}',
            f"  removed_duration_velocity: {int(post_stats.get('removed_duration_velocity', 0))}",
            f"  removed_overlap: {int(post_stats.get('removed_overlap', 0))}",
            f"  removed_near_repeat: {int(post_stats.get('removed_near_repeat', 0))}",
            f"  removed_pedal_burst: {int(post_stats.get('removed_pedal_burst', 0))}",
            f"  removed_pedal_cluster: {int(post_stats.get('removed_pedal_cluster', 0))}",
            f"  removed_cooldown: {int(post_stats.get('removed_cooldown', 0))}",
            f"  removed_boundary_zone: {int(post_stats.get('removed_boundary_zone', 0))}",
            f"  removed_pedal_tail_proxy: {int(post_stats.get('removed_pedal_tail_proxy', 0))}",
            f"  removed_pedal_mid_high: {int(post_stats.get('removed_pedal_mid_high', 0))}",
            f"  removed_weak_short: {int(post_stats.get('removed_weak_short', 0))}",
            f"  removed_mid_low_conf: {int(post_stats.get('removed_mid_low_conf', 0))}",
            f"  removed_high_low_conf: {int(post_stats.get('removed_high_low_conf', 0))}",
            f"  removed_fragmented_noise: {int(post_stats.get('removed_fragmented_noise', 0))}",
            f'  alignment_shift_s: {float(alignment_shift_seconds):+.4f}',
            f'  alignment_confidence: {float(alignment_confidence):.3f}',
            f'  alignment_applied: {bool(alignment_applied)}',
            f'  kept_register_mix: low={kept_low} ({low_rate:.1f}%), mid={kept_mid} ({mid_rate:.1f}%), high={kept_high} ({high_rate:.1f}%)',
        ]
    )
    _set_last_modern_diagnostics_text('\n'.join(lines))

def _pick_channel_index(channels_first: np.ndarray) -> int:
    if channels_first.ndim != 2 or channels_first.shape[0] <= 1:
        return 0
    rms = np.sqrt(np.mean(np.square(channels_first, dtype=np.float32), axis=1) + 1e-12)
    if rms.size == 0:
        return 0
    return int(np.argmax(rms))


def _mixdown_to_mono(audio: np.ndarray, *, modern_strategy: bool) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.ndim != 2:
        return np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] == 1:
        return arr[0].astype(np.float32, copy=False)
    if arr.shape[1] == 1:
        return arr[:, 0].astype(np.float32, copy=False)
    if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        channels_first = arr
    elif arr.shape[1] <= 8 and arr.shape[0] > arr.shape[1]:
        channels_first = arr.T
    else:
        channels_first = arr if arr.shape[0] <= arr.shape[1] else arr.T
    if channels_first.shape[0] <= 1:
        return channels_first.reshape(-1).astype(np.float32, copy=False)
    if not modern_strategy:
        return np.mean(channels_first, axis=0, dtype=np.float32)
    best_idx = _pick_channel_index(channels_first)
    rms = np.sqrt(np.mean(np.square(channels_first, dtype=np.float32), axis=1) + 1e-12)
    if channels_first.shape[0] >= 2:
        order = np.argsort(rms)
        second_idx = int(order[-2])
    else:
        second_idx = best_idx
    best_rms = float(rms[best_idx]) if rms.size > 0 else 0.0
    second_rms = float(rms[second_idx]) if rms.size > 1 else 0.0
    if second_idx != best_idx and second_rms >= best_rms * 0.85:
        w1 = max(0.0, best_rms)
        w2 = max(0.0, second_rms)
        denom = w1 + w2
        if denom <= 1e-8:
            return ((channels_first[best_idx] + channels_first[second_idx]) * 0.5).astype(np.float32, copy=False)
        return ((channels_first[best_idx] * w1 + channels_first[second_idx] * w2) / denom).astype(np.float32, copy=False)
    return channels_first[best_idx].astype(np.float32, copy=False)


def _apply_modern_input_processing(audio: np.ndarray) -> np.ndarray:
    profile = _build_modern_input_profile(audio)
    return _apply_modern_input_profile(audio, profile)


def _build_modern_input_profile(audio: np.ndarray) -> dict[str, float]:
    reference = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if reference.size == 0:
        return {'dc': 0.0, 'gain': 1.0, 'gate': 0.0}
    finite_mask = np.isfinite(reference)
    if not np.all(finite_mask):
        reference[~finite_mask] = 0.0
    dc = float(np.mean(reference))
    reference -= dc
    peak = float(np.max(np.abs(reference)))
    if peak > 1e-8:
        reference /= peak
    rms = float(np.sqrt(np.mean(np.square(reference), dtype=np.float32)))
    gain = _clamp(0.08 / rms, 0.5, 2.0) if rms > 1e-8 else 1.0
    reference *= gain
    gate = _percentile_or_default(np.abs(reference), 20.0, 0.0) * 1.2
    return {'dc': dc, 'gain': float(gain / peak) if peak > 1e-8 else float(gain), 'gate': float(max(0.0, gate))}


def _apply_modern_input_profile(audio: np.ndarray, profile: dict[str, float] | None) -> np.ndarray:
    processed = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if processed.size == 0:
        return processed
    finite_mask = np.isfinite(processed)
    if not np.all(finite_mask):
        processed[~finite_mask] = 0.0
    if not profile:
        profile = {'dc': 0.0, 'gain': 1.0, 'gate': 0.0}
    dc = float(profile.get('dc', 0.0))
    gain = float(profile.get('gain', 1.0))
    gate = float(profile.get('gate', 0.0))
    processed -= dc
    processed *= gain
    if gate > 0:
        quiet = np.abs(processed) < gate
        if np.any(quiet):
            processed[quiet] *= 0.35
    np.clip(processed, -1.0, 1.0, out=processed)
    return processed.astype(np.float32, copy=False)


def _prepare_audio_for_inference(audio: np.ndarray, *, conversion_options: ConversionOptions) -> np.ndarray:
    mono_audio = _mixdown_to_mono(audio, modern_strategy=_is_modern_mode(conversion_options))
    if _modern_input_processing_enabled(conversion_options):
        mono_audio = _apply_modern_input_processing(mono_audio)
    return np.asarray(mono_audio, dtype=np.float32).reshape(-1)


def _transcribe_audio(audio: np.ndarray, *, progress_callback: SegmentProgressCallback | None=None, stop_event: Event | None=None, model_path_override: Path | str | None=None, device_override: str | None=None, batch_size_override: int | None=None, conversion_options: ConversionOptions | None=None) -> tuple[list[dict], list[dict] | None]:
    _ensure_not_stopped(stop_event)
    options = _normalize_conversion_options(conversion_options)
    _model_path, session, output_map, device_preference = _resolve_transcription_runtime(model_path_override=model_path_override, device_override=device_override)
    segment_samples = int(pti_config.sample_rate * _SEGMENT_SECONDS)
    audio = _prepare_audio_for_inference(audio, conversion_options=options).astype(np.float32, copy=False)
    audio_batch = audio[None, :]
    audio_len = audio_batch.shape[1]
    pad_len = int(np.ceil(audio_len / segment_samples)) * segment_samples - audio_len
    if pad_len > 0:
        audio_batch = np.concatenate((audio_batch, np.zeros((1, pad_len), dtype=np.float32)), axis=1)
    segments = _iter_overlapping_segments(audio_batch, segment_samples)
    total_segments = _count_overlapping_segments(audio_batch.shape[1], segment_samples)
    outputs = _run_session_batches(session, output_map, segments, total_segments=total_segments, progress_callback=progress_callback, stop_event=stop_event, batch_size_override=batch_size_override, device_preference=device_preference)
    target_frames = _target_frames_for_audio_length(audio_len)
    _trim_outputs_to_target_frames(outputs, target_frames, smart_overlap_stitching=_modern_overlap_stitching_enabled(options))
    thresholds = _resolve_post_processor_thresholds(outputs, options)
    if _is_modern_mode(options):
        _set_last_modern_diagnostics_text(
            _format_modern_diagnostics(
                conversion_options=options,
                thresholds=thresholds,
                total_segments=total_segments,
            )
        )
    else:
        _set_last_modern_diagnostics_text('')
    post_processor = _create_post_processor(thresholds=thresholds, max_note_frames=None if _is_modern_mode(options) else 600)
    return post_processor.output_dict_to_midi_events(outputs)

def transcribe_audio_array(audio: np.ndarray, *, progress_callback: SegmentProgressCallback | None=None, stop_event: Event | None=None, model_path_override: Path | str | None=None, device_override: str | None=None, batch_size_override: int | None=None, conversion_options: ConversionOptions | None=None) -> tuple[list[dict], list[dict] | None]:
    audio_arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return _transcribe_audio(audio_arr, progress_callback=progress_callback, stop_event=stop_event, model_path_override=model_path_override, device_override=device_override, batch_size_override=batch_size_override, conversion_options=conversion_options)

def _estimate_audio_length_samples(audio_file_path: Path, librosa_module) -> int:
    try:
        duration_seconds = float(librosa_module.get_duration(path=str(audio_file_path)))
    except Exception:
        return 0
    if (not np.isfinite(duration_seconds)) or duration_seconds <= 0:
        return 0
    return max(1, int(round(duration_seconds * float(pti_config.sample_rate))))


def _is_streaming_fallback_error(exc: Exception) -> bool:
    if isinstance(exc, (OSError, EOFError, FileNotFoundError)):
        return True
    if isinstance(exc, (RuntimeError, ValueError)):
        text = str(exc or '').strip().lower()
        markers = ('stream', 'decode', 'backend', 'ffmpeg', 'audioread', 'audio_open', 'samplerate', 'resample')
        return any((marker in text for marker in markers))
    return False


def _iter_streamed_audio_segments(audio_file_path: Path, segment_samples: int, librosa_module, *, conversion_options: ConversionOptions) -> Iterator[np.ndarray]:
    target_sr = int(pti_config.sample_rate)
    native_sr = int(librosa_module.get_samplerate(str(audio_file_path)))
    if native_sr <= 0:
        raise RuntimeError('Unable to determine source sample rate for streaming.')
    frame_length_native = max(1, int(round((float(segment_samples) / float(target_sr)) * float(native_sr))))
    hop_length_native = max(1, frame_length_native // 2)
    modern_mode = _is_modern_mode(conversion_options)
    modern_input_profile: dict[str, float] | None = None
    stream = librosa_module.stream(str(audio_file_path), block_length=1, frame_length=frame_length_native, hop_length=hop_length_native, mono=(not modern_mode), fill_value=0.0, dtype=np.float32)
    for chunk in stream:
        segment = _mixdown_to_mono(np.asarray(chunk, dtype=np.float32), modern_strategy=modern_mode)
        if native_sr != target_sr:
            segment = np.asarray(librosa_module.resample(segment, orig_sr=native_sr, target_sr=target_sr), dtype=np.float32)
        if _modern_input_processing_enabled(conversion_options):
            if modern_input_profile is None:
                modern_input_profile = _build_modern_input_profile(segment)
            segment = _apply_modern_input_profile(segment, modern_input_profile)
        if segment.shape[0] < segment_samples:
            pad_len = segment_samples - segment.shape[0]
            segment = np.concatenate((segment, np.zeros((pad_len,), dtype=np.float32)), axis=0)
        elif segment.shape[0] > segment_samples:
            segment = segment[:segment_samples]
        segment_batch = segment[None, :]
        if segment_batch.dtype != np.float32 or (not segment_batch.flags.c_contiguous):
            segment_batch = np.ascontiguousarray(segment_batch, dtype=np.float32)
        yield segment_batch

def _transcribe_audio_file(audio_file_path: Path, *, progress_callback: SegmentProgressCallback | None=None, stop_event: Event | None=None, model_path_override: Path | str | None=None, device_override: str | None=None, batch_size_override: int | None=None, conversion_options: ConversionOptions | None=None) -> tuple[list[dict], list[dict] | None]:
    _ensure_not_stopped(stop_event)
    options = _normalize_conversion_options(conversion_options)
    librosa = get_librosa_module()
    model_path, session, output_map, device_preference = _resolve_transcription_runtime(model_path_override=model_path_override, device_override=device_override)
    segment_samples = int(pti_config.sample_rate * _SEGMENT_SECONDS)
    audio_len = _estimate_audio_length_samples(audio_file_path, librosa)
    total_segments: int | None = None
    if audio_len > 0:
        padded_samples = int(np.ceil(audio_len / segment_samples)) * segment_samples
        total_segments = _count_overlapping_segments(padded_samples, segment_samples)
    try:
        segments = _iter_streamed_audio_segments(audio_file_path, segment_samples, librosa, conversion_options=options)
        outputs = _run_session_batches(session, output_map, segments, total_segments=total_segments, progress_callback=progress_callback, stop_event=stop_event, batch_size_override=batch_size_override, device_preference=device_preference)
        if audio_len > 0:
            target_frames = _target_frames_for_audio_length(audio_len)
        else:
            sample_key = next(iter(outputs))
            if _modern_overlap_stitching_enabled(options):
                onset_weight_map = _build_overlap_weight_map_from_onset(outputs.get('reg_onset_output'))
                target_frames = int(_deframe_overlap_weighted(outputs[sample_key], weight_map=onset_weight_map).shape[0])
            else:
                target_frames = int(_deframe(outputs[sample_key]).shape[0])
        _trim_outputs_to_target_frames(outputs, target_frames, smart_overlap_stitching=_modern_overlap_stitching_enabled(options))
        thresholds = _resolve_post_processor_thresholds(outputs, options)
        if _is_modern_mode(options):
            _set_last_modern_diagnostics_text(
                _format_modern_diagnostics(
                    conversion_options=options,
                    thresholds=thresholds,
                    total_segments=total_segments,
                )
            )
        else:
            _set_last_modern_diagnostics_text('')
        post_processor = _create_post_processor(thresholds=thresholds, max_note_frames=None if _is_modern_mode(options) else 600)
        return post_processor.output_dict_to_midi_events(outputs)
    except InterruptedError:
        raise
    except Exception as exc:
        if not _is_streaming_fallback_error(exc):
            raise
        modern_mode = _is_modern_mode(options)
        audio, _ = librosa.load(str(audio_file_path), sr=pti_config.sample_rate, mono=(not modern_mode), dtype=np.float32)
        audio_arr = np.asarray(audio, dtype=np.float32)
        return _transcribe_audio(audio_arr, progress_callback=progress_callback, stop_event=stop_event, model_path_override=model_path, device_override=device_override, batch_size_override=batch_size_override, conversion_options=options)

def convert_audio_to_midi(audio_file_path: Path | str, custom_name: str | None=None, min_duration: float=0.02, min_velocity: int=20, progress_callback: SegmentProgressCallback | None=None, stop_event: Event | None=None, conversion_options: ConversionOptions | None=None) -> Path:
    audio_file_path = Path(audio_file_path)
    if not audio_file_path.is_file():
        raise FileNotFoundError(f'Audio file not found: {audio_file_path}')
    _ensure_not_stopped(stop_event)
    options = _normalize_conversion_options(conversion_options)
    _set_last_modern_diagnostics_text('')
    midi_dir = get_output_midi_dir()
    midi_base = safe_filename(custom_name if custom_name else audio_file_path.stem, fallback='midi')
    midi_file_path = ensure_unique_path(midi_dir / f'{midi_base}.mid')
    try:
        est_note_events, est_pedal_events = _transcribe_audio_file(audio_file_path, progress_callback=progress_callback, stop_event=stop_event, conversion_options=options)
        _ensure_not_stopped(stop_event)
        write_events_to_midi(start_time=0, note_events=est_note_events, pedal_events=est_pedal_events, midi_path=str(midi_file_path))
        if _is_modern_mode(options):
            alignment_shift = 0.0
            alignment_confidence = 0.0
            alignment_applied = False
            try:
                estimated_shift, alignment_confidence = _estimate_modern_onset_shift_seconds(
                    audio_file_path=audio_file_path,
                    midi_file_path=Path(midi_file_path),
                    max_shift_seconds=_MODERN_ALIGNMENT_MAX_SHIFT_SECONDS,
                )
                if _should_apply_modern_alignment_shift(
                    audio_file_path=audio_file_path,
                    midi_file_path=Path(midi_file_path),
                    proposed_shift_seconds=estimated_shift,
                    confidence=alignment_confidence,
                    alignment_gate_scale=options.modern_alignment_gate_scale,
                ):
                    alignment_shift = _apply_modern_global_midi_shift(
                        Path(midi_file_path),
                        estimated_shift,
                        stop_event=stop_event,
                    )
                    alignment_applied = abs(float(alignment_shift)) >= float(_MODERN_ALIGNMENT_MIN_APPLY_SECONDS)
            except Exception:
                alignment_shift = 0.0
                alignment_confidence = 0.0
                alignment_applied = False
            modern_min_velocity = _resolve_modern_post_min_velocity(est_note_events, min_velocity)
            modern_min_duration = max(float(min_duration), 0.025)
            post_stats = post_process_midi(
                midi_file_path,
                min_duration=modern_min_duration,
                min_velocity=modern_min_velocity,
                stop_event=stop_event,
                dedupe_same_pitch_overlap=True,
                overlap_tolerance_seconds=0.03,
                pedal_aware_cleanup=True,
                modern_strict_cleanup=True,
                modern_cleanup_scale=options.modern_cleanup_scale,
                modern_pedal_cluster_scale=options.modern_pedal_cluster_scale,
            )
            _append_modern_post_cleanup_diagnostics(
                post_stats=post_stats,
                min_duration=modern_min_duration,
                min_velocity=modern_min_velocity,
                alignment_shift_seconds=alignment_shift,
                alignment_confidence=alignment_confidence,
                alignment_applied=alignment_applied,
            )
        else:
            post_process_midi(
                midi_file_path,
                min_duration=min_duration,
                min_velocity=min_velocity,
                stop_event=stop_event,
                dedupe_same_pitch_overlap=True,
                overlap_tolerance_seconds=0.0,
                pedal_aware_cleanup=False,
                modern_strict_cleanup=False,
            )
        return Path(midi_file_path)
    except InterruptedError:
        try:
            if midi_file_path.exists():
                midi_file_path.unlink()
        except Exception:
            pass
        raise
