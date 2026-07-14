from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from . import resource_service, runtime_service
from .model_service import get_existing_model_bundle_dir


ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]
_SESSION_CACHE: dict[tuple[str, str, str, str], tuple[object, object, str]] = {}
_GPU_BATCH_LIMIT_CACHE: dict[tuple[str, str], int] = {}
_ACTIVE_PROVIDER = 'CPU'


@dataclass(slots=True)
class PianoEvent:
    start: float
    end: float
    pitch: int
    velocity: int
    has_onset: bool = True
    has_offset: bool = True


@dataclass(slots=True, frozen=True)
class PianoModelSpec:
    bundle_dir: Path
    scorer_path: Path
    attributes_path: Path
    frontend_path: Path
    sample_rate: int
    hop_size: int
    window_size: int
    segment_seconds: float
    segment_hop_seconds: float
    symbols: tuple[int, ...]


def _ensure_not_stopped(stop_event: Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError('Transcription stopped by user.')


def _load_spec(model_path_override: Path | str | None = None) -> PianoModelSpec:
    if model_path_override:
        scorer_path = Path(model_path_override)
        bundle_dir = scorer_path.parent
    else:
        bundle_dir = get_existing_model_bundle_dir()
        if bundle_dir is None:
            raise RuntimeError('The Transcription Model package is missing. Please download it first.')
        scorer_path = bundle_dir / 'scorer.onnx'
    manifest_path = bundle_dir / 'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise RuntimeError(f'A2M Piano Engine manifest is invalid: {exc}') from exc
    attributes_path = bundle_dir / 'attributes.onnx'
    frontend_path = bundle_dir / 'frontend.npz'
    for required in (scorer_path, attributes_path, frontend_path):
        if not required.is_file():
            raise RuntimeError(f'A2M Piano Engine component is missing: {required.name}')
    try:
        symbols = tuple(int(value) for value in manifest.get('symbols', ()))
        sample_rate = int(manifest['sample_rate'])
        hop_size = int(manifest['hop_size'])
        window_size = int(manifest['window_size'])
        segment_seconds = float(manifest['segment_seconds'])
        segment_hop_seconds = float(manifest['segment_hop_seconds'])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f'A2M Piano Engine manifest settings are invalid: {exc}') from exc
    if len(symbols) != 90 or symbols[:2] != (-64, -67) or symbols[2:] != tuple(range(21, 109)):
        raise RuntimeError('A2M Piano Engine symbol map is incompatible with this application.')
    if sample_rate <= 0 or hop_size <= 0 or window_size <= 0:
        raise RuntimeError('A2M Piano Engine manifest contains invalid audio dimensions.')
    if not math.isfinite(segment_seconds) or not math.isfinite(segment_hop_seconds):
        raise RuntimeError('A2M Piano Engine manifest contains non-finite segment timing.')
    if segment_seconds <= 0.0 or segment_hop_seconds <= 0.0 or segment_hop_seconds > segment_seconds:
        raise RuntimeError('A2M Piano Engine manifest contains invalid segment timing.')
    return PianoModelSpec(
        bundle_dir=bundle_dir,
        scorer_path=Path(scorer_path),
        attributes_path=attributes_path,
        frontend_path=frontend_path,
        sample_rate=sample_rate,
        hop_size=hop_size,
        window_size=window_size,
        segment_seconds=segment_seconds,
        segment_hop_seconds=segment_hop_seconds,
        symbols=symbols,
    )


def _resolve_device(device_override: str | None) -> tuple[str, str]:
    normalized = str(device_override or '').strip().lower()
    provider = runtime_service.get_gpu_provider_preference()
    if normalized in {'cpu'}:
        return 'cpu', provider
    if normalized in {'gpu'}:
        return 'gpu', provider
    if normalized in {'cuda'}:
        return 'gpu', 'cuda'
    if normalized in {'dml', 'directml'}:
        return 'gpu', 'dml'
    return runtime_service.get_device_preference(), provider


def _session_key(spec: PianoModelSpec, device: str, provider: str) -> tuple[str, str, str, str]:
    runtime_path = runtime_service.get_gpu_runtime_path() if runtime_service.is_gpu_runtime_enabled() else ''
    return str(spec.bundle_dir.resolve()), str(device), str(provider), str(runtime_path)


def _get_sessions(spec: PianoModelSpec, device_override: str | None = None):
    global _ACTIVE_PROVIDER
    device, provider = _resolve_device(device_override)
    key = _session_key(spec, device, provider)
    cached = _SESSION_CACHE.get(key)
    if cached is not None:
        _ACTIVE_PROVIDER = str(cached[2] or 'CPU')
        return cached
    scorer, scorer_provider = runtime_service.create_session(
        spec.scorer_path,
        device_preference=device,
        gpu_provider_preference=provider,
    )
    attributes, attributes_provider = runtime_service.create_session(
        spec.attributes_path,
        device_preference=device,
        gpu_provider_preference=provider,
    )
    active_provider = scorer_provider if scorer_provider == attributes_provider else f'{scorer_provider} + {attributes_provider}'
    _SESSION_CACHE.clear()
    _SESSION_CACHE[key] = scorer, attributes, active_provider
    _ACTIVE_PROVIDER = str(active_provider or 'CPU')
    return _SESSION_CACHE[key]


def reset_session_cache() -> None:
    global _ACTIVE_PROVIDER
    _SESSION_CACHE.clear()
    _ACTIVE_PROVIDER = 'CPU'


def get_active_provider_label() -> str:
    return str(_ACTIVE_PROVIDER or 'CPU')


def _make_frames(audio_channels: np.ndarray, *, hop_size: int, window_size: int) -> np.ndarray:
    samples = int(audio_channels.shape[-1])
    frame_count = int(math.ceil(samples / float(hop_size))) + 1
    left = window_size // 2
    right = (frame_count - 1) * hop_size + window_size - samples - left
    padded = np.pad(audio_channels, ((0, 0), (left, max(0, right))), mode='constant')
    shape = (padded.shape[0], frame_count, window_size)
    strides = (padded.strides[0], padded.strides[1] * hop_size, padded.strides[1])
    return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides, writeable=False)


def _extract_features(
    audio_channels: np.ndarray,
    *,
    spec: PianoModelSpec,
    windows: np.ndarray,
    mel_filter: np.ndarray,
    stop_event: Event | None,
) -> np.ndarray:
    try:
        from scipy.fft import rfft
    except Exception as exc:
        raise RuntimeError(f'SciPy FFT support is required by the A2M Piano Engine: {exc}') from exc
    frames = _make_frames(audio_channels, hop_size=spec.hop_size, window_size=spec.window_size)
    mean = np.mean(frames, dtype=np.float32)
    std = np.std(frames, dtype=np.float32, ddof=1)
    normalized = (frames - mean) / (std + np.float32(1e-8))
    frame_count = int(frames.shape[1])
    features = np.zeros((1, frame_count, mel_filter.shape[1], windows.shape[0]), dtype=np.float32)
    worker_count = resource_service.active_worker_count()
    with threadpool_limits(limits=worker_count):
        for channel in range(normalized.shape[0]):
            _ensure_not_stopped(stop_event)
            windowed = normalized[channel, :, None, :] * windows[None, :, :]
            spectrum = rfft(windowed, axis=-1, norm='ortho', workers=worker_count)
            power = np.square(np.abs(spectrum), dtype=np.float32)
            projected = power.reshape(frame_count * windows.shape[0], power.shape[-1]) @ mel_filter
            features[0] += projected.reshape(frame_count, windows.shape[0], mel_filter.shape[1]).transpose(0, 2, 1)
    features /= float(max(1, normalized.shape[0]))
    epsilon = np.float32(1e-5)
    features = (np.log(features + epsilon) - math.log(float(epsilon))) / (-math.log(float(epsilon)))
    return np.ascontiguousarray(features, dtype=np.float32)


def _decode_viterbi_backward(
    score: np.ndarray,
    noise_score: np.ndarray,
    forced_start: list[int],
    *,
    stop_event: Event | None,
) -> tuple[list[list[tuple[int, int]]], list[int]]:
    if score.ndim != 3 or score.shape[0] != score.shape[1]:
        raise RuntimeError(f'Invalid interval-score shape: {score.shape}')
    time_steps, _, tracks = score.shape
    q = np.zeros((time_steps, tracks), dtype=np.float32)
    diagonal = score[np.arange(time_steps), np.arange(time_steps), :]
    q[-1] = diagonal[-1] * (diagonal[-1] > 0)
    pointers = np.empty((time_steps - 1, tracks), dtype=np.int32)
    for distance in range(1, time_steps):
        if distance % 16 == 0:
            _ensure_not_stopped(stop_event)
        begin = time_steps - distance - 1
        candidates = np.concatenate(
            (
                q[begin + 1:begin + 2] + noise_score[begin:begin + 1],
                q[begin + 1:] + score[begin + 1:, begin, :],
            ),
            axis=0,
        )
        selection = np.argmax(candidates, axis=0)
        q[begin] = candidates[selection, np.arange(tracks)] + diagonal[begin] * (diagonal[begin] > 0)
        pointers[distance - 1] = selection.astype(np.int32) - 1
    paths: list[list[tuple[int, int]]] = []
    last_positions: list[int] = []
    for track in range(tracks):
        position = max(0, min(int(forced_start[track]), time_steps - 1))
        intervals: list[tuple[int, int]] = []
        last_position = 0
        while position < time_steps - 1:
            selection = int(pointers[time_steps - position - 2, track])
            if diagonal[position, track] > 0:
                intervals.append((position, position))
            if selection < 0:
                position += 1
            else:
                end = selection + position + 1
                intervals.append((position, end))
                position = end
                last_position = end
        if diagonal[-1, track] > 0:
            intervals.append((time_steps - 1, time_steps - 1))
        paths.append(intervals)
        last_positions.append(last_position)
    return paths, last_positions


def _continuous_bernoulli_offset(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    near_zero = np.abs(values) < 1e-3
    x = values[near_zero]
    result[near_zero] = 0.5 + x / 12.0 - (x**3) / 720.0
    x = values[~near_zero]
    result[~near_zero] = 0.5 + 0.5 / np.tanh(x / 2.0) - 1.0 / x
    return np.clip((result - 0.5) / 0.99, -0.5, 0.5).astype(np.float32)


def _events_from_segment(
    context: np.ndarray,
    paths: list[list[tuple[int, int]]],
    attributes_session,
    *,
    symbols: tuple[int, ...],
    frame_duration: float,
    last_frame_index: int,
    stop_event: Event | None,
) -> tuple[list[PianoEvent], list[int]]:
    rows = [(symbol_index, start, end) for symbol_index, intervals in enumerate(paths) for start, end in intervals]
    if not rows:
        return [], [0] * len(symbols)
    _ensure_not_stopped(stop_event)
    velocity_logits, refined_logits = attributes_session.run(
        ['velocity_logits', 'refined_logits'],
        {
            'context': np.ascontiguousarray(context, dtype=np.float32),
            'interval_indices': np.ascontiguousarray(rows, dtype=np.int64),
        },
    )
    velocities = np.argmax(velocity_logits, axis=-1).astype(np.int32)
    offsets = _continuous_bernoulli_offset(np.asarray(refined_logits)[:, :2])
    presence = np.asarray(refined_logits)[:, 2:] > 0
    events: list[PianoEvent] = []
    last_positions: list[int] = []
    cursor = 0
    for symbol_index, intervals in enumerate(paths):
        last_end = 0.0
        last_position = 0
        for start_frame, end_frame in intervals:
            start = (float(start_frame) + float(offsets[cursor, 0])) * frame_duration
            end = (float(end_frame) + float(offsets[cursor, 1])) * frame_duration
            start = max(start, last_end)
            end = max(end, start + 1e-8)
            last_end = end
            event = PianoEvent(
                    start=start,
                    end=end,
                    pitch=int(symbols[symbol_index]),
                    velocity=int(velocities[cursor]),
                    has_onset=bool(start_frame > 0 or presence[cursor, 0]),
                    has_offset=bool(end_frame < last_frame_index or presence[cursor, 1]),
            )
            events.append(event)
            if event.has_offset:
                last_position = int(end_frame)
            cursor += 1
        last_positions.append(last_position)
    events.sort(key=lambda event: (event.start, event.end, event.pitch))
    return events, last_positions


def _resolve_overlaps(events: list[PianoEvent]) -> list[PianoEvent]:
    events.sort(key=lambda event: (event.start, event.end, event.pitch))
    result: list[PianoEvent] = []
    last_by_pitch: dict[int, int] = {}
    for event in events:
        previous_index = last_by_pitch.get(event.pitch)
        if previous_index is not None and result[previous_index].end > event.start:
            result[previous_index].end = event.start
        last_by_pitch[event.pitch] = len(result)
        result.append(event)
    return [event for event in result if event.start < event.end]


def _bound_events_to_audio_duration(events: list[PianoEvent], duration_seconds: float) -> list[PianoEvent]:
    duration = max(0.0, float(duration_seconds))
    bounded: list[PianoEvent] = []
    for event in events:
        if event.start >= duration:
            continue
        event.start = max(0.0, float(event.start))
        event.end = min(duration, max(event.start, float(event.end)))
        if event.start < event.end:
            bounded.append(event)
    return bounded


def transcribe_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    *,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
    stop_event: Event | None = None,
    model_path_override: Path | str | None = None,
    device_override: str | None = None,
) -> list[PianoEvent]:
    _ensure_not_stopped(stop_event)
    spec = _load_spec(model_path_override)
    device, _requested_provider = _resolve_device(device_override)
    scorer_session, attributes_session, active_provider = _get_sessions(spec, device_override)
    audio_arr = np.asarray(audio, dtype=np.float32)
    if audio_arr.ndim == 1:
        audio_arr = audio_arr[:, None]
    if audio_arr.ndim != 2:
        raise RuntimeError('Audio must have samples by channels shape.')
    if audio_arr.shape[0] == 0 or audio_arr.shape[1] == 0:
        raise RuntimeError('Audio file contains no decodable samples.')
    if int(sample_rate) <= 0:
        raise RuntimeError(f'Audio sample rate is invalid: {sample_rate}')
    source_duration_seconds = audio_arr.shape[0] / float(sample_rate)
    if not np.isfinite(audio_arr).all():
        audio_arr = np.nan_to_num(audio_arr, copy=True, nan=0.0, posinf=1.0, neginf=-1.0)
    if int(sample_rate) != spec.sample_rate:
        try:
            import soxr
            audio_arr = np.asarray(soxr.resample(audio_arr, int(sample_rate), spec.sample_rate), dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f'Unable to resample audio for the A2M Piano Engine: {exc}') from exc
    channels = np.ascontiguousarray(audio_arr.T, dtype=np.float32)
    with np.load(spec.frontend_path, allow_pickle=False) as frontend:
        windows = np.ascontiguousarray(frontend['windows'], dtype=np.float32)
        mel_filter = np.ascontiguousarray(frontend['mel_filter'], dtype=np.float32)
    padding_seconds = spec.segment_seconds - spec.segment_hop_seconds
    pad_samples = int(math.ceil(padding_seconds * spec.sample_rate))
    channels = np.pad(channels, ((0, 0), (pad_samples, pad_samples)), mode='constant')
    step_samples = int(math.ceil(spec.segment_hop_seconds * spec.sample_rate / spec.hop_size) * spec.hop_size)
    segment_samples = int(math.ceil(spec.segment_seconds * spec.sample_rate))
    segment_starts = list(range(0, channels.shape[-1], step_samples))
    forced_start = [int(math.floor(padding_seconds * spec.sample_rate / spec.hop_size))] * len(spec.symbols)
    events_by_type: dict[int, list[PianoEvent]] = defaultdict(list)
    frame_duration = spec.hop_size / float(spec.sample_rate)
    last_frame_index = round(segment_samples / spec.hop_size)
    requested_batch = runtime_service.get_gpu_batch_size() if device == 'gpu' else 1
    batch_key = (str(spec.bundle_dir.resolve()), str(active_provider or 'GPU'))
    effective_batch = min(requested_batch, _GPU_BATCH_LIMIT_CACHE.get(batch_key, requested_batch))
    effective_batch = max(1, int(effective_batch))
    if status_callback is not None and device == 'gpu':
        status_callback(f'GPU batch: {effective_batch}')
    segment_cursor = 0
    while segment_cursor < len(segment_starts):
        _ensure_not_stopped(stop_event)
        batch_count = min(effective_batch, len(segment_starts) - segment_cursor)
        feature_batch: list[np.ndarray] = []
        for offset in range(batch_count):
            start_sample = segment_starts[segment_cursor + offset]
            segment = channels[:, start_sample:start_sample + segment_samples]
            if segment.shape[-1] < segment_samples:
                segment = np.pad(segment, ((0, 0), (0, segment_samples - segment.shape[-1])), mode='constant')
            feature_batch.append(
                _extract_features(
                    segment,
                    spec=spec,
                    windows=windows,
                    mel_filter=mel_filter,
                    stop_event=stop_event,
                )
            )
        batched_features = np.ascontiguousarray(np.concatenate(feature_batch, axis=0), dtype=np.float32)
        try:
            interval_scores, skip_scores, context = scorer_session.run(
                ['interval_scores', 'skip_scores', 'context'],
                {'mel_features': batched_features},
            )
        except Exception:
            if device != 'gpu' or batch_count <= 1:
                raise
            reduced_batch = max(1, (batch_count + 1) // 2)
            effective_batch = min(effective_batch, reduced_batch)
            _GPU_BATCH_LIMIT_CACHE[batch_key] = effective_batch
            if status_callback is not None:
                status_callback(f'GPU batch reduced to {effective_batch} because the larger batch failed.')
            del batched_features, feature_batch
            continue

        interval_array = np.asarray(interval_scores, dtype=np.float32)
        skip_array = np.asarray(skip_scores, dtype=np.float32)
        context_array = np.asarray(context, dtype=np.float32)
        for batch_index in range(batch_count):
            segment_index = segment_cursor + batch_index
            start_sample = segment_starts[segment_index]
            scores = interval_array[:, :, batch_index, :]
            noise = skip_array[:, batch_index, :]
            paths, _decoded_positions = _decode_viterbi_backward(scores, noise, forced_start, stop_event=stop_event)
            segment_events, last_positions = _events_from_segment(
                context_array[batch_index:batch_index + 1],
                paths,
                attributes_session,
                symbols=spec.symbols,
                frame_duration=frame_duration,
                last_frame_index=last_frame_index,
                stop_event=stop_event,
            )
            forced_start = [max(position - int(step_samples / spec.hop_size), 0) for position in last_positions]
            begin_time = start_sample / float(spec.sample_rate) - padding_seconds
            for event in segment_events:
                event.start = max(0.0, event.start + begin_time)
                event.end = max(event.start, event.end + begin_time)
                existing = events_by_type[event.pitch]
                if existing and event.start < existing[-1].end:
                    if event.has_onset:
                        existing[-1] = event
                    else:
                        existing[-1].has_offset = event.has_offset
                        existing[-1].end = max(existing[-1].end, event.end)
                    continue
                if event.has_onset:
                    existing.append(event)
            if progress_callback:
                progress_callback(segment_index + 1, len(segment_starts))
        segment_cursor += batch_count
    for events in events_by_type.values():
        if events:
            events[-1].has_offset = True
    result = [event for events in events_by_type.values() for event in events if event.has_offset]
    return _bound_events_to_audio_duration(_resolve_overlaps(result), source_duration_seconds)


def transcribe_audio_file(
    audio_path: Path | str,
    *,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
    stop_event: Event | None = None,
    model_path_override: Path | str | None = None,
    device_override: str | None = None,
) -> list[PianoEvent]:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f'Audio file not found: {path}')
    _ensure_not_stopped(stop_event)
    try:
        import soundfile as sf
        samples, sample_rate = sf.read(str(path), dtype='float32', always_2d=True)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
    except Exception as soundfile_error:
        try:
            import audioread
            with audioread.audio_open(str(path)) as source:
                sample_rate = int(source.samplerate)
                channel_count = int(source.channels)
                if sample_rate <= 0 or channel_count <= 0:
                    raise RuntimeError('the decoder returned invalid audio metadata')
                pcm_bytes = bytearray()
                for chunk in source:
                    _ensure_not_stopped(stop_event)
                    pcm_bytes.extend(chunk)
            if len(pcm_bytes) < 2:
                raise RuntimeError('the decoder returned no audio samples')
            if len(pcm_bytes) % 2:
                del pcm_bytes[-1]
            interleaved = np.frombuffer(pcm_bytes, dtype='<i2')
            usable = interleaved.size - (interleaved.size % channel_count)
            if usable <= 0:
                raise RuntimeError('the decoder returned no complete audio frames')
            samples = np.ascontiguousarray(
                interleaved[:usable].reshape(-1, channel_count).astype(np.float32) / 32768.0,
                dtype=np.float32,
            )
            del interleaved, pcm_bytes
        except InterruptedError:
            raise
        except Exception as fallback_error:
            raise RuntimeError(
                f'Unable to decode audio file {path.name}: {soundfile_error}. '
                f'Fallback decoder: {fallback_error}'
            ) from fallback_error
    return transcribe_audio_array(
        samples,
        int(sample_rate),
        progress_callback=progress_callback,
        status_callback=status_callback,
        stop_event=stop_event,
        model_path_override=model_path_override,
        device_override=device_override,
    )
