from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

from . import piano_engine
from .config import ENGINE_UNIFORM_VELOCITY_DEFAULT, ENGINE_UNIFORM_VELOCITY_MAX, ENGINE_UNIFORM_VELOCITY_MIN, OUTPUT_MIDI_DIR

try:
    from pathvalidate import sanitize_filename
except Exception:
    def sanitize_filename(value: str) -> str:
        return ''.join(character for character in str(value) if character not in '<>:"/\\|?*')


SegmentProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]
_OUTPUT_MIDI_DIR = Path(OUTPUT_MIDI_DIR)
_STOPPED_ERROR = 'Transcription stopped by user.'
WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{index}' for index in range(1, 10)),
    *(f'LPT{index}' for index in range(1, 10)),
}


@dataclass(frozen=True, slots=True)
class MidiSummary:
    note_count: int
    duration_seconds: float
    lowest_pitch: int | None
    highest_pitch: int | None
    pedal_event_count: int


def summarize_midi_file(midi_path: Path | str) -> MidiSummary:
    try:
        import pretty_midi
    except Exception as exc:
        raise RuntimeError(f'Unable to load MIDI summary reader: {exc}') from exc
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    note_count = 0
    lowest_pitch: int | None = None
    highest_pitch: int | None = None
    pedal_events = 0
    for instrument in midi.instruments:
        for note in instrument.notes:
            pitch = int(note.pitch)
            note_count += 1
            lowest_pitch = pitch if lowest_pitch is None else min(lowest_pitch, pitch)
            highest_pitch = pitch if highest_pitch is None else max(highest_pitch, pitch)
        pedal_events += sum(
            1
            for change in instrument.control_changes
            if int(change.number) in {64, 66, 67} and int(change.value) > 0
        )
    return MidiSummary(
        note_count=note_count,
        duration_seconds=max(0.0, float(midi.get_end_time())),
        lowest_pitch=lowest_pitch,
        highest_pitch=highest_pitch,
        pedal_event_count=pedal_events,
    )


def _format_elapsed(seconds: float) -> str:
    raw_seconds = max(0.0, float(seconds))
    total_seconds = max(1, int(round(raw_seconds))) if raw_seconds > 0.0 else 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{seconds_part:02d}'
    return f'{minutes}:{seconds_part:02d}'


def format_midi_summary(summary: MidiSummary, *, processing_seconds: float, midi_path: Path | str) -> str:
    range_text = ''
    if summary.lowest_pitch is not None and summary.highest_pitch is not None:
        try:
            import pretty_midi
            low_name = pretty_midi.note_number_to_name(summary.lowest_pitch)
            high_name = pretty_midi.note_number_to_name(summary.highest_pitch)
            range_text = f' | Pitch: {low_name}' if low_name == high_name else f' | Range: {low_name}-{high_name}'
        except Exception:
            range_text = (
                f' | Pitch: {summary.lowest_pitch}'
                if summary.lowest_pitch == summary.highest_pitch
                else f' | Range: {summary.lowest_pitch}-{summary.highest_pitch}'
            )
    note_label = 'note' if summary.note_count == 1 else 'notes'
    return (
        'Transcription complete\n'
        f'MIDI: {summary.note_count:,} {note_label} | Length: {_format_elapsed(summary.duration_seconds)}{range_text}\n'
        f'Pedals: {summary.pedal_event_count:,} | Processing: {_format_elapsed(processing_seconds)}\n'
        f'Saved to:\n{midi_path}'
    )


def _ensure_not_stopped(stop_event: Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError(_STOPPED_ERROR)


def reset_transcriptor() -> None:
    piano_engine.reset_session_cache()


def get_active_provider_label() -> str:
    return piano_engine.get_active_provider_label()


def safe_filename(name: str, fallback: str = 'audio', max_length: int = 180) -> str:
    raw_name = str(name or '').strip().strip('.')
    cleaned = sanitize_filename(raw_name).strip().strip('.')
    raw_stem = raw_name.split('.', 1)[0].rstrip(' .').upper()
    cleaned_stem = cleaned.split('.', 1)[0].rstrip(' .').upper()
    if not cleaned or raw_stem in WINDOWS_RESERVED_NAMES or cleaned_stem in WINDOWS_RESERVED_NAMES:
        cleaned = fallback
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned or fallback


def ensure_unique_path(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.with_suffix('')
    extension = path.suffix
    candidates = [path, *(Path(f'{base}_{index}{extension}') for index in range(2, 1000))]
    for candidate in candidates:
        try:
            descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        else:
            os.close(descriptor)
            return candidate
    raise RuntimeError(f'Unable to find a unique filename for {path}.')


def _normalize_output_midi_dir(path: Path | str | None) -> Path:
    candidate = str(path or '').strip()
    target = Path(candidate).expanduser() if candidate else Path(OUTPUT_MIDI_DIR)
    try:
        target = target.resolve()
    except Exception:
        target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_output_midi_dir(path: Path | str | None) -> Path:
    global _OUTPUT_MIDI_DIR
    _OUTPUT_MIDI_DIR = _normalize_output_midi_dir(path)
    return Path(_OUTPUT_MIDI_DIR)


def get_output_midi_dir() -> Path:
    target = Path(_OUTPUT_MIDI_DIR)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_midi(
    events: list[piano_engine.PianoEvent],
    midi_path: Path,
    *,
    include_pedals: bool = True,
    velocity_mode: str = 'expressive',
    uniform_velocity: int = ENGINE_UNIFORM_VELOCITY_DEFAULT,
) -> None:
    try:
        import pretty_midi
    except Exception as exc:
        raise RuntimeError(f'Unable to load MIDI writer: {exc}') from exc
    output = pretty_midi.PrettyMIDI(resolution=960)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'),
        name='A2M Piano',
    )
    normalized_velocity_mode = 'uniform' if str(velocity_mode).strip().lower() == 'uniform' else 'expressive'
    fixed_velocity = max(ENGINE_UNIFORM_VELOCITY_MIN, min(ENGINE_UNIFORM_VELOCITY_MAX, int(uniform_velocity)))
    for event in events:
        velocity = max(1, min(127, int(event.velocity)))
        if event.pitch > 0:
            if normalized_velocity_mode == 'uniform':
                velocity = fixed_velocity
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=max(0, min(127, int(event.pitch))),
                    start=max(0.0, float(event.start)),
                    end=max(float(event.start) + 1e-8, float(event.end)),
                )
            )
        else:
            if not include_pedals:
                continue
            control = -int(event.pitch)
            pedal_on_value = max(64, velocity)
            instrument.control_changes.append(pretty_midi.ControlChange(control, pedal_on_value, max(0.0, float(event.start))))
            instrument.control_changes.append(pretty_midi.ControlChange(control, 0, max(0.0, float(event.end))))
    instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    instrument.control_changes.sort(key=lambda change: (change.time, change.number, change.value))
    output.instruments.append(instrument)
    output.write(str(midi_path))


def convert_audio_to_midi(
    audio_file_path: Path | str,
    custom_name: str | None = None,
    *,
    progress_callback: SegmentProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
    stop_event: Event | None = None,
    include_pedals: bool = True,
    velocity_mode: str = 'expressive',
    uniform_velocity: int = ENGINE_UNIFORM_VELOCITY_DEFAULT,
) -> Path:
    audio_path = Path(audio_file_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f'Audio file not found: {audio_path}')
    _ensure_not_stopped(stop_event)
    midi_name = safe_filename(custom_name if custom_name else audio_path.stem, fallback='midi')
    midi_path = ensure_unique_path(get_output_midi_dir() / f'{midi_name}.mid')
    try:
        events = piano_engine.transcribe_audio_file(
            audio_path,
            progress_callback=progress_callback,
            status_callback=status_callback,
            stop_event=stop_event,
        )
        _ensure_not_stopped(stop_event)
        _write_midi(
            events,
            midi_path,
            include_pedals=include_pedals,
            velocity_mode=velocity_mode,
            uniform_velocity=uniform_velocity,
        )
        return midi_path
    except Exception:
        try:
            midi_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
