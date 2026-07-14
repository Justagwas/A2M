from __future__ import annotations
import threading
import time
from pathlib import Path
from a2m.core.conversion_service import convert_audio_to_midi, format_midi_summary, summarize_midi_file
from .base_worker import WorkerBase
from .payloads import ConversionPayload

class ConversionWorker(WorkerBase):

    def __init__(
        self,
        selected_file: Path | str,
        *,
        include_pedals: bool = True,
        velocity_mode: str = 'expressive',
        uniform_velocity: int = 96,
    ):
        super().__init__()
        self.selected_file = Path(selected_file)
        self.include_pedals = bool(include_pedals)
        self.velocity_mode = str(velocity_mode or 'expressive')
        self.uniform_velocity = int(uniform_velocity)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        started_at = time.monotonic()
        self.reset_progress_eta()
        self.logChanged.emit('Loading audio file...\nPlease wait...')
        self.logChanged.emit(f'Converting audio file:\n{self.selected_file}\nPlease wait...')
        self.emit_progress(0.0, self.progress_text_with_eta('Transcribing', 0.0))

        def on_segment(current: int, total: int) -> None:
            percent = 0.0 if total <= 0 else (float(current) / float(total) * 100.0)
            percent = max(0.0, min(percent, 100.0))
            self.emit_progress(percent, self.progress_text_with_eta('Transcribing', percent))

        def on_status(message: str) -> None:
            text = str(message or '').strip()
            if text:
                self.logChanged.emit(text)

        def task():
            return convert_audio_to_midi(
                self.selected_file,
                progress_callback=on_segment,
                status_callback=on_status,
                stop_event=self._stop_event,
                include_pedals=self.include_pedals,
                velocity_mode=self.velocity_mode,
                uniform_velocity=self.uniform_velocity,
            )

        def on_success(midi_path):
            self.emit_progress(100.0, 'Done')
            processing_seconds = time.monotonic() - started_at
            try:
                summary = summarize_midi_file(midi_path)
                result_text = format_midi_summary(
                    summary,
                    processing_seconds=processing_seconds,
                    midi_path=midi_path,
                )
            except Exception:
                result_text = f'Transcription complete\nSaved to:\n{midi_path}'
            self.logChanged.emit(result_text)
            self.finishedSuccess.emit(ConversionPayload(midi_path=str(midi_path)))

        def on_stopped() -> None:
            self.emit_progress(0.0, 'Stopped')
            self.logChanged.emit('Transcription stopped.')
            self.finishedStopped.emit()

        def on_error(exc: Exception) -> None:
            self.emit_progress(0.0, '0.00%')
            self.logChanged.emit(f'Error: {exc}')
            self.errorRaised.emit(str(exc))

        self.run_task(task, on_success=on_success, on_stopped=on_stopped, on_error=on_error)
