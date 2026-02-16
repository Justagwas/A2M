from __future__ import annotations
import threading
from pathlib import Path
from a2m.core.conversion_service import ConversionOptions, convert_audio_to_midi, get_last_modern_diagnostics_text
from .base_worker import WorkerBase
from .payloads import ConversionPayload

class ConversionWorker(WorkerBase):

    def __init__(self, selected_file: Path | str, *, conversion_options: ConversionOptions | None=None):
        super().__init__()
        self.selected_file = Path(selected_file)
        self._conversion_options = conversion_options
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self.logChanged.emit('Loading audio file...\nPlease wait...')
        self.logChanged.emit(f'Converting audio file:\n{self.selected_file}\nPlease wait...')
        self.emit_progress(0.0, 'Transcribing 0.00%')

        def on_segment(current: int, total: int) -> None:
            percent = 0.0 if total <= 0 else (float(current) / float(total) * 100.0)
            percent = max(0.0, min(percent, 100.0))
            self.emit_progress(percent, f'Transcribing {percent:.2f}%')

        def task():
            return convert_audio_to_midi(self.selected_file, progress_callback=on_segment, stop_event=self._stop_event, conversion_options=self._conversion_options)

        def on_success(midi_path):
            self.emit_progress(100.0, 'Done')
            diagnostics_text = get_last_modern_diagnostics_text()
            if diagnostics_text:
                self.logChanged.emit(f'{diagnostics_text}\n\nDone! Saved at:\n{midi_path}')
            else:
                self.logChanged.emit(f'Done! Saved at:\n{midi_path}')
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
