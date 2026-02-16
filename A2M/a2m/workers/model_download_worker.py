from __future__ import annotations
import threading
from a2m.core.model_service import ensure_model_file
from .base_worker import WorkerBase
from .payloads import ModelDownloadPayload

class ModelDownloadWorker(WorkerBase):

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self.logChanged.emit('Downloading model...\nPlease wait...')

        def on_progress(percent: float) -> None:
            value = max(0.0, min(float(percent), 100.0))
            self.emit_progress(value, f'Downloading {value:.2f}%')

        def task():
            return ensure_model_file(progress_callback=on_progress, log_callback=self.logChanged.emit, stop_event=self._stop_event)

        def on_success(model_path):
            self.emit_progress(100.0, 'Downloaded')
            self.logChanged.emit('Model download complete.')
            self.finishedSuccess.emit(ModelDownloadPayload(model_path=str(model_path)))

        def on_stopped() -> None:
            self.emit_progress(0.0, 'Stopped')
            self.logChanged.emit('Model download stopped.')
            self.finishedStopped.emit()

        def on_error(exc: Exception) -> None:
            self.emit_progress(0.0, '0.00%')
            self.logChanged.emit(f'Model download failed: {exc}')
            self.errorRaised.emit(f'Model download failed: {exc}')

        self.run_task(task, on_success=on_success, on_stopped=on_stopped, on_error=on_error)
