from __future__ import annotations

import threading

from a2m.core.runtime_pack_service import download_and_install_pack, provider_display_name
from .base_worker import WorkerBase
from .payloads import RuntimePackDownloadPayload


class RuntimePackDownloadWorker(WorkerBase):

    def __init__(self, provider: str) -> None:
        super().__init__()
        self.provider = str(provider or '').strip().lower()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        provider_label = provider_display_name(self.provider)
        self.logChanged.emit(f'Downloading {provider_label} runtime pack...\nPlease wait...')

        def on_progress(percent: float) -> None:
            value = max(0.0, min(float(percent), 100.0))
            self.emit_progress(value, f'Downloading {provider_label} runtime pack {value:.2f}%')

        def task():
            return download_and_install_pack(self.provider, progress_callback=on_progress, stop_event=self._stop_event)

        def on_success(runtime_path):
            self.emit_progress(100.0, 'Downloaded')
            self.logChanged.emit(f'{provider_label} runtime pack installed:\n{runtime_path}')
            self.finishedSuccess.emit(RuntimePackDownloadPayload(provider=self.provider, runtime_path=str(runtime_path)))

        def on_stopped() -> None:
            self.emit_progress(0.0, 'Stopped')
            self.logChanged.emit(f'{provider_label} runtime pack download stopped.')
            self.finishedStopped.emit()

        def on_error(exc: Exception) -> None:
            self.emit_progress(0.0, '0.00%')
            self.logChanged.emit(f'{provider_label} runtime pack download failed: {exc}')
            self.errorRaised.emit(str(exc))

        self.run_task(task, on_success=on_success, on_stopped=on_stopped, on_error=on_error)
