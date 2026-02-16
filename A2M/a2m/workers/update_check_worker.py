from __future__ import annotations
import threading
from a2m.core.constants import APP_VERSION
from a2m.core.update_service import fetch_update_manifest, is_newer_version, normalize_version
from .base_worker import WorkerBase
from .payloads import UpdateCheckPayload

class UpdateCheckWorker(WorkerBase):

    def __init__(self, *, manual: bool):
        super().__init__()
        self.manual = manual
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        def task():
            latest_version, download_url = fetch_update_manifest(stop_event=self._stop_event)
            return UpdateCheckPayload(
                manual=self.manual,
                latest_version=latest_version,
                latest_display=normalize_version(latest_version) or str(latest_version),
                download_url=download_url,
                update_available=is_newer_version(latest_version, APP_VERSION),
            )

        def on_success(payload) -> None:
            self.finishedSuccess.emit(payload)

        def on_stopped() -> None:
            self.finishedStopped.emit()

        def on_error(exc: Exception) -> None:
            if self.manual:
                self.errorRaised.emit(f'Unable to check for updates:\n{exc}')

        self.run_task(task, on_success=on_success, on_stopped=on_stopped, on_error=on_error)
