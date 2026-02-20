from __future__ import annotations

import threading

from a2m.core import update_service
from a2m.core.config import APP_VERSION, OFFICIAL_PAGE_URL

from .base_worker import WorkerBase


class UpdateCheckWorker(WorkerBase):

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        def task() -> dict[str, object]:
            check = update_service.check_for_updates(APP_VERSION, stop_event=self._stop_event)
            if check.update_available:
                return {
                    'status': 'available',
                    'latest': str(check.latest_version or ''),
                    'url': str(check.page_url or OFFICIAL_PAGE_URL),
                    'setup_url': str(check.setup_url or ''),
                    'setup_sha256': str(check.setup_sha256 or ''),
                    'setup_size': int(check.setup_size or 0),
                    'released': str(check.released or ''),
                    'notes': list(check.notes or []),
                    'install_supported': bool(check.install_supported),
                    'setup_managed_install': bool(check.setup_managed_install),
                    'channel': str(check.channel or 'stable'),
                    'minimum_supported_version': str(check.minimum_supported_version or '1.0.0'),
                    'requires_manual_update': bool(check.requires_manual_update),
                }
            return {
                'status': 'up_to_date',
                'latest': str(check.latest_version or ''),
                'url': str(check.page_url or OFFICIAL_PAGE_URL),
            }

        def on_success(payload: dict[str, object]) -> None:
            self.finishedSuccess.emit(payload)

        def on_stopped() -> None:
            self.finishedStopped.emit()

        def on_error(exc: Exception) -> None:
            self.errorRaised.emit(str(exc))

        self.run_task(task, on_success=on_success, on_stopped=on_stopped, on_error=on_error)
