from __future__ import annotations

import threading
import time

from a2m.core import update_service
from a2m.core.config import APP_VERSION
from PySide6.QtCore import Signal

from .base_worker import WorkerBase


class UpdateInstallWorker(WorkerBase):
    handoffRequested = Signal(object)

    def __init__(self, *, payload: dict[str, object]) -> None:
        super().__init__()
        self.payload = dict(payload)
        self._stop_event = threading.Event()
        self._handoff_event = threading.Event()
        self._handoff_continue = False
        self._handoff_restart = True

    def stop(self) -> None:
        self._stop_event.set()
        self._handoff_event.set()

    def set_handoff_decision(self, *, continue_update: bool, restart_after_update: bool) -> None:
        self._handoff_continue = bool(continue_update)
        self._handoff_restart = bool(restart_after_update)
        self._handoff_event.set()

    def run(self) -> None:
        def task() -> dict[str, object]:
            install_payload = dict(self.payload)
            install_payload.setdefault('update_available', True)
            install_payload.setdefault('current_version', APP_VERSION)
            self.emit_progress(0.0, 'Preparing update...')
            prepared = None
            try:
                prepared = update_service.prepare_update_from_payload(
                    install_payload,
                    stop_event=self._stop_event,
                    progress_callback=self.emit_progress,
                )
                self.emit_progress(99.0, 'Ready to hand off to installer. Waiting for your confirmation...')
                self._handoff_continue = False
                self._handoff_restart = True
                self._handoff_event.clear()
                self.handoffRequested.emit(
                    {
                        'version': str(getattr(prepared, 'latest_version', '') or ''),
                        'requires_elevation': bool(getattr(prepared, 'requires_elevation', False)),
                    }
                )
                while not self._handoff_event.is_set():
                    if self._stop_event.is_set():
                        raise InterruptedError('Update operation stopped.')
                    time.sleep(0.05)
                if not self._handoff_continue:
                    update_service.discard_prepared_update(prepared)
                    return {
                        'status': 'aborted',
                        'url': str(self.payload.get('url') or ''),
                    }
                update_service.launch_prepared_update(
                    prepared,
                    restart_after_update=bool(self._handoff_restart),
                )
                return {
                    'status': 'ready',
                    'version': str(getattr(prepared, 'latest_version', '') or ''),
                    'restart_after_update': bool(self._handoff_restart),
                }
            except InterruptedError:
                if prepared is not None:
                    try:
                        update_service.discard_prepared_update(prepared)
                    except Exception:
                        pass
                raise
            except Exception:
                if prepared is not None:
                    try:
                        update_service.discard_prepared_update(prepared)
                    except Exception:
                        pass
                raise

        def on_success(payload: dict[str, object]) -> None:
            self.finishedSuccess.emit(payload)

        def on_stopped() -> None:
            self.finishedStopped.emit()

        def on_error(exc: Exception) -> None:
            self.errorRaised.emit(str(exc))

        self.run_task(task, on_success=on_success, on_stopped=on_stopped, on_error=on_error)
