from __future__ import annotations
import time
from collections.abc import Callable
from typing import Any
from PySide6.QtCore import QObject, Signal

class WorkerBase(QObject):
    progressChanged = Signal(float, str)
    logChanged = Signal(str)
    errorRaised = Signal(str)
    finishedSuccess = Signal(object)
    finishedStopped = Signal()
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._last_progress_emit = 0.0

    def emit_progress(self, percent: int | float, text: str) -> None:
        now = time.monotonic()
        value = max(0.0, min(100.0, float(percent)))
        if value in {0.0, 100.0} or now - self._last_progress_emit >= 0.1:
            self._last_progress_emit = now
            self.progressChanged.emit(value, text)

    def run_task(self, task: Callable[[], Any], *, on_success: Callable[[Any], None] | None=None, on_stopped: Callable[[], None] | None=None, on_error: Callable[[Exception], None] | None=None) -> None:
        try:
            result = task()
        except InterruptedError:
            if on_stopped is not None:
                on_stopped()
        except Exception as exc:
            if on_error is not None:
                on_error(exc)
        else:
            if on_success is not None:
                on_success(result)
        finally:
            self.finished.emit()
