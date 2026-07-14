from __future__ import annotations
import math
import time
from collections import deque
from collections.abc import Callable
from typing import Any
from PySide6.QtCore import QObject, Signal


def format_eta_seconds(seconds: float | int) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError, OverflowError):
        return ''
    if not math.isfinite(value) or value < 0:
        return ''
    total = max(0, int(round(value)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


class ProgressEtaEstimator:

    def __init__(self, *, sample_window_seconds: float=12.0, refresh_seconds: float=1.0) -> None:
        self._sample_window_seconds = max(2.0, float(sample_window_seconds))
        self._refresh_seconds = max(0.25, float(refresh_seconds))
        self._samples: deque[tuple[float, float]] = deque()
        self._last_eta = ''
        self._last_eta_at = 0.0

    def reset(self, progress: float=0.0) -> None:
        self._samples.clear()
        self._last_eta = ''
        self._last_eta_at = 0.0
        value = max(0.0, min(100.0, float(progress)))
        self._samples.append((time.monotonic(), value))

    def update(self, progress: float | int) -> str:
        value = max(0.0, min(100.0, float(progress)))
        now = time.monotonic()
        if not self._samples:
            self.reset(value)
            return ''
        if value + 0.01 < self._samples[-1][1]:
            self.reset(value)
            return ''
        if value >= 100.0:
            return ''
        if value > self._samples[-1][1] or (now - self._samples[-1][0]) >= 0.35:
            self._samples.append((now, value))
        while len(self._samples) > 2 and (now - self._samples[0][0]) > self._sample_window_seconds:
            self._samples.popleft()
        if self._last_eta and (now - self._last_eta_at) < self._refresh_seconds:
            return self._last_eta
        if len(self._samples) < 2:
            return ''
        oldest_time, oldest_progress = self._samples[0]
        newest_time, newest_progress = self._samples[-1]
        elapsed = newest_time - oldest_time
        completed = newest_progress - oldest_progress
        if elapsed < 1.0 or completed <= 0.0:
            return self._last_eta
        rate = completed / elapsed
        remaining_seconds = (100.0 - newest_progress) / rate
        eta = format_eta_seconds(remaining_seconds)
        if eta:
            self._last_eta = eta
            self._last_eta_at = now
        return self._last_eta


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
        self._progress_eta = ProgressEtaEstimator()

    def reset_progress_eta(self) -> None:
        self._progress_eta.reset(0.0)

    def progress_text_with_eta(self, prefix: str, percent: int | float) -> str:
        value = max(0.0, min(100.0, float(percent)))
        text = f'{str(prefix or "Progress").strip()} {value:.2f}%'
        if value >= 100.0:
            return text
        eta = self._progress_eta.update(value)
        return f'{text} | ETA {eta or "--"}'

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
