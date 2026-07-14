from __future__ import annotations

import threading

from a2m.core import cuda_dependency_service, runtime_service
from .base_worker import WorkerBase
from .payloads import CudnnInstallPayload


class CudnnInstallWorker(WorkerBase):

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self.reset_progress_eta()
        self.emit_progress(0.0, self.progress_text_with_eta('Downloading', 0.0))
        requirement_text = cuda_dependency_service.cuda_requirements_summary(runtime_service.get_effective_runtime_root_for_gpu_checks())
        self.logChanged.emit(f'Downloading cuDNN package for {requirement_text}...\nPlease wait...')

        def on_progress(percent: float) -> None:
            value = max(0.0, min(float(percent), 100.0))
            self.emit_progress(value, self.progress_text_with_eta('Downloading', value))

        def task():
            return cuda_dependency_service.install_cudnn_and_configure_path(
                progress_callback=on_progress,
                stop_event=self._stop_event,
                runtime_path=runtime_service.get_effective_runtime_root_for_gpu_checks(),
            )

        def on_success(result) -> None:
            bin_dir, process_changed, user_changed = result
            self.emit_progress(100.0, 'Installed')
            self.logChanged.emit(f'cuDNN installed:\n{bin_dir}')
            self.finishedSuccess.emit(
                CudnnInstallPayload(
                    bin_dir=str(bin_dir),
                    process_path_changed=bool(process_changed),
                    user_path_changed=bool(user_changed),
                )
            )

        def on_stopped() -> None:
            self.emit_progress(0.0, 'Stopped')
            self.logChanged.emit('cuDNN install stopped.')
            self.finishedStopped.emit()

        def on_error(exc: Exception) -> None:
            self.emit_progress(0.0, '0.00%')
            self.logChanged.emit(f'cuDNN install failed: {exc}')
            self.errorRaised.emit(str(exc))

        self.run_task(task, on_success=on_success, on_stopped=on_stopped, on_error=on_error)
