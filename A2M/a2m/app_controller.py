from __future__ import annotations
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from PySide6.QtCore import QByteArray, QEvent, QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget
from a2m.core import conversion_service, cuda_dependency_service, gpu_runtime_service, model_service, runtime_pack_service, runtime_service
from a2m.core.config_service import AppConfig, default_config, load_config, normalize_conversion_method, save_config
from a2m.core.constants import CUDNN_DOWNLOAD_URL, CUDA_DOWNLOAD_URL, OFFICIAL_PAGE_URL, SUPPORTED_AUDIO_FILTER, UPDATE_CHECK_TIMEOUT_SECONDS
from a2m.core.constants import MODERN_FRAME_THRESHOLD_DEFAULT, MODERN_FRAME_THRESHOLD_MAX, MODERN_FRAME_THRESHOLD_MIN
from a2m.core.constants import MODERN_OFFSET_THRESHOLD_DEFAULT, MODERN_OFFSET_THRESHOLD_MAX, MODERN_OFFSET_THRESHOLD_MIN
from a2m.core.constants import MODERN_ONSET_THRESHOLD_DEFAULT, MODERN_ONSET_THRESHOLD_MAX, MODERN_ONSET_THRESHOLD_MIN
from a2m.core.constants import MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT, MODERN_PEDAL_OFFSET_THRESHOLD_MAX, MODERN_PEDAL_OFFSET_THRESHOLD_MIN
from a2m.core.gpu_helper import REASON_PROVIDER_LOAD_FAILED, REASON_PROVIDER_NOT_EXPOSED, REASON_SESSION_CREATION_FAILED
from a2m.core.gpu_runtime_manager import GpuRuntimeManager, RuntimeDecision
from a2m.core.messages import RUNTIME_INFO_CHECKING, RUNTIME_WARNING_MISSING
from a2m.core.paths import icon_path
from a2m.controller.dialog_coordinator import DialogCoordinator
from a2m.ui.interaction import is_pointer_control as is_pointer_control_widget
from a2m.ui.interaction import sync_pointer_cursor as sync_pointer_cursor_widget
from a2m.ui.main_window import MainWindow
from a2m.ui.theme import get_theme
from a2m.ui.windows_titlebar import apply_windows_titlebar_theme
from a2m.workers.cudnn_install_worker import CudnnInstallWorker
from a2m.workers.conversion_worker import ConversionWorker
from a2m.workers.model_download_worker import ModelDownloadWorker
from a2m.workers.payloads import CudnnInstallPayload, RuntimePackDownloadPayload
from a2m.workers.runtime_pack_download_worker import RuntimePackDownloadWorker
from a2m.workers.update_check_worker import UpdateCheckWorker

class AppController(QObject):
    runtimeProbeReady = Signal(int, bool, bool, bool)
    gpuValidationReady = Signal(int, object, bool, object)
    _STARTUP_RUNTIME_PROMPTS = {
        'Verifying ONNX Runtime support...\nPlease wait...',
        'Validating ONNX GPU support...\nPlease wait...',
    }

    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        try:
            cuda_dependency_service.ensure_cuda_runtime_bins_in_process_path()
        except Exception as exc:
            print(f'[A2M] Warning: CUDA dependency bootstrap failed: {exc}', file=sys.stderr)
        self.config: AppConfig = load_config()
        startup_provider_before = str(self.config.gpu_provider_preference or '').strip().lower()
        startup_runtime_before = (bool(self.config.gpu_runtime_enabled), str(self.config.gpu_runtime_path or '').strip())
        self.config.device_preference = runtime_service.set_device_preference(self.config.device_preference)
        self.config.gpu_batch_size = runtime_service.set_gpu_batch_size(self.config.gpu_batch_size)
        self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference(self.config.gpu_provider_preference)
        self.config.gpu_runtime_enabled, self.config.gpu_runtime_path = runtime_service.set_gpu_runtime(self.config.gpu_runtime_enabled, self.config.gpu_runtime_path)
        self._sync_runtime_pack_selection(prefer_existing_config=True)
        startup_runtime_after = (bool(self.config.gpu_runtime_enabled), str(self.config.gpu_runtime_path or '').strip())
        self._config_changed_during_startup = (startup_provider_before != self.config.gpu_provider_preference) or (startup_runtime_before != startup_runtime_after)
        self.config.download_location = str(conversion_service.set_output_midi_dir(self.config.download_location))
        self.selected_file: Path | None = None
        self.model_download_in_progress = False
        self.runtime_pack_download_in_progress = False
        self.cudnn_install_in_progress = False
        self.transcription_in_progress = False
        self.update_check_in_progress = False
        self.close_after_transcription = False
        self.close_after_model_download = False
        self.close_after_runtime_pack_download = False
        self.close_after_cudnn_install = False
        self._restart_after_gpu_setup = False
        self._restart_prompt_pending = False
        self._runtime_checking = True
        self._runtime_probe_token = 0
        self._runtime_available = False
        self._cuda_available = False
        self._dml_available = False
        self._model_available = False
        self._gpu_validation_token = 0
        self._gpu_validation_in_progress = False
        self._gpu_validation_cache: dict[tuple[str, str, str], RuntimeDecision] = {}
        self._gpu_validated_this_session = False
        self._gpu_requirements_notice_shown = False
        self.gpu_runtime_manager = GpuRuntimeManager()
        self._config_save_warning_shown = False
        self._conversion_thread: QThread | None = None
        self._conversion_worker: ConversionWorker | None = None
        self._model_thread: QThread | None = None
        self._model_worker: ModelDownloadWorker | None = None
        self._runtime_pack_thread: QThread | None = None
        self._runtime_pack_worker: RuntimePackDownloadWorker | None = None
        self._cudnn_thread: QThread | None = None
        self._cudnn_worker: CudnnInstallWorker | None = None
        self._update_thread: QThread | None = None
        self._update_worker: UpdateCheckWorker | None = None
        self.window = MainWindow(theme=get_theme(self.config.theme_mode), icon_path=icon_path(), theme_mode=self.config.theme_mode, ui_scale_percent=self.config.ui_scale_percent)
        self.window.set_close_handler(self._handle_close_request)
        self._restore_window_geometry()
        self._refresh_settings_values()
        self.window.set_settings_visible(False, animated=False)
        self.window.set_progress(0, '0%')
        self.window.set_console_text('')
        self._show_startup_runtime_prompt('Verifying ONNX Runtime support...\nPlease wait...')
        self.window.set_runtime_status(checking=True, runtime_available=False, cuda_available=False, dml_available=False, active_provider='CPU')
        self._dialogs = DialogCoordinator(self.window, apply_theme=self._apply_dialog_theme, exec_dialog=self._exec_dialog)
        self._connect_signals()
        self._update_model_status()
        self._apply_controls_state()
        if self._config_changed_during_startup:
            self._save_config(report_errors=False)
        self._start_runtime_probe()
        QTimer.singleShot(80, self._prompt_model_download_if_missing)
        if self.config.auto_check_updates:
            QTimer.singleShot(900, lambda: self.check_for_updates(manual=False))

    def run(self) -> None:
        self.window.show()

    def _connect_signals(self) -> None:
        self.window.chooseFileRequested.connect(self.choose_file)
        self.window.convertRequested.connect(self.start_conversion)
        self.window.stopRequested.connect(self.stop_conversion)
        self.window.openDownloadsRequested.connect(self.open_downloads_folder)
        self.window.officialPageRequested.connect(lambda: webbrowser.open(OFFICIAL_PAGE_URL))
        self.window.devicePreferenceChanged.connect(self._on_device_preference_changed)
        self.window.conversionMethodChanged.connect(self._on_conversion_method_changed)
        self.window.modernAdaptiveThresholdsChanged.connect(lambda enabled: self._on_modern_option_changed('modern_adaptive_thresholds_enabled', enabled))
        self.window.modernInputNormalizationChanged.connect(lambda enabled: self._on_modern_option_changed('modern_input_normalization_enabled', enabled))
        self.window.modernSmartOverlapStitchingChanged.connect(lambda enabled: self._on_modern_option_changed('modern_smart_overlap_stitching_enabled', enabled))
        self.window.modernAutoCalibrationChanged.connect(lambda enabled: self._on_modern_option_changed('modern_auto_calibration_enabled', enabled))
        self.window.modernCleanupScaleChanged.connect(lambda value: self._on_modern_scale_changed('modern_cleanup_scale', value, default=1.0))
        self.window.modernPedalClusterScaleChanged.connect(lambda value: self._on_modern_scale_changed('modern_pedal_cluster_scale', value, default=1.0))
        self.window.modernAlignmentGateScaleChanged.connect(lambda value: self._on_modern_scale_changed('modern_alignment_gate_scale', value, default=1.0))
        self.window.modernOnsetThresholdChanged.connect(lambda value: self._on_modern_threshold_changed('modern_manual_onset_threshold', value, lower=MODERN_ONSET_THRESHOLD_MIN, upper=MODERN_ONSET_THRESHOLD_MAX, default=MODERN_ONSET_THRESHOLD_DEFAULT))
        self.window.modernOffsetThresholdChanged.connect(lambda value: self._on_modern_threshold_changed('modern_manual_offset_threshold', value, lower=MODERN_OFFSET_THRESHOLD_MIN, upper=MODERN_OFFSET_THRESHOLD_MAX, default=MODERN_OFFSET_THRESHOLD_DEFAULT))
        self.window.modernFrameThresholdChanged.connect(lambda value: self._on_modern_threshold_changed('modern_manual_frame_threshold', value, lower=MODERN_FRAME_THRESHOLD_MIN, upper=MODERN_FRAME_THRESHOLD_MAX, default=MODERN_FRAME_THRESHOLD_DEFAULT))
        self.window.modernPedalOffsetThresholdChanged.connect(lambda value: self._on_modern_threshold_changed('modern_manual_pedal_offset_threshold', value, lower=MODERN_PEDAL_OFFSET_THRESHOLD_MIN, upper=MODERN_PEDAL_OFFSET_THRESHOLD_MAX, default=MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT))
        self.window.gpuProviderPreferenceChanged.connect(self._on_gpu_provider_preference_changed)
        self.window.gpuBatchSizeChanged.connect(self._on_gpu_batch_size_changed)
        self.window.uiScalePercentChanged.connect(self._on_ui_scale_percent_changed)
        self.window.downloadLocationChanged.connect(self._on_download_location_changed)
        self.window.autoCheckUpdatesChanged.connect(self._on_auto_check_updates_changed)
        self.window.checkUpdatesNowRequested.connect(lambda: self.check_for_updates(manual=True))
        self.window.resetSettingsRequested.connect(self._on_reset_settings_requested)
        self.window.themeModeChanged.connect(self._on_theme_mode_changed)
        self.runtimeProbeReady.connect(self._on_runtime_probe_ready)
        self.gpuValidationReady.connect(self._on_gpu_validation_ready)

    @staticmethod
    def _runtime_has_gpu_provider(cuda_available: bool, dml_available: bool) -> bool:
        return bool(cuda_available) or bool(dml_available)

    @staticmethod
    def _normalize_runtime_path(path: str | Path | None) -> str:
        return str(path or '').strip()

    @staticmethod
    def _runtime_candidates() -> list[str]:
        candidates: list[str] = []
        for raw_candidate in (
            runtime_service.get_effective_runtime_root_for_gpu_checks(),
            runtime_service.get_packaged_runtime_root(),
        ):
            candidate = str(raw_candidate or '').strip()
            if not candidate:
                continue
            if any((existing.lower() == candidate.lower() for existing in candidates)):
                continue
            candidates.append(candidate)
        return candidates

    def _sync_runtime_pack_selection(self, *, prefer_existing_config: bool) -> bool:
        configured_enabled = bool(self.config.gpu_runtime_enabled)
        configured_path = str(self.config.gpu_runtime_path or '').strip()
        runtime_enabled = False
        runtime_path = ''
        config_runtime_is_valid = configured_enabled and configured_path and runtime_service.is_runtime_path_valid(configured_path)
        provider_pref = str(self.config.gpu_provider_preference or '').strip().lower()
        if config_runtime_is_valid and provider_pref in {'cuda', 'dml'}:
            configured_provider = gpu_runtime_service.detect_runtime_provider(configured_path)
            if configured_provider and (configured_provider != provider_pref):
                config_runtime_is_valid = False
        if config_runtime_is_valid and prefer_existing_config:
            runtime_enabled, runtime_path = runtime_service.set_gpu_runtime(True, configured_path)
        elif config_runtime_is_valid:
            _provider, installed_path = runtime_pack_service.resolve_installed_pack(self.config.gpu_provider_preference)
            if installed_path:
                runtime_enabled, runtime_path = runtime_service.set_gpu_runtime(True, installed_path)
            else:
                runtime_enabled, runtime_path = runtime_service.set_gpu_runtime(True, configured_path)
        else:
            _provider, installed_path = runtime_pack_service.resolve_installed_pack(self.config.gpu_provider_preference)
            if installed_path:
                runtime_enabled, runtime_path = runtime_service.set_gpu_runtime(True, installed_path)
            else:
                runtime_enabled, runtime_path = runtime_service.set_gpu_runtime(False, '')
        changed = (bool(self.config.gpu_runtime_enabled) != bool(runtime_enabled)) or (str(self.config.gpu_runtime_path or '').strip() != str(runtime_path or '').strip())
        self.config.gpu_runtime_enabled = bool(runtime_enabled)
        self.config.gpu_runtime_path = str(runtime_path or '')
        return changed

    def _resolved_gpu_provider(self) -> str:
        return runtime_pack_service.resolve_provider_for_preference(self.config.gpu_provider_preference)

    def _reset_cuda_session_validation(self) -> None:
        self._gpu_validation_cache.clear()
        self._gpu_validation_in_progress = False
        self._gpu_validated_this_session = False

    @staticmethod
    def _runtime_source_matches_provider_preference(provider_preference: str, *, cuda_available: bool, dml_available: bool) -> bool:
        pref = str(provider_preference or 'auto').strip().lower()
        if pref == 'cuda':
            return bool(cuda_available)
        if pref == 'dml':
            return bool(dml_available)
        return bool(cuda_available or dml_available)

    def _runtime_path_matches_provider_preference(self, runtime_path: str | Path | None) -> bool:
        candidate = self._normalize_runtime_path(runtime_path)
        if not candidate:
            return False
        active_runtime = self._normalize_runtime_path(runtime_service.get_effective_runtime_root_for_gpu_checks())
        if active_runtime and candidate.lower() == active_runtime.lower() and (not self._runtime_checking):
            if not self._runtime_available:
                return False
            return self._runtime_source_matches_provider_preference(
                self.config.gpu_provider_preference,
                cuda_available=self._cuda_available,
                dml_available=self._dml_available,
            )
        probe = self.gpu_runtime_manager.probe_runtime_path(candidate)
        if not bool(probe.get('runtime_available', False)):
            return False
        return self._runtime_source_matches_provider_preference(
            self.config.gpu_provider_preference,
            cuda_available=bool(probe.get('cuda_available', False)),
            dml_available=bool(probe.get('dml_available', False)),
        )

    def _try_activate_non_pack_runtime_source(self) -> bool:
        for candidate in self._runtime_candidates():
            if not self._runtime_path_matches_provider_preference(candidate):
                continue
            enabled, selected_path = runtime_service.set_gpu_runtime(True, candidate)
            if not enabled or (not selected_path):
                continue
            self.config.gpu_runtime_enabled = bool(enabled)
            self.config.gpu_runtime_path = str(selected_path or '')
            self._apply_state_transaction(refresh_first=False, report_save_errors=False)
            return True
        return False

    def _ensure_runtime_pack_selected(self, *, prompt_if_missing: bool) -> bool:
        if (
            self.config.gpu_runtime_enabled
            and self.config.gpu_runtime_path
            and self._runtime_available
            and self._runtime_source_matches_provider_preference(
                self.config.gpu_provider_preference,
                cuda_available=self._cuda_available,
                dml_available=self._dml_available,
            )
        ):
            return True
        changed = self._sync_runtime_pack_selection(prefer_existing_config=False)
        if changed:
            self._apply_state_transaction(refresh_first=False, report_save_errors=False)
        if self.config.gpu_runtime_enabled and self.config.gpu_runtime_path and self._runtime_path_matches_provider_preference(self.config.gpu_runtime_path):
            return True
        if self._try_activate_non_pack_runtime_source():
            return True
        if prompt_if_missing:
            self._prompt_runtime_pack_install(self._resolved_gpu_provider())
        return False

    def _refresh_settings_values(self) -> None:
        self.window.set_settings_values(
            device_preference=self.config.device_preference,
            conversion_method=normalize_conversion_method(self.config.conversion_method),
            modern_adaptive_thresholds_enabled=bool(self.config.modern_adaptive_thresholds_enabled),
            modern_input_normalization_enabled=bool(self.config.modern_input_normalization_enabled),
            modern_smart_overlap_stitching_enabled=bool(self.config.modern_smart_overlap_stitching_enabled),
            modern_auto_calibration_enabled=bool(self.config.modern_auto_calibration_enabled),
            modern_cleanup_scale=float(self.config.modern_cleanup_scale),
            modern_pedal_cluster_scale=float(self.config.modern_pedal_cluster_scale),
            modern_alignment_gate_scale=float(self.config.modern_alignment_gate_scale),
            modern_manual_onset_threshold=float(self.config.modern_manual_onset_threshold),
            modern_manual_offset_threshold=float(self.config.modern_manual_offset_threshold),
            modern_manual_frame_threshold=float(self.config.modern_manual_frame_threshold),
            modern_manual_pedal_offset_threshold=float(self.config.modern_manual_pedal_offset_threshold),
            gpu_provider_preference=self.config.gpu_provider_preference,
            gpu_batch_size=self.config.gpu_batch_size,
            ui_scale_percent=self.config.ui_scale_percent,
            download_location=self.config.download_location,
            auto_check_updates=self.config.auto_check_updates,
        )

    def _sync_settings_and_persist(self, *, refresh_first: bool, report_save_errors: bool=True) -> bool:
        if refresh_first:
            self._refresh_settings_values()
            return self._save_config(report_errors=report_save_errors)
        saved = self._save_config(report_errors=report_save_errors)
        self._refresh_settings_values()
        return saved

    def _apply_state_transaction(self, *, refresh_first: bool=False, persist: bool=True, report_save_errors: bool=True, start_runtime_probe: bool=False, apply_controls: bool=False) -> bool:
        saved = True
        if persist:
            saved = self._sync_settings_and_persist(refresh_first=refresh_first, report_save_errors=report_save_errors)
        elif refresh_first:
            self._refresh_settings_values()
        if start_runtime_probe:
            self._start_runtime_probe()
        if apply_controls:
            self._apply_controls_state()
        return saved

    def _gpu_validation_key(self, model_path: Path | str) -> tuple[str, str, str]:
        provider = self._resolved_gpu_provider()
        runtime_root = runtime_service.get_effective_runtime_root_for_gpu_checks()
        runtime_path = str(runtime_root or '').strip()
        try:
            model_key = str(Path(model_path).resolve())
        except Exception:
            model_key = str(model_path)
        return (provider, runtime_path, model_key)

    def _cached_gpu_validation_decision(self, model_path: Path | str) -> RuntimeDecision | None:
        key = self._gpu_validation_key(model_path)
        return self._gpu_validation_cache.get(key)

    def _cache_gpu_validation_decision(self, model_path: Path | str, decision: RuntimeDecision) -> None:
        key = self._gpu_validation_key(model_path)
        self._gpu_validation_cache[key] = decision

    def _should_show_gpu_setup_notice(self) -> bool:
        if self._gpu_requirements_notice_shown:
            return False
        if self._runtime_checking:
            return False
        provider = self._resolved_gpu_provider()
        provider_ready = self._runtime_has_gpu_provider(self._cuda_available, self._dml_available)
        missing_cuda_dlls: list[str] = []
        if provider == 'cuda':
            try:
                missing_cuda_dlls = cuda_dependency_service.missing_required_cuda_dlls()
            except Exception:
                missing_cuda_dlls = []
        return (not provider_ready) or bool(missing_cuda_dlls)

    def _show_gpu_setup_notice_if_needed(self) -> None:
        if not self._should_show_gpu_setup_notice():
            return
        provider = self._resolved_gpu_provider()
        if provider == 'cuda':
            message = (
                'GPU acceleration may take time to set up.\n\n'
                'Depending on your system, A2M may need up to three dependencies:\n'
                '1) ONNX GPU runtime pack, 2) CUDA Toolkit, 3) cuDNN.\n\n'
                'A2M can install two of these automatically, but the total download size can reach roughly 4 GB.\n'
                'Even after setup, GPU may not be faster on lower-end hardware.\n\n'
                'CPU mode remains available and stable.'
            )
        else:
            message = (
                'GPU acceleration may take time to set up.\n\n'
                'Depending on your system, A2M may need an ONNX GPU runtime pack and Windows DirectML support.\n'
                'The required downloads can still be large, and GPU may not be faster on lower-end hardware.\n\n'
                'CPU mode remains available and stable.'
            )
        self._gpu_requirements_notice_shown = True
        self._show_info('Before enabling GPU', message)

    def _start_gpu_runtime_validation(self, *, show_message: bool, startup: bool=False) -> bool:
        if self._gpu_validation_in_progress:
            return False
        if self._runtime_checking or (not self._runtime_available):
            return False
        if self.config.device_preference != 'gpu':
            return False
        if self._resolved_gpu_provider() not in {'cuda', 'dml'}:
            return False
        model_path = model_service.get_existing_model_path()
        if model_path is None:
            return False
        cached = self._cached_gpu_validation_decision(model_path)
        if cached is not None:
            self._apply_runtime_decision(cached, show_message=show_message and (cached.device_mode != 'gpu'))
            self._refresh_runtime_status()
            self._apply_controls_state()
            return True
        self._gpu_validation_token += 1
        token = int(self._gpu_validation_token)
        self._gpu_validation_in_progress = True
        self.window.set_progress(0, 'Validating ONNX GPU support...')
        if startup:
            self._show_startup_runtime_prompt('Validating ONNX GPU support...\nPlease wait...')
        self._apply_controls_state()

        def worker() -> None:
            try:
                decision = self.gpu_runtime_manager.prepare_gpu_runtime(self.config.gpu_provider_preference, model_path)
            except Exception as exc:
                decision = RuntimeDecision(
                    device_mode='cpu',
                    active_provider='cpu',
                    gpu_model_name='',
                    reason_code='UNKNOWN_GPU_ERROR',
                    reason_text=str(exc),
                )
            try:
                self.gpuValidationReady.emit(token, decision, bool(show_message), str(model_path))
            except RuntimeError:
                return

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _on_gpu_validation_ready(self, token: int, decision_obj: object, show_message: bool, model_path: object) -> None:
        if int(token) != int(self._gpu_validation_token):
            return
        self._gpu_validation_in_progress = False
        if not isinstance(decision_obj, RuntimeDecision):
            decision = RuntimeDecision(device_mode='cpu', active_provider='cpu', gpu_model_name='', reason_code='UNKNOWN_GPU_ERROR', reason_text='GPU validation returned an invalid result.')
        else:
            decision = decision_obj
        self._cache_gpu_validation_decision(str(model_path or ''), decision)
        self._apply_runtime_decision(decision, show_message=bool(show_message) and (decision.device_mode != 'gpu'))
        self._apply_state_transaction(
            refresh_first=False,
            report_save_errors=False,
            start_runtime_probe=(decision.device_mode != 'gpu'),
            apply_controls=True,
        )
        if decision.device_mode == 'gpu' and self._restart_prompt_pending:
            self._restart_prompt_pending = False
            self._restart_after_gpu_setup = True
            self._show_info(
                'GPU setup complete',
                'GPU dependencies are now installed and validated.\n\n'
                'A2M will now restart to apply GPU changes.',
            )
            self._try_restart_after_gpu_setup()
        self._clear_startup_runtime_prompt_if_idle()

    def _try_restart_after_gpu_setup(self) -> None:
        if not self._restart_after_gpu_setup:
            return
        if (
            self.model_download_in_progress
            or self.runtime_pack_download_in_progress
            or self.cudnn_install_in_progress
            or self.transcription_in_progress
            or self._gpu_validation_in_progress
        ):
            return
        self._restart_after_gpu_setup = False
        self._restart_application()

    def _restart_application(self) -> None:
        if getattr(sys, 'frozen', False):
            launch_args = [str(sys.executable), *sys.argv[1:]]
            working_dir = str(Path(sys.executable).resolve().parent)
        else:
            script_path = str(Path(sys.argv[0]).resolve())
            launch_args = [str(sys.executable), script_path, *sys.argv[1:]]
            working_dir = str(Path(script_path).resolve().parent)
        self._persist_config()
        try:
            subprocess.Popen(launch_args, cwd=working_dir)
        except Exception as exc:
            self._show_error('Restart failed', f'GPU setup finished, but A2M could not restart automatically.\n\n{exc}')
            return
        self.window.close()

    def _prompt_runtime_pack_install(self, provider: str) -> bool:
        if self.runtime_pack_download_in_progress:
            self._show_info('Runtime pack download', 'A runtime pack download is already in progress.')
            return False
        normalized_provider = runtime_pack_service.resolve_provider_for_preference(provider)
        provider_label = runtime_pack_service.provider_display_name(normalized_provider)
        pack_url = runtime_pack_service.provider_pack_url(normalized_provider)
        if not pack_url:
            self._show_warning('Runtime pack unavailable', f'{provider_label} runtime pack URL is not configured.')
            return False
        prompt = f'GPU acceleration requires the {provider_label} runtime pack.\n\nWould you like to download and install it now?'
        if self._ask_question('GPU runtime pack required', prompt, default_button=QMessageBox.Yes) != QMessageBox.Yes:
            return False
        self._start_runtime_pack_download(normalized_provider)
        return True

    def _show_runtime_reinstall_required(self) -> None:
        self._show_warning('ONNX Runtime missing', RUNTIME_WARNING_MISSING)

    def _start_runtime_pack_download(self, provider: str) -> None:
        provider_label = runtime_pack_service.provider_display_name(provider)
        self.runtime_pack_download_in_progress = True
        self._apply_controls_state()
        self.window.set_progress(0, 'Downloading 0.00%')
        self.window.set_console_text(f'Downloading {provider_label} runtime pack...\nPlease wait...')
        worker = RuntimePackDownloadWorker(provider)
        thread = self._create_worker_thread(worker)
        worker.progressChanged.connect(self.window.set_progress)
        worker.logChanged.connect(self.window.set_console_text)
        worker.errorRaised.connect(self._on_runtime_pack_download_error)
        worker.finishedSuccess.connect(self._on_runtime_pack_download_success)
        worker.finishedStopped.connect(self._on_runtime_pack_download_stopped)
        thread.finished.connect(self._on_runtime_pack_download_finished)
        self._runtime_pack_thread = thread
        self._runtime_pack_worker = worker
        thread.start()

    def _create_worker_thread(self, worker: object) -> QThread:
        thread = QThread(self)
        move_to_thread = getattr(worker, 'moveToThread', None)
        if callable(move_to_thread):
            move_to_thread(thread)
        run_slot = getattr(worker, 'run', None)
        if callable(run_slot):
            thread.started.connect(run_slot)
        finished_signal = getattr(worker, 'finished', None)
        delete_later = getattr(worker, 'deleteLater', None)
        if finished_signal is not None:
            finished_signal.connect(thread.quit)
            if callable(delete_later):
                finished_signal.connect(delete_later)
        thread.finished.connect(thread.deleteLater)
        return thread

    def _restore_window_geometry(self) -> None:
        encoded = (self.config.window_geometry or '').strip()
        if not encoded:
            return
        try:
            payload = QByteArray.fromBase64(encoded.encode('ascii'))
            if payload:
                self.window.restoreGeometry(payload)
        except Exception:
            return

    def _persist_config(self) -> None:
        try:
            geometry = self.window.saveGeometry().toBase64().data().decode('ascii')
        except Exception:
            geometry = ''
        self.config.window_geometry = geometry
        self._save_config(report_errors=False)

    def _save_config(self, *, report_errors: bool=True) -> bool:
        target_path = save_config(self.config)
        if target_path:
            self._config_save_warning_shown = False
            return True
        if report_errors and (not self._config_save_warning_shown):
            self._config_save_warning_shown = True
            self._show_warning(
                'Settings not saved',
                'A2M could not save your settings.\n'
                'It stores config in LOCALAPPDATA.\n'
                'Check write permissions and try again.',
            )
        return False

    def _apply_dialog_theme(self, widget: QWidget) -> None:
        theme = self.window.theme
        style = f'QDialog, QMessageBox {{ background: {theme.panel_bg}; color: {theme.text_primary}; }}QLabel {{ color: {theme.text_primary}; background: transparent; }}QPushButton {{ background: {theme.panel_bg}; color: {theme.text_primary}; border: 1px solid {theme.border}; border-radius: 6px; padding: 5px 10px; font: 600 9.5pt "Segoe UI"; min-height: 24px; }}QPushButton:hover {{ background: {theme.accent}; color: {theme.text_primary}; }}QPushButton:disabled {{ background: {theme.disabled_bg}; color: {theme.disabled_fg}; border-color: {theme.border}; }}QLineEdit, QListView, QTreeView {{ background: {theme.app_bg}; color: {theme.text_primary}; border: 1px solid {theme.border}; }}'
        widget.setStyleSheet(style)
        palette = widget.palette()
        palette.setColor(QPalette.Window, QColor(theme.panel_bg))
        palette.setColor(QPalette.WindowText, QColor(theme.text_primary))
        palette.setColor(QPalette.Base, QColor(theme.app_bg))
        palette.setColor(QPalette.AlternateBase, QColor(theme.panel_bg))
        palette.setColor(QPalette.Text, QColor(theme.text_primary))
        palette.setColor(QPalette.Button, QColor(theme.panel_bg))
        palette.setColor(QPalette.ButtonText, QColor(theme.text_primary))
        palette.setColor(QPalette.ToolTipBase, QColor(theme.panel_bg))
        palette.setColor(QPalette.ToolTipText, QColor(theme.text_primary))
        widget.setPalette(palette)
        widget.setAutoFillBackground(True)
        self._apply_windows_titlebar_theme(widget)
        self._apply_dialog_cursors(widget)

    @staticmethod
    def _is_pointer_control(widget: object) -> bool:
        return is_pointer_control_widget(widget)

    def _sync_pointer_cursor(self, widget: QWidget) -> None:
        sync_pointer_cursor_widget(widget)

    def _apply_dialog_cursors(self, widget: QWidget) -> None:
        widget.setCursor(Qt.ArrowCursor)
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)
            if self._is_pointer_control(child):
                self._sync_pointer_cursor(child)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if event.type() == QEvent.ChildAdded:
            child = event.child() if hasattr(event, 'child') else None
            if isinstance(child, QWidget):
                child.installEventFilter(self)
                if self._is_pointer_control(child):
                    self._sync_pointer_cursor(child)
                for descendant in child.findChildren(QWidget):
                    descendant.installEventFilter(self)
                    if self._is_pointer_control(descendant):
                        self._sync_pointer_cursor(descendant)
        if self._is_pointer_control(watched):
            if event.type() in {QEvent.EnabledChange, QEvent.Enter, QEvent.HoverEnter, QEvent.HoverMove}:
                self._sync_pointer_cursor(watched)
        return super().eventFilter(watched, event)

    def _apply_windows_titlebar_theme(self, widget: QWidget) -> None:
        apply_windows_titlebar_theme(widget, dark=self.window.current_theme_mode() != 'light')

    def _show_info(self, title: str, text: str) -> QMessageBox.StandardButton:
        return self._dialogs.show_info(title, text)

    def _show_warning(self, title: str, text: str) -> QMessageBox.StandardButton:
        return self._dialogs.show_warning(title, text)

    def _show_error(self, title: str, text: str) -> QMessageBox.StandardButton:
        return self._dialogs.show_error(title, text)

    def _ask_question(self, title: str, text: str, *, default_button: QMessageBox.StandardButton=QMessageBox.NoButton) -> QMessageBox.StandardButton:
        return self._dialogs.ask_question(title, text, default_button=default_button)

    def _restore_app_cursor_state(self) -> None:
        self.window.restore_cursor_state_after_modal()

    def _exec_dialog(self, dialog: QWidget) -> int:
        try:
            return int(dialog.exec())
        finally:
            self._restore_app_cursor_state()

    def _pick_audio_file(self) -> str:
        selected, _ = QFileDialog.getOpenFileName(self.window, 'Choose audio file', '', SUPPORTED_AUDIO_FILTER)
        return str(selected or '').strip()

    def _start_runtime_probe(self) -> None:
        self._runtime_probe_token += 1
        token = int(self._runtime_probe_token)
        self._runtime_checking = True
        self.window.set_runtime_status(checking=True, runtime_available=False, cuda_available=False, dml_available=False, active_provider='CPU')
        self.window.set_progress(0, 'Verifying ONNX Runtime...')
        self._show_startup_runtime_prompt('Verifying ONNX Runtime support...\nPlease wait...')
        self._apply_controls_state()

        def worker() -> None:
            runtime_ok = False
            cuda_ok = False
            dml_ok = False
            probe_path = runtime_service.get_effective_runtime_root_for_gpu_checks()
            if probe_path:
                probe = self.gpu_runtime_manager.probe_runtime_path(probe_path)
                runtime_ok = bool(probe.get('runtime_available', False))
                cuda_ok = bool(probe.get('cuda_available', False))
                dml_ok = bool(probe.get('dml_available', False))
            try:
                self.runtimeProbeReady.emit(token, runtime_ok, cuda_ok, dml_ok)
            except RuntimeError:
                return
        threading.Thread(target=worker, daemon=True).start()

    def _on_runtime_probe_ready(self, token: int, runtime_available: bool, cuda_available: bool, dml_available: bool) -> None:
        if int(token) != int(self._runtime_probe_token):
            return
        self._runtime_checking = False
        self._runtime_available = bool(runtime_available)
        self._cuda_available = bool(cuda_available)
        self._dml_available = bool(dml_available)
        if (not self._runtime_has_gpu_provider(self._cuda_available, self._dml_available)) and self.config.gpu_provider_preference != 'auto':
            self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference('auto')
            self._reset_cuda_session_validation()
            self._apply_state_transaction(refresh_first=False, report_save_errors=False)
        self._refresh_runtime_status()
        self._apply_controls_state()
        self._validate_gpu_session_on_startup_if_needed()
        self._clear_startup_runtime_prompt_if_idle()

    def _refresh_runtime_status(self) -> None:
        active_provider = self._status_provider_label()
        self.window.set_runtime_status(checking=self._runtime_checking, runtime_available=self._runtime_available, cuda_available=self._cuda_available, dml_available=self._dml_available, active_provider=active_provider)

    def _status_provider_label(self) -> str:
        if self.config.device_preference == 'gpu':
            if self._gpu_validation_in_progress:
                return 'Validation pending'
            if self.config.gpu_last_reason_text:
                return f'CPU fallback: {self.config.gpu_last_reason_text}'
            validated_provider = str(self.config.gpu_last_validated_provider or '').strip().lower()
            provider_pref = str(self.config.gpu_provider_preference or 'auto').strip().lower()
            if self._gpu_validated_this_session and validated_provider in {'cuda', 'dml'}:
                if provider_pref == 'auto' or provider_pref == validated_provider:
                    return 'GPU'
            if self._runtime_has_gpu_provider(self._cuda_available, self._dml_available):
                return 'Validation pending'
        return 'CPU'

    def _record_runtime_decision(self, decision: RuntimeDecision) -> None:
        self.config.gpu_last_reason_code = str(decision.reason_code or '').strip()
        self.config.gpu_last_reason_text = str(decision.reason_text or '').strip()
        self._gpu_validated_this_session = True
        provider = str(decision.active_provider or '').strip().lower()
        if provider in {'cuda', 'dml', 'cpu'}:
            self.config.gpu_last_validated_provider = provider
        elif decision.device_mode == 'gpu':
            self.config.gpu_last_validated_provider = 'cpu'
        else:
            self.config.gpu_last_validated_provider = ''

    def _clear_gpu_runtime_validation_state(self) -> None:
        self.config.gpu_last_reason_code = ''
        self.config.gpu_last_reason_text = ''
        self.config.gpu_last_validated_provider = ''
        self._gpu_validated_this_session = False

    def _apply_runtime_decision(self, decision: RuntimeDecision, *, show_message: bool=False) -> None:
        requested_provider = self._resolved_gpu_provider()
        self._record_runtime_decision(decision)
        conversion_service.reset_transcriptor()
        if decision.device_mode != 'gpu':
            should_prompt_cuda = show_message and requested_provider == 'cuda' and self._should_prompt_cuda_toolkit_install(decision)
            self.config.device_preference = runtime_service.set_device_preference('cpu')
            self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference('auto')
            if show_message:
                fallback_text = f'GPU unavailable. Using CPU fallback: {decision.reason_text or "GPU initialization failed."}'
                if should_prompt_cuda:
                    self._prompt_cuda_dependency_setup(fallback_text)
                else:
                    self._show_info('GPU unavailable', fallback_text)
        else:
            self.config.device_preference = runtime_service.set_device_preference('gpu')
        self._refresh_runtime_status()

    def _prompt_cuda_dependency_setup(self, fallback_text: str) -> None:
        prompt_cuda = fallback_text + '\n\nCUDA mode requires CUDA 12.9.1 (Express install).\nWould you like to open the NVIDIA CUDA 12.9.1 download page now?'
        if self._ask_question('CUDA required', prompt_cuda, default_button=QMessageBox.Yes) == QMessageBox.Yes:
            webbrowser.open(CUDA_DOWNLOAD_URL)
        cudnn_url = str(CUDNN_DOWNLOAD_URL or '').strip()
        if not cudnn_url:
            return
        prompt_cudnn = 'CUDA requires cuDNN to function correctly.\n\nA2M can download cuDNN 9.19 for CUDA 12 and add its bin folder to PATH automatically.\n\nWould you like to do that now?'
        if self._ask_question('Install cuDNN', prompt_cudnn, default_button=QMessageBox.Yes) != QMessageBox.Yes:
            return
        self._start_cudnn_install()

    def _start_cudnn_install(self) -> None:
        if self.cudnn_install_in_progress:
            self._show_info('cuDNN install', 'cuDNN installation is already in progress.')
            return
        self.cudnn_install_in_progress = True
        self._apply_controls_state()
        self.window.set_progress(0, 'Downloading 0.00%')
        self.window.set_console_text('Downloading cuDNN 9.19 package...\nPlease wait...')
        worker = CudnnInstallWorker()
        thread = self._create_worker_thread(worker)
        worker.progressChanged.connect(self.window.set_progress)
        worker.logChanged.connect(self.window.set_console_text)
        worker.errorRaised.connect(self._on_cudnn_install_error)
        worker.finishedSuccess.connect(self._on_cudnn_install_success)
        worker.finishedStopped.connect(self._on_cudnn_install_stopped)
        thread.finished.connect(self._on_cudnn_install_finished)
        self._cudnn_thread = thread
        self._cudnn_worker = worker
        thread.start()

    def _on_cudnn_install_success(self, payload: object) -> None:
        if isinstance(payload, CudnnInstallPayload):
            bin_dir = str(payload.bin_dir or '').strip()
            user_changed = bool(payload.user_path_changed)
        elif isinstance(payload, dict):
            bin_dir = str(payload.get('bin_dir', '') or '').strip()
            user_changed = bool(payload.get('user_path_changed', False))
        else:
            bin_dir = ''
            user_changed = False
        restart_note = '\n\nPlease restart A2M so all checks use the updated PATH.' if user_changed else ''
        if bin_dir:
            self._show_info('cuDNN installed', f'cuDNN installed successfully.\n\nBin path:\n{bin_dir}{restart_note}')
        else:
            self._show_info('cuDNN installed', f'cuDNN installed successfully.{restart_note}')
        self._reset_cuda_session_validation()
        self._restart_prompt_pending = True
        self._start_runtime_probe()

    def _on_cudnn_install_error(self, message: str) -> None:
        self._restart_prompt_pending = False
        self._restart_after_gpu_setup = False
        self._show_error('cuDNN install failed', str(message or 'Unknown error.'))

    def _on_cudnn_install_stopped(self) -> None:
        self._restart_prompt_pending = False
        self._restart_after_gpu_setup = False
        self.window.set_progress(0, 'Stopped')

    def _on_cudnn_install_finished(self) -> None:
        self.cudnn_install_in_progress = False
        self._cudnn_thread = None
        self._cudnn_worker = None
        self._apply_controls_state()
        self._try_restart_after_gpu_setup()
        if self.close_after_cudnn_install:
            self.close_after_cudnn_install = False
            self.window.close()

    def _validate_gpu_session_on_startup_if_needed(self) -> None:
        if self.config.device_preference != 'gpu':
            return
        self._start_gpu_runtime_validation(show_message=False, startup=True)

    def _show_startup_runtime_prompt(self, text: str) -> None:
        if (
            self.model_download_in_progress
            or self.runtime_pack_download_in_progress
            or self.cudnn_install_in_progress
            or self.transcription_in_progress
        ):
            return
        self.window.set_console_text(str(text or '').strip())

    def _clear_startup_runtime_prompt_if_idle(self) -> None:
        if self._runtime_checking or self._gpu_validation_in_progress:
            return
        if (
            self.model_download_in_progress
            or self.runtime_pack_download_in_progress
            or self.cudnn_install_in_progress
            or self.transcription_in_progress
        ):
            return
        current_text = ''
        try:
            current_text = str(self.window.console.toPlainText() or '').strip()
        except Exception:
            current_text = ''
        if current_text in self._STARTUP_RUNTIME_PROMPTS:
            self.window.set_console_text('')
            self.window.set_progress(0, '0%')

    def _can_convert(self) -> bool:
        if self.selected_file is None:
            return False
        if not self._model_available:
            return False
        if self._runtime_checking:
            return False
        if not self._runtime_available:
            return False
        return True

    def _apply_controls_state(self) -> None:
        busy = (
            self.model_download_in_progress
            or self.runtime_pack_download_in_progress
            or self.cudnn_install_in_progress
            or self.transcription_in_progress
            or self._gpu_validation_in_progress
        )
        self.window.set_controls_state(file_enabled=not busy, convert_enabled=not busy and self._can_convert(), stop_enabled=self.transcription_in_progress)

    def _update_model_status(self) -> None:
        model_path = model_service.get_existing_model_path()
        self._model_available = bool(model_path)
        if model_path:
            self.window.set_model_status(ready=True)
            return
        self.window.set_model_status(ready=False)

    def _prompt_model_download_if_missing(self) -> None:
        if model_service.get_existing_model_path() is not None:
            return
        install_hint = model_service.get_model_install_hint()
        prompt = f'A2M needs the ONNX transcription model before it can convert audio.\n\nIt will be downloaded once to:\n{install_hint}\n\nWould you like to download it now?'
        should_download = self._ask_question('Transcription model required', prompt, default_button=QMessageBox.Yes)
        if should_download == QMessageBox.Yes:
            self.start_model_download()
            return
        self.window.close()

    def choose_file(self) -> None:
        file_path = self._pick_audio_file()
        if not file_path:
            return
        self.selected_file = Path(file_path)
        self.window.set_file_info(self.selected_file.name, str(self.selected_file))
        self._apply_controls_state()

    def start_model_download(self) -> None:
        if self.model_download_in_progress:
            return
        self.model_download_in_progress = True
        self._apply_controls_state()
        self.window.set_progress(0, 'Downloading 0%')
        self.window.set_console_text('Downloading ONNX model...\nPlease wait...')
        worker = ModelDownloadWorker()
        thread = self._create_worker_thread(worker)
        worker.progressChanged.connect(self.window.set_progress)
        worker.logChanged.connect(self.window.set_console_text)
        worker.errorRaised.connect(self._on_model_download_error)
        worker.finishedSuccess.connect(self._on_model_download_success)
        worker.finishedStopped.connect(self._on_model_download_stopped)
        thread.finished.connect(self._on_model_download_finished)
        self._model_thread = thread
        self._model_worker = worker
        thread.start()

    def _on_model_download_success(self, _payload: object) -> None:
        self._update_model_status()
        self.window.set_progress(100, 'Downloaded')
        self._validate_gpu_session_on_startup_if_needed()
        self._clear_startup_runtime_prompt_if_idle()

    def _on_model_download_error(self, message: str) -> None:
        self._show_error('Error', message)

    def _on_model_download_stopped(self) -> None:
        self.window.set_progress(0, 'Stopped')

    def _on_model_download_finished(self) -> None:
        self.model_download_in_progress = False
        self._model_thread = None
        self._model_worker = None
        self._apply_controls_state()
        if self.close_after_model_download:
            self.close_after_model_download = False
            self.window.close()

    def _on_runtime_pack_download_success(self, payload: object) -> None:
        if isinstance(payload, RuntimePackDownloadPayload):
            provider = str(payload.provider or '').strip().lower()
            runtime_path = str(payload.runtime_path or '').strip()
        elif isinstance(payload, dict):
            provider = str(payload.get('provider', '') or '').strip().lower()
            runtime_path = str(payload.get('runtime_path', '') or '').strip()
        else:
            self._show_error('Runtime pack download failed', 'Runtime pack download completed with invalid payload.')
            return
        provider_label = runtime_pack_service.provider_display_name(provider)
        if provider not in {'cuda', 'dml'} or (not runtime_path):
            self._show_error('Runtime pack download failed', 'Runtime pack payload is missing provider/path information.')
            return
        try:
            runtime_pack_service.validate_pack_root(Path(runtime_path), provider)
        except Exception as exc:
            self._show_error('Runtime pack download failed', str(exc))
            return
        self.config.gpu_runtime_enabled, self.config.gpu_runtime_path = runtime_service.set_gpu_runtime(True, runtime_path)
        self.window.set_progress(100, 'Downloaded')
        self.window.set_console_text(f'{provider_label} runtime pack installed:\n{runtime_path}')
        self._reset_cuda_session_validation()
        self._restart_prompt_pending = True
        if self.config.device_preference == 'gpu':
            self._start_gpu_runtime_validation(show_message=True)
        self._apply_state_transaction(refresh_first=False, report_save_errors=False, start_runtime_probe=True, apply_controls=True)

    def _on_runtime_pack_download_error(self, message: str) -> None:
        self._restart_prompt_pending = False
        self._restart_after_gpu_setup = False
        self._show_error('Runtime pack download failed', message)

    def _on_runtime_pack_download_stopped(self) -> None:
        self._restart_prompt_pending = False
        self._restart_after_gpu_setup = False
        self.window.set_progress(0, 'Stopped')

    def _on_runtime_pack_download_finished(self) -> None:
        self.runtime_pack_download_in_progress = False
        self._runtime_pack_thread = None
        self._runtime_pack_worker = None
        self._apply_controls_state()
        self._try_restart_after_gpu_setup()
        if self.close_after_runtime_pack_download:
            self.close_after_runtime_pack_download = False
            self.window.close()

    def _preflight_conversion_checks(self) -> bool:
        if self.model_download_in_progress:
            self._show_info('Download in progress', 'Model is downloading. Please wait.')
            return False
        if self.runtime_pack_download_in_progress:
            self._show_info('Runtime pack download in progress', 'A GPU runtime pack is currently downloading. Please wait.')
            return False
        if self.cudnn_install_in_progress:
            self._show_info('cuDNN install in progress', 'cuDNN installation is currently running. Please wait.')
            return False
        if self.transcription_in_progress:
            return False
        if not model_service.get_existing_model_path():
            self._prompt_model_download_if_missing()
            return False
        if self.selected_file is None:
            self._show_warning('No file', 'Please choose an audio file.')
            return False
        if not self._ensure_runtime_available(revert_to_cpu_when_checking=False, revert_to_cpu_when_missing=False):
            return False
        return True

    def _prepare_gpu_runtime_for_conversion(self) -> bool:
        if self.config.device_preference != 'gpu':
            return True
        self._show_gpu_setup_notice_if_needed()
        if not self._ensure_runtime_pack_selected(prompt_if_missing=True):
            self._revert_to_cpu_preference(report_save_errors=False)
            self._start_runtime_probe()
            return False
        model_path = model_service.get_existing_model_path()
        if model_path is None:
            self._show_warning('Model missing', 'Transcription model is missing. Please download it first.')
            return False
        cached = self._cached_gpu_validation_decision(model_path)
        if cached is None:
            if self._gpu_validation_in_progress:
                self._show_info('Validating GPU support', 'ONNX GPU support is still being validated. Please try conversion again in a moment.')
                return False
            self._start_gpu_runtime_validation(show_message=True)
            self._show_info('Validating GPU support', 'ONNX GPU support validation started. Please try conversion again once validation completes.')
            return False
        if cached.device_mode != 'gpu':
            self._apply_runtime_decision(cached, show_message=True)
            self._apply_state_transaction(refresh_first=False, report_save_errors=False, start_runtime_probe=True)
            return self.config.device_preference != 'gpu'
        return True

    def start_conversion(self) -> None:
        if not self._preflight_conversion_checks():
            return
        if not self._prepare_gpu_runtime_for_conversion():
            return
        self.transcription_in_progress = True
        self._apply_controls_state()
        self.window.set_progress(0, '0%')
        self.window.set_console_text('Starting conversion...\nRuntime: ONNX Runtime\nPlease wait, processing...')
        modern_selected = normalize_conversion_method(self.config.conversion_method) == 'modern'
        modern_manual_tuning = modern_selected and (not bool(self.config.modern_auto_calibration_enabled))
        conversion_options = conversion_service.ConversionOptions(
            conversion_method=normalize_conversion_method(self.config.conversion_method),
            modern_adaptive_thresholds_enabled=bool(self.config.modern_adaptive_thresholds_enabled),
            modern_input_normalization_enabled=bool(self.config.modern_input_normalization_enabled),
            modern_smart_overlap_stitching_enabled=bool(self.config.modern_smart_overlap_stitching_enabled),
            modern_auto_calibration_enabled=bool(self.config.modern_auto_calibration_enabled),
            modern_threshold_bias_scale=1.0,
            modern_cleanup_scale=float(self.config.modern_cleanup_scale if modern_manual_tuning else 1.0),
            modern_pedal_cluster_scale=float(self.config.modern_pedal_cluster_scale if modern_manual_tuning else 1.0),
            modern_alignment_gate_scale=float(self.config.modern_alignment_gate_scale if modern_manual_tuning else 1.0),
            modern_manual_onset_threshold=float(self.config.modern_manual_onset_threshold),
            modern_manual_offset_threshold=float(self.config.modern_manual_offset_threshold),
            modern_manual_frame_threshold=float(self.config.modern_manual_frame_threshold),
            modern_manual_pedal_offset_threshold=float(self.config.modern_manual_pedal_offset_threshold),
        )
        worker = ConversionWorker(self.selected_file, conversion_options=conversion_options)
        thread = self._create_worker_thread(worker)
        worker.progressChanged.connect(self.window.set_progress)
        worker.logChanged.connect(self.window.set_console_text)
        worker.errorRaised.connect(self._on_conversion_error)
        thread.finished.connect(self._on_conversion_finished)
        self._conversion_thread = thread
        self._conversion_worker = worker
        thread.start()

    def stop_conversion(self) -> None:
        if not self.transcription_in_progress or self._conversion_worker is None:
            return
        self._conversion_worker.stop()
        self.window.set_console_text('Stopping transcription...\nPlease wait...')
        self.window.set_progress(self.window.progress_bar.value(), 'Stopping...')
        self._apply_controls_state()

    def _on_conversion_error(self, message: str) -> None:
        if self._handle_gpu_cpu_fallback_error(message):
            return
        self._show_error('Error', message)

    def _on_conversion_finished(self) -> None:
        self.transcription_in_progress = False
        self._conversion_thread = None
        self._conversion_worker = None
        self._apply_controls_state()
        if self.close_after_transcription:
            self.close_after_transcription = False
            self.window.close()

    def _switch_to_cpu_mode(self) -> None:
        conversion_service.reset_transcriptor()
        self._revert_to_cpu_preference(start_runtime_probe=True, report_save_errors=False)

    def _show_gpu_fallback_message(self) -> None:
        self._show_info('GPU unavailable', 'GPU acceleration is unfortunately not supported on this system right now.\nA2M is falling back to CPU mode automatically.')

    def _revert_to_cpu_preference(self, *, start_runtime_probe: bool=False, report_save_errors: bool=False) -> None:
        self.config.device_preference = runtime_service.set_device_preference('cpu')
        self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference('auto')
        self._reset_cuda_session_validation()
        self._apply_state_transaction(refresh_first=True, report_save_errors=report_save_errors, start_runtime_probe=start_runtime_probe, apply_controls=True)

    def _ensure_runtime_available(self, *, revert_to_cpu_when_checking: bool=False, revert_to_cpu_when_missing: bool=False) -> bool:
        if self._runtime_checking:
            if revert_to_cpu_when_checking:
                self._revert_to_cpu_preference(report_save_errors=False)
            self._show_info('Checking runtime', RUNTIME_INFO_CHECKING)
            return False
        if not self._runtime_available:
            if revert_to_cpu_when_missing:
                self._revert_to_cpu_preference(report_save_errors=False)
            self._show_runtime_reinstall_required()
            return False
        return True

    def _ensure_gpu_runtime_ready(self, *, prompt_if_missing: bool, revert_when_checking: bool=True) -> bool:
        if not self._ensure_runtime_available(
            revert_to_cpu_when_checking=revert_when_checking,
            revert_to_cpu_when_missing=True,
        ):
            return False
        if not self._ensure_runtime_pack_selected(prompt_if_missing=prompt_if_missing):
            self._revert_to_cpu_preference(start_runtime_probe=True, report_save_errors=False)
            return False
        return True

    @staticmethod
    def _is_gpu_cpu_fallback_error(message: str) -> bool:
        normalized = str(message or '').strip().lower()
        return 'gpu mode selected, but onnx session started on cpu' in normalized

    def _handle_gpu_cpu_fallback_error(self, message: str) -> bool:
        if not self._is_gpu_cpu_fallback_error(message):
            return False
        self._switch_to_cpu_mode()
        self._show_gpu_fallback_message()
        return True

    def _should_prompt_cuda_toolkit_install(self, decision: RuntimeDecision) -> bool:
        if decision.device_mode == 'gpu':
            return False
        reason_code = str(decision.reason_code or '').strip()
        if reason_code not in {REASON_PROVIDER_LOAD_FAILED, REASON_SESSION_CREATION_FAILED, REASON_PROVIDER_NOT_EXPOSED}:
            return False
        return runtime_pack_service.resolve_provider_for_preference(self.config.gpu_provider_preference) == 'cuda'

    def _on_device_preference_changed(self, preference: str) -> None:
        if self.runtime_pack_download_in_progress:
            return
        pref = str(preference or 'cpu').strip().lower()
        if pref not in {'cpu', 'gpu'}:
            pref = 'cpu'
        if pref == 'gpu':
            self._show_gpu_setup_notice_if_needed()
            if not self._ensure_gpu_runtime_ready(prompt_if_missing=True):
                return
            self.config.device_preference = runtime_service.set_device_preference('gpu')
            self._clear_gpu_runtime_validation_state()
            self._apply_state_transaction(refresh_first=False, start_runtime_probe=True, apply_controls=True)
            return
        else:
            self.config.device_preference = runtime_service.set_device_preference(pref)
            if pref != 'gpu':
                self._clear_gpu_runtime_validation_state()
            conversion_service.reset_transcriptor()
        self._apply_state_transaction(refresh_first=False, start_runtime_probe=True, apply_controls=True)

    def _on_gpu_provider_preference_changed(self, preference: str) -> None:
        normalized = runtime_service.set_gpu_provider_preference(preference)
        if normalized == self.config.gpu_provider_preference:
            self._refresh_settings_values()
            return
        self.config.gpu_provider_preference = normalized
        self._reset_cuda_session_validation()
        if not self.transcription_in_progress:
            conversion_service.reset_transcriptor()
        self._apply_state_transaction(refresh_first=False)
        if self.config.device_preference == 'gpu' and (not self.transcription_in_progress):
            self._show_gpu_setup_notice_if_needed()
            if not self._ensure_gpu_runtime_ready(prompt_if_missing=True, revert_when_checking=False):
                return
            self._clear_gpu_runtime_validation_state()
            self._refresh_runtime_status()
            self._apply_state_transaction(persist=False, apply_controls=True)
            self._start_gpu_runtime_validation(show_message=True)

    def _on_conversion_method_changed(self, method: str) -> None:
        normalized = normalize_conversion_method(method)
        if normalized == normalize_conversion_method(self.config.conversion_method):
            self._refresh_settings_values()
            return
        self.config.conversion_method = normalized
        self._apply_state_transaction(refresh_first=False)

    def _on_modern_option_changed(self, field_name: str, enabled: bool) -> None:
        if not hasattr(self.config, field_name):
            return
        setattr(self.config, field_name, bool(enabled))
        self._save_config()

    def _on_modern_threshold_changed(self, field_name: str, value: float, *, lower: float, upper: float, default: float) -> None:
        if not hasattr(self.config, field_name):
            return
        try:
            parsed = float(value)
        except Exception:
            parsed = float(default)
        if not (parsed == parsed):
            parsed = float(default)
        normalized = max(float(lower), min(float(upper), float(parsed)))
        setattr(self.config, field_name, normalized)
        self._save_config()

    @staticmethod
    def _clamp_modern_scale(value: float | int | str | None, *, default: float=1.0) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = float(default)
        if not (parsed == parsed):
            parsed = float(default)
        return max(0.6, min(2.0, float(parsed)))

    def _on_modern_scale_changed(self, field_name: str, value: float, *, default: float) -> None:
        if not hasattr(self.config, field_name):
            return
        setattr(self.config, field_name, self._clamp_modern_scale(value, default=default))
        self._save_config()

    def _on_gpu_batch_size_changed(self, value: int) -> None:
        self.config.gpu_batch_size = runtime_service.set_gpu_batch_size(value)
        conversion_service.reset_transcriptor()
        self._save_config()

    def _on_ui_scale_percent_changed(self, value: int) -> None:
        self.config.ui_scale_percent = int(value)
        self.window.set_ui_scale_percent(self.config.ui_scale_percent)
        self._save_config()

    def _on_download_location_changed(self, path: str) -> None:
        candidate = str(path or '').strip()
        if not candidate:
            return
        self.config.download_location = str(conversion_service.set_output_midi_dir(candidate))
        self._apply_state_transaction(refresh_first=True)

    def _on_auto_check_updates_changed(self, enabled: bool) -> None:
        self.config.auto_check_updates = bool(enabled)
        self._save_config()

    def _on_theme_mode_changed(self, mode: str) -> None:
        normalized = 'light' if str(mode).strip().lower() == 'light' else 'dark'
        self.config.theme_mode = normalized
        self.window.set_theme(get_theme(normalized), normalized)
        self._save_config()

    def _on_reset_settings_requested(self) -> None:
        if (
            self.transcription_in_progress
            or self.model_download_in_progress
            or self.runtime_pack_download_in_progress
            or self.cudnn_install_in_progress
            or self.update_check_in_progress
        ):
            self._show_info('Reset blocked', 'Stop active operations before resetting settings.')
            return
        answer = self._ask_question(
            'Reset settings',
            'Reset all settings to defaults?\n\nYour current window size will be kept.',
            default_button=QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        defaults = default_config()
        try:
            defaults.window_geometry = self.window.saveGeometry().toBase64().data().decode('ascii')
        except Exception:
            defaults.window_geometry = self.config.window_geometry
        self.config = defaults
        self.config.device_preference = runtime_service.set_device_preference(self.config.device_preference)
        self.config.gpu_batch_size = runtime_service.set_gpu_batch_size(self.config.gpu_batch_size)
        self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference(self.config.gpu_provider_preference)
        self.config.gpu_runtime_enabled, self.config.gpu_runtime_path = runtime_service.set_gpu_runtime(self.config.gpu_runtime_enabled, self.config.gpu_runtime_path)
        self._sync_runtime_pack_selection(prefer_existing_config=True)
        self.config.download_location = str(conversion_service.set_output_midi_dir(self.config.download_location))
        self._reset_cuda_session_validation()
        self._clear_gpu_runtime_validation_state()
        self._gpu_requirements_notice_shown = False
        conversion_service.reset_transcriptor()
        self.window.set_theme(get_theme(self.config.theme_mode), self.config.theme_mode)
        self.window.set_ui_scale_percent(self.config.ui_scale_percent)
        self._update_model_status()
        self._apply_state_transaction(refresh_first=False, start_runtime_probe=True, apply_controls=True)
        self._show_info('Settings reset', 'All settings were reset to defaults.')

    def check_for_updates(self, *, manual: bool) -> None:
        if self.update_check_in_progress:
            if manual:
                self._show_info('Update check', 'An update check is already in progress.')
            return
        self.update_check_in_progress = True
        worker = UpdateCheckWorker(manual=manual)
        thread = self._create_worker_thread(worker)
        worker.finishedSuccess.connect(self._on_update_check_success)
        worker.finishedStopped.connect(lambda: None)
        worker.errorRaised.connect(lambda msg: self._show_error('Update error', msg))
        thread.finished.connect(self._on_update_check_finished)
        self._update_thread = thread
        self._update_worker = worker
        thread.start()

    def _on_update_check_success(self, payload: object) -> None:
        if isinstance(payload, dict):
            manual = bool(payload.get('manual', False))
            update_available = bool(payload.get('update_available', False))
            latest_display = str(payload.get('latest_display', '') or '')
            download_url = str(payload.get('download_url', '') or '')
        else:
            manual = bool(getattr(payload, 'manual', False))
            update_available = bool(getattr(payload, 'update_available', False))
            latest_display = str(getattr(payload, 'latest_display', '') or '')
            download_url = str(getattr(payload, 'download_url', '') or '')
        if not latest_display and not download_url and (not update_available) and (not manual):
            return
        if update_available:
            prompt = f'A newer version - v{latest_display} is available!\nWould you like to download it now?'
            if self._ask_question('Update available', prompt, default_button=QMessageBox.Yes) == QMessageBox.Yes:
                webbrowser.open(download_url)
                self.window.close()
        elif manual:
            self._show_info('Update check', 'You are already on the latest version.')

    def _on_update_check_finished(self) -> None:
        self.update_check_in_progress = False
        self._update_thread = None
        self._update_worker = None

    def _cancel_update_check(self) -> None:
        worker = self._update_worker
        thread = self._update_thread
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
            for signal in (worker.finishedSuccess, worker.finishedStopped, worker.errorRaised, worker.finished):
                try:
                    signal.disconnect()
                except Exception:
                    pass
        if thread is not None:
            for signal in (thread.started, thread.finished):
                try:
                    signal.disconnect()
                except Exception:
                    pass
            if thread.isRunning():
                thread.quit()
                thread.wait(max(1000, int(UPDATE_CHECK_TIMEOUT_SECONDS * 1000) + 500))
        self.update_check_in_progress = False
        self._update_thread = None
        self._update_worker = None

    def _open_path_in_shell(self, path: Path, *, failure_title: str) -> bool:
        try:
            if os.name == 'nt':
                os.startfile(str(path))
            else:
                webbrowser.open(path.as_uri())
            return True
        except Exception as exc:
            self._show_warning(failure_title, str(exc))
            return False

    def open_downloads_folder(self) -> None:
        path = Path(self.config.download_location).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._show_warning('Open folder failed', str(exc))
            return
        self._open_path_in_shell(path, failure_title='Open folder failed')

    def _handle_close_request(self) -> bool:
        if self.close_after_transcription or self.close_after_model_download or self.close_after_runtime_pack_download or self.close_after_cudnn_install:
            return False
        if self.model_download_in_progress:
            reply = self._ask_question('Download in progress', 'Model download is still in progress.\n\nStop download and exit A2M?', default_button=QMessageBox.No)
            if reply != QMessageBox.Yes:
                return False
            self.close_after_model_download = True
            self.window.set_console_text('Stopping model download...\nPlease wait...')
            if self._model_worker is not None:
                self._model_worker.stop()
            return False
        if self.runtime_pack_download_in_progress:
            reply = self._ask_question('Runtime pack download in progress', 'A runtime pack download is still running.\n\nStop download and exit A2M?', default_button=QMessageBox.No)
            if reply != QMessageBox.Yes:
                return False
            self.close_after_runtime_pack_download = True
            self.window.set_console_text('Stopping runtime pack download...\nPlease wait...')
            if self._runtime_pack_worker is not None:
                self._runtime_pack_worker.stop()
            return False
        if self.cudnn_install_in_progress:
            reply = self._ask_question('cuDNN install in progress', 'cuDNN installation is still running.\n\nStop install and exit A2M?', default_button=QMessageBox.No)
            if reply != QMessageBox.Yes:
                return False
            self.close_after_cudnn_install = True
            self.window.set_console_text('Stopping cuDNN installation...\nPlease wait...')
            if self._cudnn_worker is not None:
                self._cudnn_worker.stop()
            return False
        if self.transcription_in_progress:
            reply = self._ask_question('Transcription in progress', 'A transcription is still running.\n\nStop transcription and exit A2M?', default_button=QMessageBox.No)
            if reply != QMessageBox.Yes:
                return False
            self.close_after_transcription = True
            self.stop_conversion()
            return False
        if self.update_check_in_progress:
            self._cancel_update_check()
        self._persist_config()
        return True


