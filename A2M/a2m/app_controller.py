from __future__ import annotations
import os
import sys
import threading
import webbrowser
from pathlib import Path
from PySide6.QtCore import QByteArray, QEvent, QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QVBoxLayout, QWidget
from a2m.core import conversion_service, cuda_dependency_service, gpu_runtime_service, model_service, resource_service, runtime_pack_service, runtime_service
from a2m.core.config_service import AppConfig, default_config, load_config, save_config
from a2m.core.config import APP_RESTART_EXIT_CODE, CUDA_DOWNLOAD_URL, ENGINE_UNIFORM_VELOCITY_DEFAULT, ENGINE_UNIFORM_VELOCITY_MAX, ENGINE_UNIFORM_VELOCITY_MIN, OFFICIAL_PAGE_URL, SUPPORTED_AUDIO_FILTER
from a2m.core.gpu_helper import REASON_PROVIDER_LOAD_FAILED, REASON_PROVIDER_NOT_EXPOSED, REASON_SESSION_CREATION_FAILED
from a2m.core.gpu_runtime_manager import GpuRuntimeManager, RuntimeDecision
from a2m.core.runtime_inventory_service import ProviderRuntimeStatus, RuntimeInventory
from a2m.core.messages import RUNTIME_INFO_CHECKING, RUNTIME_WARNING_MISSING
from a2m.core.paths import icon_path, normalized_path_key
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
from a2m.workers.update_install_worker import UpdateInstallWorker

class AppController(QObject):
    runtimeProbeReady = Signal(int, bool, bool, bool)
    gpuValidationReady = Signal(int, object, bool, object)
    _STARTUP_RUNTIME_PROMPTS = {
        'Verifying ONNX Runtime support...\nPlease wait...',
        'Validating ONNX GPU support...\nPlease wait...',
    }

    def __init__(self, app, *, startup_runtime_inventory: object=None) -> None:
        super().__init__()
        self.app = app
        self.gpu_runtime_manager = GpuRuntimeManager()
        self._runtime_inventory = (
            startup_runtime_inventory
            if isinstance(startup_runtime_inventory, RuntimeInventory)
            else RuntimeInventory()
        )
        self._startup_inventory_complete = isinstance(startup_runtime_inventory, RuntimeInventory)
        self._runtime_probe_cache: dict[str, dict[str, object]] = dict(self._runtime_inventory.probes)
        try:
            cuda_dependency_service.ensure_cuda_runtime_bins_in_process_path()
        except Exception as exc:
            print(f'[A2M] Warning: CUDA dependency bootstrap failed: {exc}', file=sys.stderr)
        self.config: AppConfig = load_config()
        startup_provider_before = str(self.config.gpu_provider_preference or '').strip().lower()
        startup_device_before = str(self.config.device_preference or '').strip().lower()
        startup_runtime_before = (bool(self.config.gpu_runtime_enabled), str(self.config.gpu_runtime_path or '').strip())
        startup_cpu_performance_before = str(self.config.cpu_performance_mode or '').strip().lower()
        startup_gpu_performance_before = str(self.config.gpu_performance_mode or '').strip().lower()
        if not self.config.hardware_defaults_initialized:
            self.config.cpu_performance_mode = resource_service.recommended_performance_mode()
        self.config.cpu_performance_mode = resource_service.normalize_performance_mode(self.config.cpu_performance_mode)
        self.config.gpu_performance_mode = resource_service.normalize_performance_mode(self.config.gpu_performance_mode)
        self.config.device_preference = runtime_service.set_device_preference(self.config.device_preference)
        self._activate_performance_mode_for_device()
        self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference(self.config.gpu_provider_preference)
        self.config.gpu_runtime_enabled, self.config.gpu_runtime_path = runtime_service.set_gpu_runtime(self.config.gpu_runtime_enabled, self.config.gpu_runtime_path)
        self._sync_runtime_pack_selection(prefer_existing_config=True)
        if self.config.device_preference == 'gpu' and not self.config.gpu_runtime_enabled:
            packaged_provider = gpu_runtime_service.detect_runtime_provider(
                runtime_service.get_packaged_runtime_root()
            )
            if packaged_provider != self.config.gpu_provider_preference:
                self.config.device_preference = runtime_service.set_device_preference('cpu')
                self._activate_performance_mode_for_device('cpu')
        startup_runtime_after = (bool(self.config.gpu_runtime_enabled), str(self.config.gpu_runtime_path or '').strip())
        self._config_changed_during_startup = (
            (startup_provider_before != self.config.gpu_provider_preference)
            or (startup_device_before != self.config.device_preference)
            or (startup_runtime_before != startup_runtime_after)
            or (startup_cpu_performance_before != self.config.cpu_performance_mode)
            or (startup_gpu_performance_before != self.config.gpu_performance_mode)
        )
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
        self.close_after_update_operation = False
        self._restart_after_gpu_setup = False
        self._restart_prompt_pending = False
        self._update_installer_handoff_in_progress = False
        self._runtime_checking = not self._startup_inventory_complete
        self._runtime_probe_token = 0
        self._runtime_probe_target_path = ''
        self._runtime_probed_path = ''
        self._runtime_available = False
        self._cuda_available = False
        self._dml_available = False
        self._model_available = False
        self._gpu_validation_token = 0
        self._gpu_validation_in_progress = False
        self._gpu_validation_cache: dict[tuple[str, str, str], RuntimeDecision] = {}
        self._seed_startup_gpu_validation_cache()
        self._gpu_validated_this_session = False
        self._show_next_gpu_validation_failure = False
        self._gpu_requirements_notice_shown = False
        self._pending_gpu_provider_switch = ''
        self._pending_gpu_provider_snapshot: dict[str, object] | None = None
        self._pending_gpu_device_enable = False
        self._gpu_provider_switch_rollback: dict[str, object] | None = (
            dict(self.config.gpu_provider_rollback) if self.config.gpu_provider_rollback else None
        )
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
        self._update_check_manual = False
        self._update_install_thread: QThread | None = None
        self._update_install_worker: UpdateInstallWorker | None = None
        self._update_install_fallback_url = ''
        self._update_progress_dialog: QProgressDialog | None = None
        self.window = MainWindow(theme=get_theme(self.config.theme_mode), icon_path=icon_path(), theme_mode=self.config.theme_mode, ui_scale_percent=self.config.ui_scale_percent)
        self.window.set_close_handler(self._handle_close_request)
        self._restore_window_geometry()
        self._refresh_settings_values()
        self.window.set_settings_visible(False, animated=False)
        self.window.set_progress(0, '0%')
        self.window.set_console_text('')
        if not self._startup_inventory_complete:
            self._show_startup_runtime_prompt('Verifying ONNX Runtime support...\nPlease wait...')
        self.window.set_runtime_status(checking=True, runtime_available=False, cuda_available=False, dml_available=False, active_provider='CPU')
        self._dialogs = DialogCoordinator(self.window, apply_theme=self._apply_dialog_theme, exec_dialog=self._exec_dialog)
        self._connect_signals()
        self._update_model_status()
        self._apply_controls_state()
        if self._config_changed_during_startup:
            self._save_config(report_errors=False)
        self._start_runtime_probe()
        QTimer.singleShot(80, self._run_startup_prompts)

    def run(self) -> None:
        self.window.show()

    def _connect_signals(self) -> None:
        self.window.chooseFileRequested.connect(self.choose_file)
        self.window.convertRequested.connect(self.start_conversion)
        self.window.stopRequested.connect(self.stop_conversion)
        self.window.openDownloadsRequested.connect(self.open_downloads_folder)
        self.window.officialPageRequested.connect(lambda: webbrowser.open(OFFICIAL_PAGE_URL))
        self.window.devicePreferenceChanged.connect(self._on_device_preference_changed)
        self.window.gpuProviderPreferenceChanged.connect(self._on_gpu_provider_preference_changed)
        self.window.enginePedalsEnabledChanged.connect(self._on_engine_pedals_enabled_changed)
        self.window.engineVelocityModeChanged.connect(self._on_engine_velocity_mode_changed)
        self.window.engineUniformVelocityChanged.connect(self._on_engine_uniform_velocity_changed)
        self.window.transcriptionPerformanceModeChanged.connect(self._on_transcription_performance_mode_changed)
        self.window.gpuMemoryUsageChanged.connect(self._on_gpu_memory_usage_changed)
        self.window.resetEngineSettingsRequested.connect(self._on_reset_engine_settings_requested)
        self.window.uiScalePercentChanged.connect(self._on_ui_scale_percent_changed)
        self.window.downloadLocationChanged.connect(self._on_download_location_changed)
        self.window.autoCheckUpdatesChanged.connect(self._on_auto_check_updates_changed)
        self.window.checkUpdatesNowRequested.connect(lambda: self.check_for_updates(manual=True))
        self.window.resetSettingsRequested.connect(self._on_reset_settings_requested)
        self.window.themeModeChanged.connect(self._on_theme_mode_changed)
        self.runtimeProbeReady.connect(self._on_runtime_probe_ready)
        self.gpuValidationReady.connect(self._on_gpu_validation_ready)

    @staticmethod
    def _normalize_runtime_path(path: str | Path | None) -> str:
        return str(path or '').strip()

    @staticmethod
    def _runtime_cache_key(path: str | Path | None) -> str:
        candidate = str(path or '').strip()
        return normalized_path_key(candidate) if candidate else ''

    def _provider_runtime_status(self, provider: str | None) -> ProviderRuntimeStatus:
        return self._runtime_inventory.provider_status(provider)

    def _cached_runtime_probe(self, runtime_path: str | Path | None) -> dict[str, object] | None:
        key = self._runtime_cache_key(runtime_path)
        return self._runtime_probe_cache.get(key) if key else None

    def _remember_runtime_probe(self, runtime_path: str | Path | None, probe: dict[str, object]) -> None:
        key = self._runtime_cache_key(runtime_path)
        if key:
            self._runtime_probe_cache[key] = dict(probe)

    def _seed_startup_gpu_validation_cache(self) -> None:
        model_path = str(self._runtime_inventory.model_path or '').strip()
        if not model_path:
            return
        try:
            model_key = str(Path(model_path).resolve())
        except Exception:
            model_key = model_path
        for provider in ('cuda', 'dml'):
            status = self._provider_runtime_status(provider)
            if not status.runtime_path or status.decision is None:
                continue
            runtime_key = self._runtime_cache_key(status.runtime_path)
            self._gpu_validation_cache[(provider, runtime_key, model_key)] = status.decision

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
        configured_probe = self._cached_runtime_probe(configured_path)
        config_runtime_is_valid = bool(
            configured_enabled
            and configured_path
            and (
                bool(configured_probe.get('runtime_available', False))
                if configured_probe is not None
                else runtime_service.is_runtime_path_valid(configured_path)
            )
        )
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

    def _target_gpu_provider_for_active_runtime(self) -> str:
        return self._resolved_gpu_provider()

    def _reset_cuda_session_validation(self, *, clear_cache: bool=True) -> None:
        self._gpu_validation_token += 1
        if clear_cache:
            self._gpu_validation_cache.clear()
        self._gpu_validation_in_progress = False
        self._gpu_validated_this_session = False

    @staticmethod
    def _runtime_source_matches_provider_preference(provider_preference: str, *, cuda_available: bool, dml_available: bool) -> bool:
        pref = str(provider_preference or 'dml').strip().lower()
        if pref == 'cuda':
            return bool(cuda_available)
        return bool(dml_available)

    def _runtime_path_matches_provider_preference(self, runtime_path: str | Path | None) -> bool:
        candidate = self._normalize_runtime_path(runtime_path)
        if not candidate:
            return False
        active_runtime = self._normalize_runtime_path(runtime_service.get_effective_runtime_root_for_gpu_checks())
        probed_runtime = self._normalize_runtime_path(getattr(self, '_runtime_probed_path', ''))
        if (
            active_runtime
            and candidate.lower() == active_runtime.lower()
            and probed_runtime
            and candidate.lower() == probed_runtime.lower()
            and (not self._runtime_checking)
        ):
            if not self._runtime_available:
                return False
            return self._runtime_source_matches_provider_preference(
                self.config.gpu_provider_preference,
                cuda_available=self._cuda_available,
                dml_available=self._dml_available,
            )
        probe = self._cached_runtime_probe(candidate)
        if probe is None:
            probe = self.gpu_runtime_manager.probe_runtime_path(candidate)
            self._remember_runtime_probe(candidate, probe)
        if not bool(probe.get('runtime_available', False)):
            return False
        return self._runtime_source_matches_provider_preference(
            self.config.gpu_provider_preference,
            cuda_available=bool(probe.get('cuda_available', False)),
            dml_available=bool(probe.get('dml_available', False)),
        )

    @staticmethod
    def _probe_supports_provider(probe: dict[str, object], provider: str) -> bool:
        normalized = runtime_pack_service.resolve_provider_for_preference(provider)
        if not bool(probe.get('runtime_available', False)):
            return False
        if normalized == 'cuda':
            return bool(probe.get('cuda_available', False))
        return bool(probe.get('dml_available', False))

    def _runtime_source_for_provider(self, provider: str) -> str:
        normalized = runtime_pack_service.resolve_provider_for_preference(provider)
        known_status = self._provider_runtime_status(normalized)
        if known_status.available and known_status.runtime_path:
            return self._normalize_runtime_path(known_status.runtime_path)
        candidates: list[str] = []
        _installed_provider, installed_path = runtime_pack_service.resolve_installed_pack(normalized)
        if installed_path:
            return self._normalize_runtime_path(installed_path)
        for raw_candidate in self._runtime_candidates():
            candidate = self._normalize_runtime_path(raw_candidate)
            if not candidate or any(existing.lower() == candidate.lower() for existing in candidates):
                continue
            candidates.append(candidate)
        for candidate in candidates:
            probe = self._cached_runtime_probe(candidate)
            if probe is None:
                probe = self.gpu_runtime_manager.probe_runtime_path(candidate)
                self._remember_runtime_probe(candidate, probe)
            if self._probe_supports_provider(probe, normalized):
                return candidate
        return ''

    def _known_available_runtime_path(self, provider: str) -> str:
        normalized = runtime_pack_service.resolve_provider_for_preference(provider)
        known_status = self._provider_runtime_status(normalized)
        if known_status.available and known_status.runtime_path:
            return self._normalize_runtime_path(known_status.runtime_path)
        if (
            not self._runtime_checking
            and normalized == self._resolved_gpu_provider()
            and self._runtime_available
            and self._runtime_source_matches_provider_preference(
                normalized,
                cuda_available=self._cuda_available,
                dml_available=self._dml_available,
            )
        ):
            return self._normalize_runtime_path(
                runtime_service.get_effective_runtime_root_for_gpu_checks()
            )
        if not self._startup_inventory_complete:
            return self._runtime_source_for_provider(normalized)
        return ''

    def _capture_gpu_provider_state(self) -> dict[str, object]:
        return {
            'provider': str(self.config.gpu_provider_preference or 'dml').strip().lower(),
            'runtime_enabled': bool(self.config.gpu_runtime_enabled),
            'runtime_path': str(self.config.gpu_runtime_path or '').strip(),
            'device': str(self.config.device_preference or 'cpu').strip().lower(),
            'gpu_batch_size': int(self.config.gpu_batch_size),
            'gpu_memory_max_batch': int(self.config.gpu_memory_max_batch),
        }

    def _set_gpu_provider_switch_rollback(self, snapshot: dict[str, object] | None) -> None:
        normalized = dict(snapshot) if snapshot else None
        self._gpu_provider_switch_rollback = normalized
        self.config.gpu_provider_rollback = dict(normalized) if normalized else None

    def _clear_gpu_provider_switch_rollback(self) -> None:
        self._set_gpu_provider_switch_rollback(None)

    def _clear_pending_gpu_provider_switch(self) -> None:
        self._pending_gpu_provider_switch = ''
        self._pending_gpu_provider_snapshot = None

    def _activate_gpu_provider_switch(
        self,
        provider: str,
        runtime_path: str,
        snapshot: dict[str, object],
        *,
        enable_gpu: bool=False,
    ) -> bool:
        normalized = runtime_pack_service.resolve_provider_for_preference(provider)
        enabled, selected_path = runtime_service.set_gpu_runtime(True, runtime_path)
        if not enabled or not selected_path:
            runtime_service.set_gpu_runtime(
                bool(snapshot.get('runtime_enabled', False)),
                str(snapshot.get('runtime_path', '') or ''),
            )
            return False
        restart_required = runtime_service.is_runtime_restart_required()
        if str(snapshot.get('device', 'cpu')).strip().lower() == 'gpu':
            self._set_gpu_provider_switch_rollback(snapshot)
        else:
            self._clear_gpu_provider_switch_rollback()
        self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference(normalized)
        self.config.gpu_runtime_enabled = bool(enabled)
        self.config.gpu_runtime_path = str(selected_path)
        if enable_gpu:
            self.config.device_preference = runtime_service.set_device_preference('gpu')
            self._activate_performance_mode_for_device('gpu', reset_session=True)
        presets = resource_service.gpu_memory_presets(self.config.gpu_memory_max_batch)
        self.config.gpu_batch_size = presets['balanced']
        runtime_service.set_gpu_batch_size(self.config.gpu_batch_size)
        self._reset_cuda_session_validation(clear_cache=False)
        self._clear_gpu_runtime_validation_state()
        self._show_next_gpu_validation_failure = True
        if not self.transcription_in_progress:
            conversion_service.reset_transcriptor()
        self._apply_state_transaction(
            refresh_first=False,
            start_runtime_probe=not restart_required,
            apply_controls=True,
        )
        if restart_required:
            self._request_runtime_backend_restart(normalized)
        return True

    def _restore_gpu_provider_switch(self) -> tuple[str, str] | None:
        snapshot = self._gpu_provider_switch_rollback or self.config.gpu_provider_rollback
        self._clear_gpu_provider_switch_rollback()
        if not snapshot:
            return None
        previous_provider = runtime_pack_service.resolve_provider_for_preference(str(snapshot.get('provider', 'dml')))
        failed_provider = runtime_pack_service.resolve_provider_for_preference(self.config.gpu_provider_preference)
        self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference(previous_provider)
        enabled, runtime_path = runtime_service.set_gpu_runtime(
            bool(snapshot.get('runtime_enabled', False)),
            str(snapshot.get('runtime_path', '') or ''),
        )
        self.config.gpu_runtime_enabled = bool(enabled)
        self.config.gpu_runtime_path = str(runtime_path or '')
        device = str(snapshot.get('device', 'cpu') or 'cpu').strip().lower()
        self.config.device_preference = runtime_service.set_device_preference(device)
        self.config.gpu_memory_max_batch = resource_service.normalize_gpu_memory_max_batch(
            snapshot.get('gpu_memory_max_batch', self.config.gpu_memory_max_batch)
        )
        self.config.gpu_batch_size = min(
            resource_service.normalize_gpu_batch_size(snapshot.get('gpu_batch_size', self.config.gpu_batch_size)),
            self.config.gpu_memory_max_batch,
        )
        self._reset_cuda_session_validation()
        self._clear_gpu_runtime_validation_state()
        self._show_next_gpu_validation_failure = False
        self._restart_prompt_pending = False
        self._restart_after_gpu_setup = False
        self._activate_performance_mode_for_device(self.config.device_preference, reset_session=True)
        return (failed_provider, previous_provider)

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
            gpu_provider_preference=self.config.gpu_provider_preference,
            engine_pedals_enabled=self.config.engine_pedals_enabled,
            engine_velocity_mode=self.config.engine_velocity_mode,
            engine_uniform_velocity=self.config.engine_uniform_velocity,
            cpu_performance_mode=self.config.cpu_performance_mode,
            gpu_performance_mode=self.config.gpu_performance_mode,
            gpu_batch_size=self.config.gpu_batch_size,
            gpu_memory_max_batch=self.config.gpu_memory_max_batch,
            ui_scale_percent=self.config.ui_scale_percent,
            download_location=self.config.download_location,
            auto_check_updates=self.config.auto_check_updates,
        )

    def _performance_mode_for_device(self, device: str | None = None) -> str:
        selected_device = resource_service.normalize_performance_device(device or self.config.device_preference)
        if selected_device == 'gpu':
            return resource_service.normalize_performance_mode(self.config.gpu_performance_mode)
        return resource_service.normalize_performance_mode(self.config.cpu_performance_mode)

    def _set_performance_mode_for_device(self, device: str, mode: str) -> str:
        selected_device = resource_service.normalize_performance_device(device)
        normalized = resource_service.normalize_performance_mode(mode)
        if selected_device == 'gpu':
            self.config.gpu_performance_mode = normalized
        else:
            self.config.cpu_performance_mode = normalized
        return normalized

    def _activate_performance_mode_for_device(self, device: str | None = None, *, reset_session: bool = False) -> str:
        selected_device = resource_service.normalize_performance_device(device or self.config.device_preference)
        active_mode = resource_service.set_performance_mode(self._performance_mode_for_device(selected_device))
        if selected_device == 'gpu':
            self.config.gpu_memory_max_batch = resource_service.normalize_gpu_memory_max_batch(
                self.config.gpu_memory_max_batch
            )
            self.config.gpu_batch_size = min(
                resource_service.normalize_gpu_batch_size(self.config.gpu_batch_size),
                self.config.gpu_memory_max_batch,
            )
            runtime_service.set_gpu_batch_size(self.config.gpu_batch_size)
        else:
            runtime_service.set_gpu_batch_size(1)
        if reset_session:
            conversion_service.reset_transcriptor()
        return active_mode

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
        provider = self._target_gpu_provider_for_active_runtime()
        runtime_root = runtime_service.get_effective_runtime_root_for_gpu_checks()
        runtime_path = self._runtime_cache_key(runtime_root)
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
        provider, runtime_key, _model_key = key
        runtime_path = str(runtime_service.get_effective_runtime_root_for_gpu_checks() or '').strip()
        if provider in {'cuda', 'dml'} and runtime_path and runtime_key:
            self._runtime_inventory.providers[provider] = ProviderRuntimeStatus(
                provider=provider,
                runtime_path=runtime_path,
                runtime_installed=True,
                available=decision.device_mode == 'gpu',
                decision=decision,
            )
            self._runtime_inventory.model_path = str(model_path or '')

    def _should_show_gpu_setup_notice(self) -> bool:
        if self._gpu_requirements_notice_shown:
            return False
        if self._runtime_checking:
            return False
        provider = self._target_gpu_provider_for_active_runtime()
        provider_ready = self._runtime_source_matches_provider_preference(
            self.config.gpu_provider_preference,
            cuda_available=self._cuda_available,
            dml_available=self._dml_available,
        )
        missing_cuda_dlls: list[str] = []
        if provider == 'cuda':
            try:
                missing_cuda_dlls = cuda_dependency_service.missing_required_cuda_dlls()
            except Exception as exc:
                print(f'[A2M] Warning: CUDA dependency check failed: {exc}', file=sys.stderr)
                missing_cuda_dlls = ['CUDA dependency check failed']
        return (not provider_ready) or bool(missing_cuda_dlls)

    def _show_gpu_setup_notice_if_needed(self) -> None:
        if not self._should_show_gpu_setup_notice():
            return
        provider = self._target_gpu_provider_for_active_runtime()
        if provider == 'cuda':
            requirement_text = cuda_dependency_service.cuda_requirements_summary()
            message = (
                'GPU acceleration may take time to set up.\n\n'
                'Depending on your system, A2M may need up to three dependencies:\n'
                f'1) ONNX GPU runtime, 2) {requirement_text}.\n\n'
                'A2M can install some dependencies automatically, but the total download size can be large.\n'
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
        if not self._runtime_source_matches_provider_preference(
            self.config.gpu_provider_preference,
            cuda_available=self._cuda_available,
            dml_available=self._dml_available,
        ):
            return False
        model_path = model_service.get_existing_model_path()
        if model_path is None:
            return False
        cached = self._cached_gpu_validation_decision(model_path)
        if cached is not None:
            self._gpu_validation_token += 1
            token = int(self._gpu_validation_token)
            self._gpu_validation_in_progress = True
            self._on_gpu_validation_ready(
                token,
                cached,
                bool(show_message),
                str(model_path),
            )
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
        if decision.device_mode != 'gpu' and self._gpu_provider_switch_rollback:
            restored = self._restore_gpu_provider_switch()
            if restored is not None:
                failed_provider, previous_provider = restored
                failed_label = runtime_pack_service.provider_display_name(failed_provider)
                previous_label = runtime_pack_service.provider_display_name(previous_provider)
                reason = str(decision.reason_text or 'GPU initialization failed.').strip()
                self.window.set_progress(0, f'{failed_label} unavailable; keeping {previous_label}')
                if show_message:
                    self._show_info(
                        f'{failed_label} unavailable',
                        f'{failed_label} could not start, so A2M kept using {previous_label}.\n\nReason: {reason}',
                    )
                self._apply_state_transaction(
                    refresh_first=False,
                    report_save_errors=False,
                    start_runtime_probe=True,
                    apply_controls=True,
                )
                self._clear_startup_runtime_prompt_if_idle()
                return
        if decision.device_mode == 'gpu':
            self._clear_gpu_provider_switch_rollback()
        self._apply_runtime_decision(decision, show_message=bool(show_message) and (decision.device_mode != 'gpu'))
        if decision.device_mode == 'gpu':
            provider_name = runtime_pack_service.provider_display_name(decision.active_provider)
            self.window.set_progress(100, f'{provider_name} acceleration ready')
        else:
            self.window.set_progress(0, 'GPU unavailable; using CPU')
        self._apply_state_transaction(
            refresh_first=False,
            report_save_errors=False,
            start_runtime_probe=(decision.device_mode != 'gpu'),
            apply_controls=True,
        )
        if decision.device_mode != 'gpu':
            self._restart_prompt_pending = False
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
            or self.update_check_in_progress
            or self._update_worker is not None
            or self._update_install_worker is not None
            or self._update_installer_handoff_in_progress
        ):
            return
        self._restart_after_gpu_setup = False
        self._restart_application()

    def _request_runtime_backend_restart(self, provider: str | None = None) -> None:
        already_requested = bool(self._restart_after_gpu_setup)
        self._restart_prompt_pending = False
        self._restart_after_gpu_setup = True
        if not already_requested:
            provider_label = runtime_pack_service.provider_display_name(provider)
            self._show_info(
                'Restart required',
                f'A2M will restart to switch safely to the {provider_label} runtime.',
            )
        self._try_restart_after_gpu_setup()

    def _restart_application(self) -> None:
        self._persist_config()
        self.app.exit(APP_RESTART_EXIT_CODE)

    def _prompt_runtime_pack_install(self, provider: str) -> bool:
        if self.runtime_pack_download_in_progress:
            self._show_info('Runtime pack download', 'A runtime pack download is already in progress.')
            return False
        normalized_provider = runtime_pack_service.resolve_provider_for_preference(provider)
        provider_label = runtime_pack_service.provider_display_name(normalized_provider)
        size_summary = runtime_pack_service.provider_pack_size_summary(normalized_provider)
        pack_url = runtime_pack_service.provider_pack_url(normalized_provider)
        if not pack_url:
            self._show_warning('Runtime pack unavailable', f'{provider_label} runtime pack URL is not configured.')
            return False
        if normalized_provider == 'cuda':
            explanation = (
                'CUDA is intended for NVIDIA graphics cards and can be the fastest option. '
                'It may also require NVIDIA CUDA and cuDNN components.'
            )
        else:
            explanation = (
                'DirectML works with most modern Windows graphics cards and is the easiest '
                'GPU option to set up.'
            )
        prompt = (
            f'GPU acceleration requires the {provider_label} runtime pack.\n\n'
            f'{explanation}\n\nEstimated size: {size_summary}\n\n'
            'Download and install it now?'
        )
        if self._ask_question('GPU runtime pack required', prompt, default_button=QMessageBox.Yes) != QMessageBox.Yes:
            return False
        self._start_runtime_pack_download(normalized_provider)
        return True

    def _choose_gpu_provider_for_setup(self) -> str:
        directml_size = runtime_pack_service.provider_pack_size_summary('dml')
        cuda_size = runtime_pack_service.provider_pack_size_summary('cuda')
        dialog = QMessageBox(self.window)
        dialog.setWindowTitle('Choose GPU acceleration')
        dialog.setIcon(QMessageBox.Information)
        dialog.setText('A2M could not start a GPU backend yet.')
        dialog.setInformativeText(
            'DirectML (recommended) works with most modern Windows graphics cards and has '
            f'the simplest setup. {directml_size}\n\n'
            'CUDA is for NVIDIA graphics cards. It can be faster, but may need additional '
            f'NVIDIA CUDA and cuDNN components. {cuda_size} Additional NVIDIA components '
            'are not included in that estimate.'
        )
        directml_button = dialog.addButton('Set up DirectML (Recommended)', QMessageBox.AcceptRole)
        cuda_button = dialog.addButton('Set up CUDA', QMessageBox.ActionRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(directml_button)
        self._apply_dialog_theme(dialog)
        self._exec_dialog(dialog)
        clicked = dialog.clickedButton()
        if clicked is directml_button:
            return 'dml'
        if clicked is cuda_button:
            return 'cuda'
        return ''

    def _explain_unavailable_provider(self, status: ProviderRuntimeStatus) -> None:
        provider = runtime_pack_service.resolve_provider_for_preference(status.provider)
        provider_label = runtime_pack_service.provider_display_name(provider)
        reason = str(status.reason_text or f'{provider_label} could not start on this system.').strip()
        if provider == 'cuda' and status.runtime_path:
            self._prompt_cuda_dependency_setup(
                f'{provider_label} is installed but cannot start yet.\n\nReason: {reason}',
                runtime_path=status.runtime_path,
            )
            return
        self._show_info(
            f'{provider_label} unavailable',
            f'{provider_label} is installed but could not start on this system.\n\nReason: {reason}',
        )

    def _show_runtime_reinstall_required(self) -> None:
        self._show_warning('ONNX Runtime missing', RUNTIME_WARNING_MISSING)

    def _start_runtime_pack_download(self, provider: str) -> None:
        provider_label = runtime_pack_service.provider_display_name(provider)
        self.runtime_pack_download_in_progress = True
        self._apply_controls_state()
        self.window.set_progress(0, f'Downloading {provider_label} runtime pack 0.00% | ETA --')
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
        style = f'QDialog, QMessageBox {{ background: {theme.panel_bg}; color: {theme.text_primary}; }}QLabel {{ color: {theme.text_primary}; background: transparent; }}QCheckBox {{ color: {theme.text_primary}; background: transparent; }}QCheckBox:disabled {{ color: {theme.disabled_fg}; }}QPushButton {{ background: {theme.panel_bg}; color: {theme.text_primary}; border: 1px solid {theme.border}; border-radius: 6px; padding: 5px 10px; font: 600 9.5pt "Segoe UI"; min-height: 24px; }}QPushButton:hover {{ background: {theme.accent}; color: {theme.text_primary}; }}QPushButton:disabled {{ background: {theme.disabled_bg}; color: {theme.disabled_fg}; border-color: {theme.border}; }}QLineEdit, QListView, QTreeView {{ background: {theme.app_bg}; color: {theme.text_primary}; border: 1px solid {theme.border}; }}'
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
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(theme.disabled_fg))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(theme.disabled_fg))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(theme.disabled_fg))
        palette.setColor(QPalette.Disabled, QPalette.Button, QColor(theme.disabled_bg))
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

    def _ask_update_install_preference(
        self,
        *,
        latest_version: str,
        details_text: str = '',
        install_supported: bool = True,
    ) -> tuple[bool, bool]:
        dialog = QDialog(self.window)
        dialog.setModal(True)
        dialog.setWindowTitle('Update available')
        icon_obj = self.window.windowIcon()
        if not icon_obj.isNull():
            dialog.setWindowIcon(icon_obj)
        self._apply_dialog_theme(dialog)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel(f'Version {str(latest_version or "latest")} is available.', dialog)
        title.setStyleSheet('font: 700 9pt "Segoe UI";')
        layout.addWidget(title)

        auto_install_checkbox = QCheckBox('Install update automatically', dialog)
        auto_install_checkbox.setChecked(bool(install_supported))
        auto_install_checkbox.setEnabled(bool(install_supported))
        layout.addWidget(auto_install_checkbox)

        mode_hint = QLabel('', dialog)
        mode_hint.setWordWrap(True)
        mode_hint.setStyleSheet('font: 600 8pt "Segoe UI";')

        def _refresh_hint() -> None:
            if auto_install_checkbox.isEnabled() and auto_install_checkbox.isChecked():
                mode_hint.setText('A2M will download, verify, install the update, and relaunch automatically.')
            elif auto_install_checkbox.isEnabled():
                mode_hint.setText('A2M will open the download page so you can install the update manually.')
            else:
                mode_hint.setText('Automatic install is unavailable here. A2M will open the download page.')

        auto_install_checkbox.toggled.connect(lambda _checked=False: _refresh_hint())
        _refresh_hint()
        layout.addWidget(mode_hint)

        extra = str(details_text or '').strip()
        if extra:
            details_label = QLabel(extra, dialog)
            details_label.setWordWrap(True)
            details_label.setStyleSheet('font: 600 8pt "Segoe UI";')
            layout.addWidget(details_label)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(8)
        buttons.addStretch(1)
        later_button = QPushButton('Not now', dialog)
        continue_button = QPushButton('Update', dialog)
        continue_button.setDefault(True)
        buttons.addWidget(later_button)
        buttons.addWidget(continue_button)
        layout.addLayout(buttons)

        later_button.clicked.connect(dialog.reject)
        continue_button.clicked.connect(dialog.accept)

        accepted = self._exec_dialog(dialog) == QDialog.Accepted
        auto_install = bool(auto_install_checkbox.isEnabled() and auto_install_checkbox.isChecked())
        return accepted, auto_install

    def _ask_update_handoff(
        self,
        *,
        version: str,
        default_restart: bool = True,
        requires_elevation: bool = False,
    ) -> tuple[bool, bool]:
        dialog = QDialog(self.window)
        dialog.setModal(True)
        dialog.setWindowTitle('Install update')
        icon_obj = self.window.windowIcon()
        if not icon_obj.isNull():
            dialog.setWindowIcon(icon_obj)
        self._apply_dialog_theme(dialog)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel('Update is ready to install', dialog)
        title.setStyleSheet('font: 700 9pt "Segoe UI";')
        body = QLabel(
            (
                f'A2M v{str(version or "latest")} has been downloaded.\n\n'
                'The installer will now take over.\n\n'
                'Click Update to close A2M and begin installation.'
            ),
            dialog,
        )
        body.setWordWrap(True)
        body.setStyleSheet('font: 600 8pt "Segoe UI";')
        uac_hint = QLabel('Windows may ask for administrator permission to continue this update.', dialog)
        uac_hint.setWordWrap(True)
        uac_hint.setStyleSheet('font: 600 8pt "Segoe UI";')
        uac_hint.setVisible(bool(requires_elevation))

        restart_checkbox = QCheckBox('Restart A2M after update', dialog)
        restart_checkbox.setChecked(bool(default_restart))

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(8)
        buttons.addStretch(1)
        abort_button = QPushButton('Abort', dialog)
        continue_button = QPushButton('Update', dialog)
        continue_button.setDefault(True)
        buttons.addWidget(abort_button)
        buttons.addWidget(continue_button)

        layout.addWidget(title)
        layout.addWidget(body)
        layout.addWidget(uac_hint)
        layout.addWidget(restart_checkbox)
        layout.addLayout(buttons)

        abort_button.clicked.connect(dialog.reject)
        continue_button.clicked.connect(dialog.accept)

        accepted = self._exec_dialog(dialog) == QDialog.Accepted
        restart = bool(restart_checkbox.isChecked())
        return accepted, restart

    def _restore_app_cursor_state(self) -> None:
        self.window.restore_cursor_state_after_modal()

    def _exec_dialog(self, dialog: QWidget) -> int:
        try:
            return int(dialog.exec())
        finally:
            self._restore_app_cursor_state()

    def _show_update_progress_dialog(self) -> None:
        dialog = self._update_progress_dialog
        if dialog is not None:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = QProgressDialog('Preparing update...', 'Cancel', 0, 100, self.window)
        dialog.setWindowTitle('Updating A2M')
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        self._apply_dialog_theme(dialog)
        dialog.setStyleSheet(
            dialog.styleSheet()
            + 'QProgressBar { border: 1px solid #2a2a2a; border-radius: 4px; text-align: center; }'
            + 'QProgressBar::chunk { background-color: #2fbf71; border-radius: 3px; }'
        )
        dialog.canceled.connect(self._on_update_progress_canceled)
        self._update_progress_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _update_update_progress_dialog(self, percent: float, text: str) -> None:
        dialog = self._update_progress_dialog
        if dialog is None:
            return
        value = max(0, min(100, int(round(float(percent)))))
        dialog.setValue(value)
        message = str(text or '').strip() or f'Updating... {value}%'
        dialog.setLabelText(message)

    def _on_update_progress_canceled(self) -> None:
        self._update_update_progress_dialog(0, 'Canceling update...')
        for worker in (self._update_worker, self._update_install_worker):
            if worker is None:
                continue
            try:
                worker.stop()
            except Exception:
                pass

    def _close_update_progress_dialog(self) -> None:
        dialog = self._update_progress_dialog
        if dialog is None:
            return
        self._update_progress_dialog = None
        try:
            dialog.close()
        finally:
            dialog.deleteLater()

    def _on_update_install_progress(self, percent: float, text: str) -> None:
        message = str(text or '').strip() or f'Updating... {float(percent):.1f}%'
        self.window.set_progress(percent, message)
        self._update_update_progress_dialog(percent, message)

    def _pick_audio_file(self) -> str:
        selected, _ = QFileDialog.getOpenFileName(self.window, 'Choose audio file', '', SUPPORTED_AUDIO_FILTER)
        return str(selected or '').strip()

    def _start_runtime_probe(self) -> None:
        self._runtime_probe_token += 1
        token = int(self._runtime_probe_token)
        probe_path = self._normalize_runtime_path(
            runtime_service.get_effective_runtime_root_for_gpu_checks()
        )
        self._runtime_probe_target_path = probe_path
        cached_probe = self._cached_runtime_probe(probe_path)
        if cached_probe is not None or (self._startup_inventory_complete and not probe_path):
            probe = cached_probe or {}
            self._runtime_checking = False
            self._on_runtime_probe_ready(
                token,
                bool(probe.get('runtime_available', False)),
                bool(probe.get('cuda_available', False)),
                bool(probe.get('dml_available', False)),
            )
            return
        self._runtime_checking = True
        self.window.set_runtime_status(checking=True, runtime_available=False, cuda_available=False, dml_available=False, active_provider='CPU')
        self.window.set_progress(0, 'Verifying ONNX Runtime...')
        self._show_startup_runtime_prompt('Verifying ONNX Runtime support...\nPlease wait...')
        self._apply_controls_state()

        def worker() -> None:
            runtime_ok = False
            cuda_ok = False
            dml_ok = False
            if probe_path:
                probe = self.gpu_runtime_manager.probe_runtime_path(probe_path)
                self._remember_runtime_probe(probe_path, probe)
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
        self._runtime_probed_path = self._normalize_runtime_path(self._runtime_probe_target_path)
        self._runtime_available = bool(runtime_available)
        self._cuda_available = bool(cuda_available)
        self._dml_available = bool(dml_available)
        provider_ready = self._runtime_source_matches_provider_preference(
            self.config.gpu_provider_preference,
            cuda_available=self._cuda_available,
            dml_available=self._dml_available,
        )
        if self._gpu_provider_switch_rollback and (not provider_ready):
            restored = self._restore_gpu_provider_switch()
            if restored is not None:
                failed_provider, previous_provider = restored
                failed_label = runtime_pack_service.provider_display_name(failed_provider)
                previous_label = runtime_pack_service.provider_display_name(previous_provider)
                self.window.set_progress(0, f'{failed_label} unavailable; keeping {previous_label}')
                self._show_info(
                    f'{failed_label} unavailable',
                    f'{failed_label} could not load on this system, so A2M kept using {previous_label}.',
                )
                self._apply_state_transaction(
                    refresh_first=False,
                    report_save_errors=False,
                    start_runtime_probe=True,
                    apply_controls=True,
                )
                self._clear_startup_runtime_prompt_if_idle()
                return
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
            provider_pref = str(self.config.gpu_provider_preference or 'dml').strip().lower()
            if self._gpu_validated_this_session and validated_provider in {'cuda', 'dml'}:
                if provider_pref == validated_provider:
                    return runtime_pack_service.provider_display_name(validated_provider)
            if self._runtime_source_matches_provider_preference(
                self.config.gpu_provider_preference,
                cuda_available=self._cuda_available,
                dml_available=self._dml_available,
            ):
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
        requested_provider = self._target_gpu_provider_for_active_runtime()
        self._record_runtime_decision(decision)
        conversion_service.reset_transcriptor()
        if decision.device_mode != 'gpu':
            should_prompt_cuda = show_message and requested_provider == 'cuda' and self._should_prompt_cuda_toolkit_install(decision)
            self.config.device_preference = runtime_service.set_device_preference('cpu')
            if show_message:
                fallback_text = f'GPU unavailable. Using CPU fallback: {decision.reason_text or "GPU initialization failed."}'
                if should_prompt_cuda:
                    self._prompt_cuda_dependency_setup(fallback_text)
                else:
                    self._show_info('GPU unavailable', fallback_text)
        else:
            self.config.device_preference = runtime_service.set_device_preference('gpu')
        self._activate_performance_mode_for_device(self.config.device_preference)
        self._refresh_runtime_status()

    def _prompt_cuda_dependency_setup(self, fallback_text: str, *, runtime_path: str | Path | None=None) -> None:
        target_runtime = str(runtime_path or runtime_service.get_effective_runtime_root_for_gpu_checks() or '').strip()
        requirement_text = cuda_dependency_service.cuda_requirements_summary(target_runtime)
        prompt_cuda = fallback_text + f'\n\nCUDA mode requires {requirement_text}.\nWould you like to open the NVIDIA CUDA download page now?'
        if self._ask_question('CUDA required', prompt_cuda, default_button=QMessageBox.Yes) == QMessageBox.Yes:
            webbrowser.open(CUDA_DOWNLOAD_URL)
        cudnn_url = str(cuda_dependency_service.selected_cudnn_download(target_runtime)[0] or '').strip()
        if not cudnn_url:
            return
        cudnn_major = cuda_dependency_service.get_cuda_runtime_requirements(target_runtime).cudnn_major
        prompt_cudnn = f'CUDA requires cuDNN {cudnn_major} to function correctly.\n\nA2M can download cuDNN 9.19 into its private application data and configure it automatically.\n\nWould you like to do that now?'
        if self._ask_question('Install cuDNN', prompt_cudnn, default_button=QMessageBox.Yes) != QMessageBox.Yes:
            return
        self._start_cudnn_install()

    def _start_cudnn_install(self) -> None:
        if self.cudnn_install_in_progress:
            self._show_info('cuDNN install', 'cuDNN installation is already in progress.')
            return
        self.cudnn_install_in_progress = True
        self._apply_controls_state()
        self.window.set_progress(0, 'Downloading 0.00% | ETA --')
        requirement_text = cuda_dependency_service.cuda_requirements_summary(runtime_service.get_effective_runtime_root_for_gpu_checks())
        self.window.set_console_text(f'Downloading cuDNN package for {requirement_text}...\nPlease wait...')
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
        restart_note = '\n\nA2M will restart if the active runtime needs to reload the new libraries.' if user_changed else ''
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
        show_message = bool(self._show_next_gpu_validation_failure)
        if self._start_gpu_runtime_validation(show_message=show_message, startup=True):
            self._show_next_gpu_validation_failure = False

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
            or self._update_install_worker is not None
        )
        self.window.set_controls_state(file_enabled=not busy, convert_enabled=not busy and self._can_convert(), stop_enabled=self.transcription_in_progress)

    def _update_model_status(self) -> None:
        model_path = model_service.get_existing_model_path()
        self._model_available = bool(model_path)
        if model_path:
            self.window.set_model_status(ready=True)
            return
        self.window.set_model_status(ready=False)

    def _run_startup_prompts(self) -> None:
        self._continue_startup_prompts()

    def _continue_startup_prompts(self) -> None:
        if not self.window.isVisible():
            return
        self._prompt_model_download_if_missing()
        if self.window.isVisible() and self.config.auto_check_updates:
            QTimer.singleShot(820, lambda: self.check_for_updates(manual=False))

    def _apply_gpu_memory_level(self, batch_size: int) -> None:
        self.config.gpu_batch_size = min(
            resource_service.normalize_gpu_batch_size(batch_size),
            resource_service.normalize_gpu_memory_max_batch(self.config.gpu_memory_max_batch),
        )
        if self.config.device_preference == 'gpu':
            runtime_service.set_gpu_batch_size(self.config.gpu_batch_size)
            conversion_service.reset_transcriptor()
        self._apply_state_transaction(refresh_first=False, report_save_errors=False)

    def _prompt_model_download_if_missing(self) -> None:
        if model_service.get_existing_model_path() is not None:
            return
        install_hint = model_service.get_model_install_hint()
        prompt = f'A2M needs the Transcription Model package before it can convert audio.\n\nThe approximately 48 MB package will be downloaded once to:\n{install_hint}\n\nWould you like to download it now?'
        should_download = self._ask_question('Transcription Model required', prompt, default_button=QMessageBox.Yes)
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
        self.window.set_progress(0, 'Transcribing 0.00% | ETA --')
        self._apply_controls_state()

    def start_model_download(self) -> None:
        if self.model_download_in_progress:
            return
        self.model_download_in_progress = True
        self._apply_controls_state()
        self.window.set_progress(0, 'Downloading 0.00% | ETA --')
        self.window.set_console_text('Downloading Transcription Model...\nPlease wait...')
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
        pending_device_enable = bool(self._pending_gpu_device_enable)
        self._pending_gpu_device_enable = False
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
        installed_decision: RuntimeDecision | None = None
        validated_model_path = model_service.get_existing_model_path()
        if provider == 'dml' and validated_model_path is not None:
            installed_decision = RuntimeDecision(
                device_mode='gpu',
                active_provider='dml',
                gpu_model_name=gpu_runtime_service.get_gpu_model_name('dml'),
                reason_code='',
                reason_text='',
            )
            try:
                model_key = str(validated_model_path.resolve())
            except Exception:
                model_key = str(validated_model_path)
            self._gpu_validation_cache[
                ('dml', self._runtime_cache_key(runtime_path), model_key)
            ] = installed_decision
            self._runtime_inventory.model_path = str(validated_model_path)
        self._runtime_inventory.providers[provider] = ProviderRuntimeStatus(
            provider=provider,
            runtime_path=runtime_path,
            runtime_installed=True,
            available=installed_decision is not None,
            decision=installed_decision,
        )
        self._runtime_probe_cache.pop(self._runtime_cache_key(runtime_path), None)
        pending_provider = str(self._pending_gpu_provider_switch or '').strip().lower()
        pending_snapshot = self._pending_gpu_provider_snapshot
        self._clear_pending_gpu_provider_switch()
        activated_pending_switch = pending_provider == provider and pending_snapshot is not None
        if activated_pending_switch:
            if not self._activate_gpu_provider_switch(
                provider,
                runtime_path,
                pending_snapshot,
                enable_gpu=pending_device_enable,
            ):
                self._show_error('Runtime pack activation failed', f'{provider_label} was installed but could not be activated.')
                return
        else:
            self.config.gpu_runtime_enabled, self.config.gpu_runtime_path = runtime_service.set_gpu_runtime(True, runtime_path)
            if pending_device_enable:
                self.config.device_preference = runtime_service.set_device_preference('gpu')
                self._activate_performance_mode_for_device('gpu', reset_session=True)
                self._clear_gpu_runtime_validation_state()
                self._show_next_gpu_validation_failure = True
        restart_required = runtime_service.is_runtime_restart_required()
        self.window.set_progress(100, 'Downloaded')
        self.window.set_console_text(f'{provider_label} runtime pack installed:\n{runtime_path}')
        if not activated_pending_switch:
            self._reset_cuda_session_validation()
        self._restart_prompt_pending = False
        if not activated_pending_switch:
            self._apply_state_transaction(
                refresh_first=False,
                report_save_errors=False,
                start_runtime_probe=not restart_required,
                apply_controls=True,
            )
            if restart_required:
                self._request_runtime_backend_restart(provider)

    def _on_runtime_pack_download_error(self, message: str) -> None:
        self._pending_gpu_device_enable = False
        self._clear_pending_gpu_provider_switch()
        self._refresh_settings_values()
        self._restart_prompt_pending = False
        self._restart_after_gpu_setup = False
        self._show_error('Runtime pack download failed', message)

    def _on_runtime_pack_download_stopped(self) -> None:
        self._pending_gpu_device_enable = False
        self._clear_pending_gpu_provider_switch()
        self._refresh_settings_values()
        self._restart_prompt_pending = False
        self._restart_after_gpu_setup = False
        self.window.set_progress(0, 'Stopped')

    def _on_runtime_pack_download_finished(self) -> None:
        self.runtime_pack_download_in_progress = False
        self._runtime_pack_thread = None
        self._runtime_pack_worker = None
        self._clear_pending_gpu_provider_switch()
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
        if runtime_service.is_runtime_restart_required():
            self._apply_state_transaction(refresh_first=False, report_save_errors=False)
            self._request_runtime_backend_restart(self.config.gpu_provider_preference)
            return False
        model_path = model_service.get_existing_model_path()
        if model_path is None:
            self._show_warning('Model missing', 'The Transcription Model package is missing. Please download it first.')
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
        active_device = resource_service.normalize_performance_device(self.config.device_preference)
        active_performance_mode = self._activate_performance_mode_for_device(active_device)
        self.transcription_in_progress = True
        self._apply_controls_state()
        self.window.set_progress(0, 'Transcribing 0.00% | ETA --')
        performance_name = resource_service.performance_display_name(active_performance_mode)
        gpu_memory_text = (
            f'GPU memory usage: {resource_service.gpu_memory_level_name(runtime_service.get_gpu_batch_size(), self.config.gpu_memory_max_batch)} '
            f'({runtime_service.get_gpu_batch_size()} sections together)\n'
            if active_device == 'gpu'
            else ''
        )
        pedal_text = 'Included' if self.config.engine_pedals_enabled else 'Excluded'
        dynamics_text = 'Expressive' if self.config.engine_velocity_mode == 'expressive' else f'Uniform ({self.config.engine_uniform_velocity})'
        self.window.set_console_text(
            'Starting conversion...\n'
            'Engine: A2M Piano Engine\n'
            'Runtime: ONNX Runtime\n'
            f'CPU usage: {performance_name}\n'
            f'{gpu_memory_text}'
            f'Pedal events: {pedal_text}\n'
            f'Dynamics: {dynamics_text}\n'
            'Please wait, processing...'
        )
        worker = ConversionWorker(
            self.selected_file,
            include_pedals=self.config.engine_pedals_enabled,
            velocity_mode=self.config.engine_velocity_mode,
            uniform_velocity=self.config.engine_uniform_velocity,
        )
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
        self._revert_to_cpu_preference(start_runtime_probe=True, report_save_errors=False)

    def _show_gpu_fallback_message(self, reason: str='') -> None:
        detail = str(reason or '').strip()
        suffix = f'\n\nReason: {detail}' if detail else ''
        self._show_info('GPU unavailable', f'GPU acceleration is unavailable right now.\nA2M is falling back to CPU mode automatically.{suffix}')

    def _revert_to_cpu_preference(self, *, start_runtime_probe: bool=False, report_save_errors: bool=False) -> None:
        self.config.device_preference = runtime_service.set_device_preference('cpu')
        self._activate_performance_mode_for_device('cpu', reset_session=True)
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

    @staticmethod
    def _is_gpu_cpu_fallback_error(message: str) -> bool:
        normalized = str(message or '').strip().lower()
        fallback_markers = (
            'gpu mode selected, but onnx session started on cpu',
            'gpu mode selected, but no gpu provider is available',
            'gpu mode selected, but cuda provider is not available',
            'gpu mode selected, but directml provider is not available',
            'gpu mode selected, but onnx gpu session could not be created',
        )
        return any(marker in normalized for marker in fallback_markers)

    def _handle_gpu_cpu_fallback_error(self, message: str) -> bool:
        if not self._is_gpu_cpu_fallback_error(message):
            return False
        self._switch_to_cpu_mode()
        self._show_gpu_fallback_message(message)
        return True

    def _should_prompt_cuda_toolkit_install(self, decision: RuntimeDecision) -> bool:
        if decision.device_mode == 'gpu':
            return False
        reason_code = str(decision.reason_code or '').strip()
        if reason_code not in {REASON_PROVIDER_LOAD_FAILED, REASON_SESSION_CREATION_FAILED, REASON_PROVIDER_NOT_EXPOSED}:
            return False
        return self._target_gpu_provider_for_active_runtime() == 'cuda'

    def _on_device_preference_changed(self, preference: str) -> None:
        if self.runtime_pack_download_in_progress:
            return
        pref = str(preference or 'cpu').strip().lower()
        if pref not in {'cpu', 'gpu'}:
            pref = 'cpu'
        if pref == 'gpu':
            self._pending_gpu_device_enable = False
            preferred_provider = self._resolved_gpu_provider()
            alternate_provider = 'cuda' if preferred_provider == 'dml' else 'dml'
            provider = ''
            runtime_path = ''
            for candidate in (preferred_provider, alternate_provider):
                candidate_path = self._known_available_runtime_path(candidate)
                if candidate_path:
                    provider = candidate
                    runtime_path = candidate_path
                    break
            if not runtime_path:
                provider = self._choose_gpu_provider_for_setup()
                if not provider:
                    self._refresh_settings_values()
                    return
                status = self._provider_runtime_status(provider)
                if status.runtime_installed and status.runtime_path:
                    self._explain_unavailable_provider(status)
                    self._refresh_settings_values()
                    return
                snapshot = self._capture_gpu_provider_state()
                self._pending_gpu_device_enable = True
                self._pending_gpu_provider_switch = provider
                self._pending_gpu_provider_snapshot = snapshot
                if not runtime_pack_service.provider_pack_url(provider):
                    self._show_warning(
                        'Runtime pack unavailable',
                        f'{runtime_pack_service.provider_display_name(provider)} runtime pack URL is not configured.',
                    )
                    self._pending_gpu_device_enable = False
                    self._clear_pending_gpu_provider_switch()
                else:
                    self._start_runtime_pack_download(provider)
                self._refresh_settings_values()
                return
            current_provider = self._resolved_gpu_provider()
            current_path = self._runtime_cache_key(runtime_service.get_effective_runtime_root_for_gpu_checks())
            if provider != current_provider or self._runtime_cache_key(runtime_path) != current_path:
                snapshot = self._capture_gpu_provider_state()
                if not self._activate_gpu_provider_switch(
                    provider,
                    runtime_path,
                    snapshot,
                    enable_gpu=True,
                ):
                    self._refresh_settings_values()
                    self._show_warning('GPU unavailable', 'The available GPU runtime could not be activated.')
                return
            self.config.device_preference = runtime_service.set_device_preference('gpu')
            self._activate_performance_mode_for_device('gpu', reset_session=True)
            self._clear_gpu_runtime_validation_state()
            self._show_next_gpu_validation_failure = False
            self._apply_state_transaction(refresh_first=False, start_runtime_probe=True, apply_controls=True)
            return
        else:
            self._show_next_gpu_validation_failure = False
            self.config.device_preference = runtime_service.set_device_preference(pref)
            if pref != 'gpu':
                self._clear_gpu_runtime_validation_state()
            self._activate_performance_mode_for_device('cpu', reset_session=True)
        self._apply_state_transaction(refresh_first=False, start_runtime_probe=True, apply_controls=True)

    def _on_gpu_provider_preference_changed(self, preference: str) -> None:
        normalized = runtime_pack_service.resolve_provider_for_preference(preference)
        if normalized == self.config.gpu_provider_preference:
            self._refresh_settings_values()
            return
        if (
            self.transcription_in_progress
            or self.runtime_pack_download_in_progress
            or self.cudnn_install_in_progress
            or self._gpu_validation_in_progress
            or self._runtime_checking
        ):
            self._refresh_settings_values()
            return
        snapshot = self._capture_gpu_provider_state()
        status = self._provider_runtime_status(normalized)
        runtime_path = self._known_available_runtime_path(normalized)
        if not runtime_path:
            if status.runtime_installed and status.runtime_path:
                self._explain_unavailable_provider(status)
                self._refresh_settings_values()
                return
            self._pending_gpu_provider_switch = normalized
            self._pending_gpu_provider_snapshot = snapshot
            if not self._prompt_runtime_pack_install(normalized):
                self._clear_pending_gpu_provider_switch()
            self._refresh_settings_values()
            return
        if not self._activate_gpu_provider_switch(normalized, runtime_path, snapshot):
            self._refresh_settings_values()
            self._show_warning(
                'GPU provider unavailable',
                f'{runtime_pack_service.provider_display_name(normalized)} could not be activated. The current provider was not changed.',
            )

    def _on_engine_pedals_enabled_changed(self, enabled: bool) -> None:
        if self.transcription_in_progress:
            self._refresh_settings_values()
            return
        self.config.engine_pedals_enabled = bool(enabled)
        self._apply_state_transaction(refresh_first=False)

    def _on_engine_velocity_mode_changed(self, mode: str) -> None:
        if self.transcription_in_progress:
            self._refresh_settings_values()
            return
        normalized = 'uniform' if str(mode).strip().lower() == 'uniform' else 'expressive'
        self.config.engine_velocity_mode = normalized
        self._apply_state_transaction(refresh_first=False)

    def _on_engine_uniform_velocity_changed(self, value: int) -> None:
        if self.transcription_in_progress:
            self._refresh_settings_values()
            return
        self.config.engine_uniform_velocity = max(
            ENGINE_UNIFORM_VELOCITY_MIN,
            min(ENGINE_UNIFORM_VELOCITY_MAX, int(value)),
        )
        self._apply_state_transaction(refresh_first=False)

    def _on_transcription_performance_mode_changed(self, device: str, mode: str) -> None:
        if self.transcription_in_progress:
            self._refresh_settings_values()
            return
        selected_device = resource_service.normalize_performance_device(device)
        normalized = resource_service.normalize_performance_mode(mode)
        if not self._confirm_transcription_performance_mode(selected_device, normalized):
            self._refresh_settings_values()
            return
        if normalized == self._performance_mode_for_device(selected_device):
            self._refresh_settings_values()
            return
        self._set_performance_mode_for_device(selected_device, normalized)
        if selected_device == self.config.device_preference:
            self._activate_performance_mode_for_device(selected_device, reset_session=True)
        self._apply_state_transaction(refresh_first=False)

    def _confirm_transcription_performance_mode(self, device: str, mode: str) -> bool:
        selected_device = resource_service.normalize_performance_device(device)
        normalized = resource_service.normalize_performance_mode(mode)
        display_name = resource_service.performance_display_name(normalized)
        dialog = QDialog(self.window)
        dialog.setModal(True)
        dialog.setWindowTitle('Choose CPU usage')
        dialog.setMinimumWidth(420)
        icon = self.window.windowIcon()
        if not icon.isNull():
            dialog.setWindowIcon(icon)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        title = QLabel(f'Set CPU usage to {display_name}?', dialog)
        title.setStyleSheet('font: 700 12pt "Segoe UI";')
        layout.addWidget(title)

        simple_explanation = QLabel(resource_service.performance_description(normalized, selected_device), dialog)
        simple_explanation.setWordWrap(True)
        layout.addWidget(simple_explanation)

        technical_explanation = QLabel(
            resource_service.performance_resource_description(normalized, device=selected_device),
            dialog,
        )
        technical_explanation.setWordWrap(True)
        technical_explanation.setObjectName('muted')
        layout.addWidget(technical_explanation)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = QPushButton('Cancel', dialog)
        apply_button = QPushButton('Apply', dialog)
        apply_button.setDefault(True)
        apply_button.setAutoDefault(True)
        buttons.addWidget(cancel_button)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        cancel_button.clicked.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)

        self._apply_dialog_theme(dialog)
        technical_explanation.setStyleSheet(f'color: {self.window.theme.text_secondary};')
        return self._exec_dialog(dialog) == QDialog.Accepted

    def _on_gpu_memory_usage_changed(self, batch_size: int) -> None:
        if self.transcription_in_progress or self.config.device_preference != 'gpu':
            self._refresh_settings_values()
            return
        normalized = resource_service.normalize_gpu_batch_size(batch_size)
        if not self._confirm_gpu_memory_usage(normalized):
            self._refresh_settings_values()
            return
        self._apply_gpu_memory_level(normalized)

    def _confirm_gpu_memory_usage(self, batch_size: int) -> bool:
        normalized = resource_service.normalize_gpu_batch_size(batch_size)
        level_name = resource_service.gpu_memory_level_name(normalized, self.config.gpu_memory_max_batch)
        dialog = QDialog(self.window)
        dialog.setModal(True)
        dialog.setWindowTitle('Choose GPU memory usage')
        dialog.setMinimumWidth(420)
        icon = self.window.windowIcon()
        if not icon.isNull():
            dialog.setWindowIcon(icon)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)
        title = QLabel(f'Set GPU memory usage to {level_name}?', dialog)
        title.setStyleSheet('font: 700 12pt "Segoe UI";')
        layout.addWidget(title)
        simple_explanation = QLabel(resource_service.gpu_memory_description(normalized), dialog)
        simple_explanation.setWordWrap(True)
        layout.addWidget(simple_explanation)
        technical_explanation = QLabel(resource_service.gpu_memory_resource_description(normalized), dialog)
        technical_explanation.setWordWrap(True)
        layout.addWidget(technical_explanation)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = QPushButton('Cancel', dialog)
        apply_button = QPushButton('Apply', dialog)
        apply_button.setDefault(True)
        apply_button.setAutoDefault(True)
        buttons.addWidget(cancel_button)
        buttons.addWidget(apply_button)
        layout.addLayout(buttons)
        cancel_button.clicked.connect(dialog.reject)
        apply_button.clicked.connect(dialog.accept)
        self._apply_dialog_theme(dialog)
        technical_explanation.setStyleSheet(f'color: {self.window.theme.text_secondary};')
        return self._exec_dialog(dialog) == QDialog.Accepted

    def _on_reset_engine_settings_requested(self) -> None:
        if self.transcription_in_progress:
            self._show_info('Reset blocked', 'Stop transcription before resetting Piano Engine settings.')
            return
        answer = self._ask_question(
            'Restore Piano Engine defaults',
            'Restore Piano Engine output and resource settings to the recommended defaults for this computer?',
            default_button=QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.config.engine_pedals_enabled = True
        self.config.engine_velocity_mode = 'expressive'
        self.config.engine_uniform_velocity = ENGINE_UNIFORM_VELOCITY_DEFAULT
        self.config.cpu_performance_mode = resource_service.recommended_performance_mode()
        self.config.gpu_performance_mode = 'balanced'
        self.config.gpu_batch_size = resource_service.gpu_memory_presets(
            self.config.gpu_memory_max_batch
        )['balanced']
        self.config.hardware_defaults_initialized = True
        self._activate_performance_mode_for_device(reset_session=True)
        self._apply_state_transaction(refresh_first=False)
        self._show_info('Piano Engine reset', 'Piano Engine settings were restored to the recommended defaults.')

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
            or self._runtime_checking
            or self._gpu_validation_in_progress
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
        detected_memory_max = resource_service.normalize_gpu_memory_max_batch(
            self.config.gpu_memory_max_batch
        )
        defaults = default_config()
        defaults.cpu_performance_mode = resource_service.recommended_performance_mode()
        defaults.gpu_performance_mode = 'balanced'
        defaults.gpu_memory_max_batch = detected_memory_max
        defaults.gpu_batch_size = resource_service.gpu_memory_presets(detected_memory_max)['balanced']
        defaults.hardware_defaults_initialized = True
        try:
            defaults.window_geometry = self.window.saveGeometry().toBase64().data().decode('ascii')
        except Exception:
            defaults.window_geometry = self.config.window_geometry
        self.config = defaults
        self._pending_gpu_device_enable = False
        self._clear_pending_gpu_provider_switch()
        self._clear_gpu_provider_switch_rollback()
        self.config.device_preference = runtime_service.set_device_preference(self.config.device_preference)
        self.config.gpu_provider_preference = runtime_service.set_gpu_provider_preference(self.config.gpu_provider_preference)
        self.config.cpu_performance_mode = resource_service.normalize_performance_mode(self.config.cpu_performance_mode)
        self.config.gpu_performance_mode = resource_service.normalize_performance_mode(self.config.gpu_performance_mode)
        self._activate_performance_mode_for_device()
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
        restart_required = runtime_service.is_runtime_restart_required()
        self._apply_state_transaction(
            refresh_first=False,
            start_runtime_probe=not restart_required,
            apply_controls=True,
        )
        if restart_required:
            self._request_runtime_backend_restart(self.config.gpu_provider_preference)
        else:
            self._show_info('Settings reset', 'All settings were reset to defaults.')

    def check_for_updates(self, *, manual: bool) -> None:
        self._trigger_update_check(bool(manual))

    def _trigger_update_check(self, manual: bool) -> None:
        if self.update_check_in_progress:
            if manual:
                self._show_info('Update check', 'An update check is already in progress.')
            return
        if self._update_worker is not None or self._update_install_worker is not None:
            if manual:
                self._show_info('Update check', 'An update operation is already in progress.')
            return
        self.update_check_in_progress = True
        self._update_check_manual = bool(manual)
        worker = UpdateCheckWorker()
        thread = self._create_worker_thread(worker)
        worker.finishedSuccess.connect(self._on_update_check_success, Qt.QueuedConnection)
        worker.finishedStopped.connect(self._on_update_check_stopped, Qt.QueuedConnection)
        worker.errorRaised.connect(self._on_update_check_error, Qt.QueuedConnection)
        thread.finished.connect(self._on_update_check_worker_finished)
        self._update_thread = thread
        self._update_worker = worker
        thread.start()

    def _trigger_update_install(self, payload: dict[str, object]) -> None:
        if self._update_install_worker is not None:
            return
        if self._restart_after_gpu_setup:
            self._show_info(
                'Restart required',
                'A2M must restart to finish applying the GPU runtime before installing an update.',
            )
            self._try_restart_after_gpu_setup()
            return
        if (
            self.transcription_in_progress
            or self.model_download_in_progress
            or self.runtime_pack_download_in_progress
            or self.cudnn_install_in_progress
        ):
            self._show_info('Update install blocked', 'Stop active operations before installing updates.')
            return
        self._update_install_fallback_url = str(payload.get('url') or payload.get('download_url') or OFFICIAL_PAGE_URL)
        self.update_check_in_progress = True
        self._apply_controls_state()
        self.window.set_console_text('Preparing update package...\nPlease wait...')
        self.window.set_progress(0, 'Preparing update...')
        self._show_update_progress_dialog()
        worker = UpdateInstallWorker(payload=dict(payload))
        thread = self._create_worker_thread(worker)
        worker.progressChanged.connect(self._on_update_install_progress, Qt.QueuedConnection)
        worker.handoffRequested.connect(self._on_update_install_handoff_requested, Qt.QueuedConnection)
        worker.finishedSuccess.connect(self._on_update_install_success, Qt.QueuedConnection)
        worker.finishedStopped.connect(self._on_update_install_stopped, Qt.QueuedConnection)
        worker.errorRaised.connect(self._on_update_install_error, Qt.QueuedConnection)
        thread.finished.connect(self._on_update_install_worker_finished)
        self._update_install_thread = thread
        self._update_install_worker = worker
        thread.start()

    def _on_update_check_success(self, payload: object) -> None:
        self._finish_update_check(bool(self._update_check_manual), payload)

    def _on_update_check_stopped(self) -> None:
        self._finish_update_check(bool(self._update_check_manual), {'status': 'stopped'})

    def _on_update_check_error(self, msg: str) -> None:
        self._finish_update_check(
            bool(self._update_check_manual),
            {'status': 'error', 'error': str(msg or 'Unknown error')},
        )

    def _on_update_check_worker_finished(self) -> None:
        self._update_thread = None
        self._update_worker = None
        self.update_check_in_progress = self._update_install_worker is not None
        self._close_if_update_cancel_requested()
        self._try_restart_after_gpu_setup()

    def _on_update_install_worker_finished(self) -> None:
        self._update_install_thread = None
        self._update_install_worker = None
        self.update_check_in_progress = self._update_worker is not None
        self._apply_controls_state()
        if self._update_worker is None:
            self._close_update_progress_dialog()
        if self._update_worker is None:
            self._update_install_fallback_url = ''
        self._close_if_update_cancel_requested()
        self._try_restart_after_gpu_setup()

    def _on_update_install_success(self, payload: object) -> None:
        self._finish_update_install(payload)

    def _on_update_install_stopped(self) -> None:
        self._finish_update_install({'status': 'canceled'})

    def _on_update_install_error(self, msg: str) -> None:
        self._finish_update_install({'status': 'error', 'error': str(msg or 'Unknown error')})

    def _on_update_install_handoff_requested(self, payload: object) -> None:
        info = payload if isinstance(payload, dict) else {}
        version = str(info.get('version') or '')
        requires_elevation = bool(info.get('requires_elevation', False))
        self._update_update_progress_dialog(99, 'Ready to hand off to installer. Waiting for your confirmation...')
        dialog = self._update_progress_dialog
        if dialog is not None:
            dialog.setCancelButton(None)
        continue_update, restart_after = self._ask_update_handoff(
            version=version,
            default_restart=True,
            requires_elevation=requires_elevation,
        )
        worker = self._update_install_worker
        if worker is None:
            return
        worker.set_handoff_decision(
            continue_update=bool(continue_update),
            restart_after_update=bool(restart_after),
        )

    def _finish_update_check(self, manual: bool, result: object) -> None:
        payload = result if isinstance(result, dict) else {}
        status = str(payload.get('status', 'error') or 'error')
        if status == 'stopped':
            return
        if status == 'available':
            latest = str(payload.get('latest', '') or '')
            download_url = str(payload.get('url', '') or OFFICIAL_PAGE_URL)
            setup_url = str(payload.get('setup_url', '') or '')
            setup_sha256 = str(payload.get('setup_sha256', '') or '')
            try:
                setup_size = max(0, int(payload.get('setup_size', 0) or 0))
            except Exception:
                setup_size = 0
            can_install = bool(payload.get('install_supported', False)) and bool(setup_url) and bool(setup_sha256) and setup_size > 0
            requires_manual_update = bool(payload.get('requires_manual_update', False))
            if not getattr(sys, 'frozen', False):
                can_install = False
            if requires_manual_update:
                can_install = False
            notes = [str(item or '').strip() for item in (payload.get('notes') or []) if str(item or '').strip()]
            notes_text = ''
            if notes:
                notes_text = '\n\nWhat\'s new:\n' + '\n'.join((f'- {line}' for line in notes[:8]))
            if requires_manual_update:
                minimum_supported = str(payload.get('minimum_supported_version') or '1.0.0').strip() or '1.0.0'
                notes_text += (
                    '\n\nYour current version is below the minimum supported '
                    f'auto-update baseline ({minimum_supported}).'
                )
            proceed, auto_install = self._ask_update_install_preference(
                latest_version=latest,
                details_text=notes_text,
                install_supported=can_install,
            )
            if proceed:
                if can_install and auto_install:
                    QTimer.singleShot(0, lambda install_payload=dict(payload): self._trigger_update_install(install_payload))
                else:
                    webbrowser.open(download_url)
            return
        if status == 'up_to_date':
            if manual:
                self._show_info('Update check', 'You are already on the latest version.')
            return
        if manual:
            message = str(payload.get('error', 'Unknown error') or 'Unknown error')
            self._show_warning('Update check failed', f'Could not check for updates:\n{message}')

    def _finish_update_install(self, payload: object) -> None:
        result = payload if isinstance(payload, dict) else {}
        status = str(result.get('status', 'error') or 'error')
        if status == 'canceled':
            self._close_update_progress_dialog()
            if self.close_after_update_operation:
                return
            self._show_info('Update canceled', 'Update was canceled before installation started.')
            return
        if status == 'aborted':
            self._close_update_progress_dialog()
            self._show_info('Update canceled', 'Update was aborted. No installer was launched.')
            return
        if status == 'ready':
            self._update_installer_handoff_in_progress = True
            version = str(result.get('version') or '').strip() or 'latest'
            if bool(result.get('restart_after_update', True)):
                self.window.set_progress(100, f'Handing off to installer for v{version} (app will restart after update).')
            else:
                self.window.set_progress(100, f'Handing off to installer for v{version}...')
            self.window.set_console_text('Installer handoff in progress...\nClosing A2M to continue update.')
            QTimer.singleShot(250, self.window.close)
            return
        self._close_update_progress_dialog()
        message = str(result.get('error', 'Unknown error') or 'Unknown error')
        self._show_warning('Update install failed', f'Could not install update:\n{message}')
        fallback_url = str(self._update_install_fallback_url or '').strip()
        if not fallback_url:
            return
        answer = self._ask_question(
            'Update install',
            'Would you like to open the download page instead?',
            default_button=QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            webbrowser.open(fallback_url)
        except Exception as exc:
            self._show_warning('Update install', f'Could not open update page:\n{exc}')

    def _cancel_update_check(self) -> bool:
        for worker in (self._update_worker, self._update_install_worker):
            if worker is None:
                continue
            try:
                worker.stop()
            except Exception:
                pass
        any_running = False
        for thread in (self._update_thread, self._update_install_thread):
            if thread is None:
                continue
            if thread.isRunning():
                thread.quit()
                any_running = True
        if any_running:
            self.update_check_in_progress = True
            return False
        self._update_thread = None
        self._update_worker = None
        self._update_install_thread = None
        self._update_install_worker = None
        self.update_check_in_progress = False
        self._close_update_progress_dialog()
        self._apply_controls_state()
        return True

    def _close_if_update_cancel_requested(self) -> None:
        if not self.close_after_update_operation:
            return
        if self._update_worker is not None or self._update_install_worker is not None:
            return
        self.close_after_update_operation = False
        self.window.close()

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
        if (
            self.close_after_transcription
            or self.close_after_model_download
            or self.close_after_runtime_pack_download
            or self.close_after_cudnn_install
            or self.close_after_update_operation
        ):
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
            self.close_after_update_operation = True
            self.window.set_console_text('Stopping update operation...\nPlease wait...')
            if not self._cancel_update_check():
                return False
            self.close_after_update_operation = False
        self._persist_config()
        return True
