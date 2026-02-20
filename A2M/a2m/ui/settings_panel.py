from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QSizePolicy, QVBoxLayout, QWidget
from a2m.core.config_service import normalize_conversion_method
from a2m.core.config import APP_VERSION, GPU_BATCH_SIZE_MAX, GPU_BATCH_SIZE_MIN, UI_SCALE_PERCENT_MAX, UI_SCALE_PERCENT_MIN
from a2m.core.config import MODERN_FRAME_THRESHOLD_DEFAULT, MODERN_FRAME_THRESHOLD_MAX, MODERN_FRAME_THRESHOLD_MIN
from a2m.core.config import MODERN_OFFSET_THRESHOLD_DEFAULT, MODERN_OFFSET_THRESHOLD_MAX, MODERN_OFFSET_THRESHOLD_MIN
from a2m.core.config import MODERN_ONSET_THRESHOLD_DEFAULT, MODERN_ONSET_THRESHOLD_MAX, MODERN_ONSET_THRESHOLD_MIN
from a2m.core.config import MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT, MODERN_PEDAL_OFFSET_THRESHOLD_MAX, MODERN_PEDAL_OFFSET_THRESHOLD_MIN
from a2m.core.messages import RUNTIME_LABEL_CHECKING, RUNTIME_LABEL_CPU_NO_GPU_PACK, RUNTIME_LABEL_CPU_WITH_GPU_PACK
from a2m.core.messages import RUNTIME_LABEL_GPU_ACTIVE, RUNTIME_LABEL_GPU_NO_PACK, RUNTIME_LABEL_GPU_UNAVAILABLE, RUNTIME_LABEL_GPU_VALIDATING
from a2m.core.messages import RUNTIME_LABEL_MISSING
from .interaction import sync_pointer_cursor
from .theme import ThemePalette
from .widgets.custom_controls import RoundHandleSliderStyle, SquareCheckBoxStyle

class SettingsPanel(QFrame):
    devicePreferenceChanged = Signal(str)
    conversionMethodChanged = Signal(str)
    modernAdaptiveThresholdsChanged = Signal(bool)
    modernInputNormalizationChanged = Signal(bool)
    modernSmartOverlapStitchingChanged = Signal(bool)
    modernAutoCalibrationChanged = Signal(bool)
    modernOnsetThresholdChanged = Signal(float)
    modernOffsetThresholdChanged = Signal(float)
    modernFrameThresholdChanged = Signal(float)
    modernPedalOffsetThresholdChanged = Signal(float)
    modernCleanupScaleChanged = Signal(float)
    modernPedalClusterScaleChanged = Signal(float)
    modernAlignmentGateScaleChanged = Signal(float)
    gpuProviderPreferenceChanged = Signal(str)
    gpuBatchSizeChanged = Signal(int)
    uiScalePercentChanged = Signal(int)
    downloadLocationChanged = Signal(str)
    autoCheckUpdatesChanged = Signal(bool)
    checkUpdatesNowRequested = Signal()
    resetSettingsRequested = Signal()
    _UI_SCALE_STEP = 5
    _RUNTIME_LABEL_LINES = 2
    _SETTINGS_ACTION_BUTTON_HEIGHT = 32
    _MODERN_THRESHOLD_SCALE = 1000
    _MODERN_SCALE_MIN = 0.6
    _MODERN_SCALE_MAX = 2.0

    def __init__(self, theme: ThemePalette, parent: QWidget | None=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setObjectName('settingsBody')
        self.setMinimumWidth(0)
        self._updating = False
        self._runtime_checking = True
        self._runtime_available = False
        self._cuda_available = False
        self._dml_available = False
        self._active_provider = 'CPU'
        self._selected_device = 'cpu'
        self._conversion_method = 'legacy_v1'
        self._modern_adaptive_thresholds_enabled = True
        self._modern_input_normalization_enabled = True
        self._modern_smart_overlap_stitching_enabled = True
        self._modern_auto_calibration_enabled = True
        self._modern_cleanup_scale = 1.0
        self._modern_pedal_cluster_scale = 1.0
        self._modern_alignment_gate_scale = 1.0
        self._modern_manual_onset_threshold = float(MODERN_ONSET_THRESHOLD_DEFAULT)
        self._modern_manual_offset_threshold = float(MODERN_OFFSET_THRESHOLD_DEFAULT)
        self._modern_manual_frame_threshold = float(MODERN_FRAME_THRESHOLD_DEFAULT)
        self._modern_manual_pedal_offset_threshold = float(MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT)
        self._pending_modern_onset_threshold: float | None = None
        self._pending_modern_offset_threshold: float | None = None
        self._pending_modern_frame_threshold: float | None = None
        self._pending_modern_pedal_offset_threshold: float | None = None
        self._pending_modern_cleanup_scale: float | None = None
        self._pending_modern_pedal_cluster_scale: float | None = None
        self._pending_modern_alignment_gate_scale: float | None = None
        self._gpu_provider_preference = 'auto'
        self._pending_ui_scale_percent: int | None = None
        self._slider_style: RoundHandleSliderStyle | None = None
        self._ui_scale_slider_style: RoundHandleSliderStyle | None = None
        self._checkbox_style: SquareCheckBoxStyle | None = None
        self._modern_checkbox_styles: list[SquareCheckBoxStyle] = []
        self._modern_slider_styles: list[RoundHandleSliderStyle] = []
        self._settings_card_layouts: list[QVBoxLayout] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(7)
        self._root_layout = layout
        title = QLabel('Settings', self)
        title.setObjectName('title')
        layout.addWidget(title)
        self._build_device_card()
        self._build_conversion_engine_card()
        self._build_performance_card()
        self._build_interface_card()
        self._build_downloads_card()
        self._build_updates_card()
        self._set_modern_feature_toggles_ui(adaptive_thresholds_enabled=True, input_normalization_enabled=True, smart_overlap_stitching_enabled=True, auto_calibration_enabled=True)
        self._set_modern_manual_thresholds_ui(
            onset_threshold=MODERN_ONSET_THRESHOLD_DEFAULT,
            offset_threshold=MODERN_OFFSET_THRESHOLD_DEFAULT,
            frame_threshold=MODERN_FRAME_THRESHOLD_DEFAULT,
            pedal_offset_threshold=MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT,
        )
        self._set_modern_calibration_scales_ui(
            cleanup_scale=1.0,
            pedal_cluster_scale=1.0,
            alignment_gate_scale=1.0,
        )
        self._refresh_conversion_feature_toggles_ui()
        layout.addStretch(1)
        self._install_control_styles()
        self._refresh_runtime_ui()
        self._apply_interaction_cursors()
        self.set_scale(1.0)

    def _build_device_card(self) -> None:
        device_card, device_layout = self._create_settings_card('Device')
        device_label = QLabel('Device mode', device_card)
        device_label.setObjectName('settingsSubtext')
        device_layout.addWidget(device_label)
        mode_holder = QFrame(device_card)
        mode_holder.setObjectName('modeHolder')
        mode_holder_layout = QHBoxLayout(mode_holder)
        mode_holder_layout.setContentsMargins(2, 2, 2, 2)
        mode_holder_layout.setSpacing(2)
        self._mode_holder_layout = mode_holder_layout
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.cpu_button = QPushButton('CPU', mode_holder)
        self.gpu_button = QPushButton('GPU', mode_holder)
        for button, mode in ((self.cpu_button, 'cpu'), (self.gpu_button, 'gpu')):
            self._configure_mode_button(button)
            button.clicked.connect(lambda checked, m=mode: self._on_device_clicked(checked, m))
            mode_holder_layout.addWidget(button)
            self.mode_group.addButton(button)
        self.cpu_button.setChecked(True)
        device_layout.addWidget(mode_holder)
        provider_label = QLabel('GPU provider', device_card)
        provider_label.setObjectName('settingsSubtext')
        device_layout.addWidget(provider_label)
        provider_holder = QFrame(device_card)
        provider_holder.setObjectName('modeHolder')
        provider_holder_layout = QHBoxLayout(provider_holder)
        provider_holder_layout.setContentsMargins(2, 2, 2, 2)
        provider_holder_layout.setSpacing(2)
        self._provider_holder_layout = provider_holder_layout
        self.provider_group = QButtonGroup(self)
        self.provider_group.setExclusive(True)
        self.auto_provider_button = QPushButton('Auto', provider_holder)
        self.cuda_provider_button = QPushButton('CUDA', provider_holder)
        self.dml_provider_button = QPushButton('DirectML', provider_holder)
        provider_buttons = (
            (self.auto_provider_button, 'auto'),
            (self.cuda_provider_button, 'cuda'),
            (self.dml_provider_button, 'dml'),
        )
        for button, provider in provider_buttons:
            self._configure_mode_button(button)
            button.clicked.connect(lambda checked, pref=provider: self._on_provider_clicked(checked, pref))
            provider_holder_layout.addWidget(button)
            self.provider_group.addButton(button)
        self.auto_provider_button.setChecked(True)
        device_layout.addWidget(provider_holder)
        self.runtime_label = QLabel(RUNTIME_LABEL_CHECKING, device_card)
        self.runtime_label.setObjectName('muted')
        self.runtime_label.setWordWrap(True)
        self.runtime_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.runtime_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        device_layout.addWidget(self.runtime_label)

    def _build_performance_card(self) -> None:
        performance_card, performance_layout = self._create_settings_card('Performance')
        batch_row = QHBoxLayout()
        self.batch_label = QLabel('GPU batch size', performance_card)
        self.batch_label.setObjectName('settingsSubtext')
        self.batch_value = QLabel('4', performance_card)
        self.batch_value.setObjectName('muted')
        self.batch_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.batch_value.setMinimumWidth(44)
        batch_row.addWidget(self.batch_label)
        batch_row.addStretch(1)
        batch_row.addWidget(self.batch_value)
        performance_layout.addLayout(batch_row)
        self.batch_slider = QSlider(Qt.Horizontal, performance_card)
        self.batch_slider.setRange(GPU_BATCH_SIZE_MIN, GPU_BATCH_SIZE_MAX)
        self.batch_slider.setValue(4)
        self.batch_slider.valueChanged.connect(self._on_batch_changed)
        performance_layout.addWidget(self.batch_slider)

    def _build_conversion_engine_card(self) -> None:
        method_card, method_layout = self._create_settings_card('Transcription Engine')
        method_label = QLabel('Conversion method', method_card)
        method_label.setObjectName('settingsSubtext')
        method_layout.addWidget(method_label)
        method_holder = QFrame(method_card)
        method_holder.setObjectName('modeHolder')
        method_holder_layout = QVBoxLayout(method_holder)
        method_holder_layout.setContentsMargins(2, 2, 2, 2)
        method_holder_layout.setSpacing(2)
        self._conversion_holder_layout = method_holder_layout
        self.conversion_method_group = QButtonGroup(self)
        self.conversion_method_group.setExclusive(True)
        self.legacy_method_button = QPushButton('Legacy v1.0.0', method_holder)
        self.modern_method_button = QPushButton(f'Modern v{APP_VERSION}', method_holder)
        for button, method in (
            (self.legacy_method_button, 'legacy_v1'),
            (self.modern_method_button, 'modern'),
        ):
            self._configure_mode_button(button)
            button.clicked.connect(lambda checked, m=method: self._on_conversion_method_clicked(checked, m))
            method_holder_layout.addWidget(button)
            self.conversion_method_group.addButton(button)
        self.legacy_method_button.setChecked(True)
        method_layout.addWidget(method_holder)
        self.modern_features_label = QLabel('Modern features', method_card)
        self.modern_features_label.setObjectName('settingsSubtext')
        method_layout.addWidget(self.modern_features_label)
        self.modern_adaptive_thresholds_checkbox = QCheckBox('Adaptive thresholds per file', method_card)
        self.modern_adaptive_thresholds_checkbox.toggled.connect(self._on_modern_adaptive_thresholds_toggled)
        method_layout.addWidget(self.modern_adaptive_thresholds_checkbox)
        self.modern_input_normalization_checkbox = QCheckBox('Input normalization / denoise', method_card)
        self.modern_input_normalization_checkbox.toggled.connect(self._on_modern_input_normalization_toggled)
        method_layout.addWidget(self.modern_input_normalization_checkbox)
        self.modern_smart_overlap_stitching_checkbox = QCheckBox('Smarter overlap stitching', method_card)
        self.modern_smart_overlap_stitching_checkbox.toggled.connect(self._on_modern_smart_overlap_stitching_toggled)
        method_layout.addWidget(self.modern_smart_overlap_stitching_checkbox)
        self.modern_auto_calibration_checkbox = QCheckBox('Smart auto calibration', method_card)
        self.modern_auto_calibration_checkbox.toggled.connect(self._on_modern_auto_calibration_toggled)
        method_layout.addWidget(self.modern_auto_calibration_checkbox)
        self.modern_manual_tuning_label = QLabel('Manual threshold tuning', method_card)
        self.modern_manual_tuning_label.setObjectName('settingsSubtext')
        method_layout.addWidget(self.modern_manual_tuning_label)
        self.modern_manual_tuning_container = QFrame(method_card)
        self.modern_manual_tuning_container.setObjectName('cardContent')
        manual_layout = QVBoxLayout(self.modern_manual_tuning_container)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(6)
        self._modern_manual_layout = manual_layout
        self._build_modern_threshold_slider(
            parent=method_card,
            layout=manual_layout,
            title='Onset threshold',
            slider_attr='modern_onset_slider',
            value_attr='modern_onset_value',
            minimum=MODERN_ONSET_THRESHOLD_MIN,
            maximum=MODERN_ONSET_THRESHOLD_MAX,
            default=MODERN_ONSET_THRESHOLD_DEFAULT,
            on_change=self._on_modern_onset_threshold_changed,
            on_release=self._on_modern_onset_threshold_released,
        )
        self._build_modern_threshold_slider(
            parent=method_card,
            layout=manual_layout,
            title='Offset threshold',
            slider_attr='modern_offset_slider',
            value_attr='modern_offset_value',
            minimum=MODERN_OFFSET_THRESHOLD_MIN,
            maximum=MODERN_OFFSET_THRESHOLD_MAX,
            default=MODERN_OFFSET_THRESHOLD_DEFAULT,
            on_change=self._on_modern_offset_threshold_changed,
            on_release=self._on_modern_offset_threshold_released,
        )
        self._build_modern_threshold_slider(
            parent=method_card,
            layout=manual_layout,
            title='Frame threshold',
            slider_attr='modern_frame_slider',
            value_attr='modern_frame_value',
            minimum=MODERN_FRAME_THRESHOLD_MIN,
            maximum=MODERN_FRAME_THRESHOLD_MAX,
            default=MODERN_FRAME_THRESHOLD_DEFAULT,
            on_change=self._on_modern_frame_threshold_changed,
            on_release=self._on_modern_frame_threshold_released,
        )
        self._build_modern_threshold_slider(
            parent=method_card,
            layout=manual_layout,
            title='Pedal offset threshold',
            slider_attr='modern_pedal_offset_slider',
            value_attr='modern_pedal_offset_value',
            minimum=MODERN_PEDAL_OFFSET_THRESHOLD_MIN,
            maximum=MODERN_PEDAL_OFFSET_THRESHOLD_MAX,
            default=MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT,
            on_change=self._on_modern_pedal_offset_threshold_changed,
            on_release=self._on_modern_pedal_offset_threshold_released,
        )
        self.modern_calibration_tuning_label = QLabel('Manual calibration tuning', method_card)
        self.modern_calibration_tuning_label.setObjectName('settingsSubtext')
        manual_layout.addWidget(self.modern_calibration_tuning_label)
        self._build_modern_scale_slider(
            parent=method_card,
            layout=manual_layout,
            title='Cleanup strength',
            slider_attr='modern_cleanup_scale_slider',
            value_attr='modern_cleanup_scale_value',
            default=1.0,
            on_change=self._on_modern_cleanup_scale_changed,
            on_release=self._on_modern_cleanup_scale_released,
        )
        self._build_modern_scale_slider(
            parent=method_card,
            layout=manual_layout,
            title='Pedal cluster cleanup',
            slider_attr='modern_pedal_cluster_scale_slider',
            value_attr='modern_pedal_cluster_scale_value',
            default=1.0,
            on_change=self._on_modern_pedal_cluster_scale_changed,
            on_release=self._on_modern_pedal_cluster_scale_released,
        )
        self._build_modern_scale_slider(
            parent=method_card,
            layout=manual_layout,
            title='Alignment strictness',
            slider_attr='modern_alignment_gate_scale_slider',
            value_attr='modern_alignment_gate_scale_value',
            default=1.0,
            on_change=self._on_modern_alignment_gate_scale_changed,
            on_release=self._on_modern_alignment_gate_scale_released,
        )
        method_layout.addWidget(self.modern_manual_tuning_container)

    def _build_modern_threshold_slider(self, *, parent: QWidget, layout: QVBoxLayout, title: str, slider_attr: str, value_attr: str, minimum: float, maximum: float, default: float, on_change, on_release=None) -> None:
        row = QHBoxLayout()
        label = QLabel(str(title), parent)
        label.setObjectName('settingsSubtext')
        value_label = QLabel('', parent)
        value_label.setObjectName('muted')
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setMinimumWidth(52)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(value_label)
        layout.addLayout(row)
        slider = QSlider(Qt.Horizontal, parent)
        slider.setRange(
            self._threshold_to_slider_value(float(minimum)),
            self._threshold_to_slider_value(float(maximum)),
        )
        slider.setValue(self._threshold_to_slider_value(float(default)))
        slider.valueChanged.connect(on_change)
        if callable(on_release):
            slider.sliderReleased.connect(on_release)
        layout.addWidget(slider)
        setattr(self, slider_attr, slider)
        setattr(self, value_attr, value_label)

    def _build_modern_scale_slider(self, *, parent: QWidget, layout: QVBoxLayout, title: str, slider_attr: str, value_attr: str, default: float, on_change, on_release=None) -> None:
        row = QHBoxLayout()
        label = QLabel(str(title), parent)
        label.setObjectName('settingsSubtext')
        value_label = QLabel('', parent)
        value_label.setObjectName('muted')
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setMinimumWidth(52)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(value_label)
        layout.addLayout(row)
        slider = QSlider(Qt.Horizontal, parent)
        slider.setRange(
            self._threshold_to_slider_value(float(self._MODERN_SCALE_MIN)),
            self._threshold_to_slider_value(float(self._MODERN_SCALE_MAX)),
        )
        slider.setValue(self._threshold_to_slider_value(self._clamp_scale(default)))
        slider.valueChanged.connect(on_change)
        if callable(on_release):
            slider.sliderReleased.connect(on_release)
        layout.addWidget(slider)
        setattr(self, slider_attr, slider)
        setattr(self, value_attr, value_label)

    def _build_interface_card(self) -> None:
        interface_card, interface_layout = self._create_settings_card('Interface')
        ui_scale_row = QHBoxLayout()
        ui_scale_label = QLabel('UI size', interface_card)
        ui_scale_label.setObjectName('settingsSubtext')
        self.ui_scale_value = QLabel('100%', interface_card)
        self.ui_scale_value.setObjectName('muted')
        self.ui_scale_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.ui_scale_value.setMinimumWidth(44)
        ui_scale_row.addWidget(ui_scale_label)
        ui_scale_row.addStretch(1)
        ui_scale_row.addWidget(self.ui_scale_value)
        interface_layout.addLayout(ui_scale_row)
        self.ui_scale_slider = QSlider(Qt.Horizontal, interface_card)
        self.ui_scale_slider.setRange(UI_SCALE_PERCENT_MIN, UI_SCALE_PERCENT_MAX)
        self.ui_scale_slider.setSingleStep(self._UI_SCALE_STEP)
        self.ui_scale_slider.setPageStep(self._UI_SCALE_STEP)
        self.ui_scale_slider.setTickInterval(self._UI_SCALE_STEP)
        self.ui_scale_slider.setValue(100)
        self.ui_scale_slider.valueChanged.connect(self._on_ui_scale_changed)
        self.ui_scale_slider.sliderReleased.connect(self._on_ui_scale_slider_released)
        interface_layout.addWidget(self.ui_scale_slider)

    def _build_downloads_card(self) -> None:
        downloads_card, downloads_layout = self._create_settings_card('Downloads')
        location_label = QLabel('Location', downloads_card)
        location_label.setObjectName('settingsSubtext')
        downloads_layout.addWidget(location_label)
        location_hint = QLabel('Choose where converted MIDI files are saved.', downloads_card)
        location_hint.setObjectName('settingsSubtext')
        location_hint.setWordWrap(True)
        downloads_layout.addWidget(location_hint)
        self.download_location_edit = QLineEdit(downloads_card)
        self.download_location_edit.setObjectName('settingsReadOnlyInput')
        self.download_location_edit.setReadOnly(True)
        downloads_layout.addWidget(self.download_location_edit)
        self.download_path_browse_btn = QPushButton('Change save folder', downloads_card)
        self.download_path_browse_btn.setObjectName('settingsActionButton')
        self.download_path_browse_btn.setMinimumWidth(0)
        self.download_path_browse_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.download_path_browse_btn.clicked.connect(self._browse_download_location)
        downloads_layout.addWidget(self.download_path_browse_btn)

    def _build_updates_card(self) -> None:
        updates_card, updates_layout = self._create_settings_card('Updates')
        self.auto_updates_checkbox = QCheckBox('Automatic Update', updates_card)
        self.auto_updates_checkbox.toggled.connect(self._on_auto_updates_toggled)
        updates_layout.addWidget(self.auto_updates_checkbox)
        self.check_updates_btn = QPushButton('Check for updates now', updates_card)
        self.check_updates_btn.setObjectName('settingsActionButton')
        self.check_updates_btn.setMinimumWidth(0)
        self.check_updates_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.check_updates_btn.clicked.connect(self.checkUpdatesNowRequested.emit)
        updates_layout.addWidget(self.check_updates_btn)
        self.reset_settings_btn = QPushButton('Reset all settings', updates_card)
        self.reset_settings_btn.setObjectName('settingsActionButton')
        self.reset_settings_btn.setMinimumWidth(0)
        self.reset_settings_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.reset_settings_btn.clicked.connect(self.resetSettingsRequested.emit)
        updates_layout.addWidget(self.reset_settings_btn)

    @staticmethod
    def _configure_mode_button(button: QPushButton) -> None:
        button.setObjectName('modeButton')
        button.setCheckable(True)
        button.setFocusPolicy(Qt.NoFocus)
        button.setMinimumWidth(0)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    @classmethod
    def _threshold_to_slider_value(cls, value: float) -> int:
        return int(round(float(value) * cls._MODERN_THRESHOLD_SCALE))

    @classmethod
    def _slider_value_to_threshold(cls, value: int) -> float:
        return float(value) / float(cls._MODERN_THRESHOLD_SCALE)

    @staticmethod
    def _clamp_threshold(value: float, *, minimum: float, maximum: float, default: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = float(default)
        if not (parsed == parsed):
            parsed = float(default)
        return max(float(minimum), min(float(maximum), float(parsed)))

    @classmethod
    def _clamp_scale(cls, value: float | int | str | None, *, default: float=1.0) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = float(default)
        if not (parsed == parsed):
            parsed = float(default)
        return max(float(cls._MODERN_SCALE_MIN), min(float(cls._MODERN_SCALE_MAX), float(parsed)))

    def _create_settings_card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(self)
        card.setObjectName('settingsCard')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(6)
        header = QLabel(title, card)
        header.setObjectName('settingsCardTitle')
        layout.addWidget(header)
        self._settings_card_layouts.append(layout)
        self._root_layout.addWidget(card)
        return (card, layout)

    @staticmethod
    def _set_control_cursor(widget: QWidget) -> None:
        sync_pointer_cursor(widget)

    def _modern_feature_checkboxes(self) -> tuple[QCheckBox, QCheckBox, QCheckBox, QCheckBox]:
        return (
            self.modern_adaptive_thresholds_checkbox,
            self.modern_input_normalization_checkbox,
            self.modern_smart_overlap_stitching_checkbox,
            self.modern_auto_calibration_checkbox,
        )

    def _modern_manual_sliders(self) -> tuple[QSlider, QSlider, QSlider, QSlider]:
        return (
            self.modern_onset_slider,
            self.modern_offset_slider,
            self.modern_frame_slider,
            self.modern_pedal_offset_slider,
            self.modern_cleanup_scale_slider,
            self.modern_pedal_cluster_scale_slider,
            self.modern_alignment_gate_scale_slider,
        )

    def _interactive_controls(self) -> tuple[QWidget, ...]:
        return (
            self.cpu_button,
            self.gpu_button,
            self.legacy_method_button,
            self.modern_method_button,
            self.auto_provider_button,
            self.cuda_provider_button,
            self.dml_provider_button,
            self.batch_slider,
            self.ui_scale_slider,
            self.download_path_browse_btn,
            self.auto_updates_checkbox,
            self.check_updates_btn,
            self.reset_settings_btn,
            *self._modern_feature_checkboxes(),
            *self._modern_manual_sliders(),
        )

    def _refresh_styled_controls(self) -> None:
        self.batch_slider.update()
        self.ui_scale_slider.update()
        self.auto_updates_checkbox.update()
        for checkbox in self._modern_feature_checkboxes():
            checkbox.update()
        for slider in self._modern_manual_sliders():
            slider.update()

    def _apply_interaction_cursors(self) -> None:
        for widget in self._interactive_controls():
            self._set_control_cursor(widget)

    def refresh_cursor_state(self) -> None:
        self.setCursor(Qt.ArrowCursor)
        self._apply_interaction_cursors()

    def _install_control_styles(self) -> None:
        self._slider_style = RoundHandleSliderStyle(handle_color=self.theme.text_primary, border_color=self.theme.border, groove_color=self.theme.border, fill_color=self.theme.accent, handle_size=18, groove_height=6, parent=self.batch_slider)
        self.batch_slider.setStyle(self._slider_style)
        self._ui_scale_slider_style = RoundHandleSliderStyle(handle_color=self.theme.text_primary, border_color=self.theme.border, groove_color=self.theme.border, fill_color=self.theme.accent, handle_size=18, groove_height=6, parent=self.ui_scale_slider)
        self.ui_scale_slider.setStyle(self._ui_scale_slider_style)
        self._checkbox_style = SquareCheckBoxStyle(border_color=self.theme.border, fill_color=self.theme.accent, check_color=self.theme.text_primary, size=20, radius=4, parent=self.auto_updates_checkbox)
        self.auto_updates_checkbox.setStyle(self._checkbox_style)
        self._modern_checkbox_styles = []
        for checkbox in self._modern_feature_checkboxes():
            style = SquareCheckBoxStyle(border_color=self.theme.border, fill_color=self.theme.accent, check_color=self.theme.text_primary, size=20, radius=4, parent=checkbox)
            checkbox.setStyle(style)
            self._modern_checkbox_styles.append(style)
        self._modern_slider_styles = []
        for slider in self._modern_manual_sliders():
            style = RoundHandleSliderStyle(handle_color=self.theme.text_primary, border_color=self.theme.border, groove_color=self.theme.border, fill_color=self.theme.accent, handle_size=18, groove_height=6, parent=slider)
            slider.setStyle(style)
            self._modern_slider_styles.append(style)

    @staticmethod
    def _scaled(value: int, scale: float, minimum: int=1) -> int:
        return max(minimum, int(round(value * scale)))

    @classmethod
    def _normalize_ui_scale(cls, value: int | str) -> int:
        normalized = max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, int(value)))
        normalized = int(round(normalized / cls._UI_SCALE_STEP) * cls._UI_SCALE_STEP)
        return max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, normalized))

    def _update_action_button_heights(self, scale: float) -> None:
        action_height = self._scaled(self._SETTINGS_ACTION_BUTTON_HEIGHT, scale, 24)
        self.download_path_browse_btn.setMinimumHeight(action_height)
        self.check_updates_btn.setMinimumHeight(action_height)
        self.reset_settings_btn.setMinimumHeight(action_height)

    def set_scale(self, scale: float) -> None:
        scale = max(0.5, min(2.0, float(scale)))
        margin = self._scaled(14, scale, 8)
        self._root_layout.setContentsMargins(margin, margin, margin, margin)
        self._root_layout.setSpacing(self._scaled(7, scale, 4))
        card_margin_x = self._scaled(8, scale, 5)
        card_margin_y = self._scaled(7, scale, 4)
        card_spacing = self._scaled(6, scale, 4)
        for card_layout in self._settings_card_layouts:
            card_layout.setContentsMargins(card_margin_x, card_margin_y, card_margin_x, card_margin_y)
            card_layout.setSpacing(card_spacing)
        mode_pad = self._scaled(2, scale, 1)
        self._mode_holder_layout.setContentsMargins(mode_pad, mode_pad, mode_pad, mode_pad)
        self._mode_holder_layout.setSpacing(self._scaled(2, scale, 1))
        self._conversion_holder_layout.setContentsMargins(mode_pad, mode_pad, mode_pad, mode_pad)
        self._conversion_holder_layout.setSpacing(self._scaled(2, scale, 1))
        self._provider_holder_layout.setContentsMargins(mode_pad, mode_pad, mode_pad, mode_pad)
        self._provider_holder_layout.setSpacing(self._scaled(2, scale, 1))
        self._modern_manual_layout.setSpacing(self._scaled(6, scale, 4))
        handle = self._scaled(18, scale, 14)
        groove = self._scaled(6, scale, 4)
        checkbox_size = self._scaled(20, scale, 14)
        checkbox_radius = self._scaled(4, scale, 2)
        if self._slider_style is not None:
            self._slider_style.set_metrics(handle_size=handle, groove_height=groove)
        if self._ui_scale_slider_style is not None:
            self._ui_scale_slider_style.set_metrics(handle_size=handle, groove_height=groove)
        for style in self._modern_slider_styles:
            style.set_metrics(handle_size=handle, groove_height=groove)
        if self._checkbox_style is not None:
            self._checkbox_style.set_metrics(size=checkbox_size, radius=checkbox_radius)
        for style in self._modern_checkbox_styles:
            style.set_metrics(size=checkbox_size, radius=checkbox_radius)
        self.download_location_edit.setMinimumHeight(self._scaled(30, scale, 22))
        self._update_action_button_heights(scale)
        self._update_value_label_widths(scale)
        self._update_runtime_label_height(scale)
        self._refresh_styled_controls()

    def _update_value_label_widths(self, scale: float) -> None:
        padding = self._scaled(12, scale, 8)
        batch_width = QFontMetrics(self.batch_value.font()).horizontalAdvance(str(GPU_BATCH_SIZE_MAX))
        ui_scale_width = QFontMetrics(self.ui_scale_value.font()).horizontalAdvance(f'{UI_SCALE_PERCENT_MAX}%')
        threshold_width = QFontMetrics(self.modern_onset_value.font()).horizontalAdvance('0.000')
        scale_width = QFontMetrics(self.modern_cleanup_scale_value.font()).horizontalAdvance('2.000')
        self.batch_value.setMinimumWidth(max(self._scaled(28, scale, 22), batch_width + padding))
        self.ui_scale_value.setMinimumWidth(max(self._scaled(52, scale, 38), ui_scale_width + padding))
        for label in (self.modern_onset_value, self.modern_offset_value, self.modern_frame_value, self.modern_pedal_offset_value):
            label.setMinimumWidth(max(self._scaled(52, scale, 36), threshold_width + padding))
        for label in (self.modern_cleanup_scale_value, self.modern_pedal_cluster_scale_value, self.modern_alignment_gate_scale_value):
            label.setMinimumWidth(max(self._scaled(52, scale, 36), scale_width + padding))

    def _update_runtime_label_height(self, scale: float) -> None:
        line_height = QFontMetrics(self.runtime_label.font()).lineSpacing()
        padding = self._scaled(2, scale, 2)
        reserved = line_height * self._RUNTIME_LABEL_LINES + padding
        self.runtime_label.setMinimumHeight(int(reserved))
        self.runtime_label.setMaximumHeight(int(reserved))

    def set_theme(self, theme: ThemePalette) -> None:
        self.theme = theme
        if self._slider_style is not None:
            self._slider_style.set_colors(handle_color=self.theme.text_primary, border_color=self.theme.border, groove_color=self.theme.border, fill_color=self.theme.accent)
            self._slider_style.set_metrics(handle_size=18, groove_height=6)
        if self._ui_scale_slider_style is not None:
            self._ui_scale_slider_style.set_colors(handle_color=self.theme.text_primary, border_color=self.theme.border, groove_color=self.theme.border, fill_color=self.theme.accent)
            self._ui_scale_slider_style.set_metrics(handle_size=18, groove_height=6)
        for style in self._modern_slider_styles:
            style.set_colors(handle_color=self.theme.text_primary, border_color=self.theme.border, groove_color=self.theme.border, fill_color=self.theme.accent)
            style.set_metrics(handle_size=18, groove_height=6)
        if self._checkbox_style is not None:
            self._checkbox_style.set_colors(border_color=self.theme.border, fill_color=self.theme.accent, check_color=self.theme.text_primary)
            self._checkbox_style.set_metrics(size=20, radius=4)
        for style in self._modern_checkbox_styles:
            style.set_colors(border_color=self.theme.border, fill_color=self.theme.accent, check_color=self.theme.text_primary)
            style.set_metrics(size=20, radius=4)
        self._refresh_styled_controls()

    def _on_device_clicked(self, checked: bool, preference: str) -> None:
        if not checked or self._updating:
            return
        self._selected_device = 'gpu' if preference == 'gpu' else 'cpu'
        self.devicePreferenceChanged.emit(self._selected_device)
        self._refresh_runtime_ui()

    def _set_conversion_method_ui(self, method: str) -> None:
        normalized = normalize_conversion_method(method)
        self._conversion_method = normalized
        self.legacy_method_button.setChecked(normalized == 'legacy_v1')
        self.modern_method_button.setChecked(normalized == 'modern')
        self._refresh_conversion_feature_toggles_ui()

    def _refresh_conversion_feature_toggles_ui(self) -> None:
        modern_selected = self._conversion_method == 'modern'
        self.modern_features_label.setVisible(modern_selected)
        for checkbox in self._modern_feature_checkboxes():
            checkbox.setVisible(modern_selected)
        self.modern_input_normalization_checkbox.setEnabled(modern_selected)
        self.modern_smart_overlap_stitching_checkbox.setEnabled(modern_selected)
        self.modern_auto_calibration_checkbox.setEnabled(modern_selected)
        self.modern_adaptive_thresholds_checkbox.setEnabled(modern_selected and self._modern_auto_calibration_enabled)
        manual_visible = modern_selected and (not self._modern_auto_calibration_enabled)
        self.modern_manual_tuning_label.setVisible(manual_visible)
        self.modern_manual_tuning_container.setVisible(manual_visible)
        self.modern_calibration_tuning_label.setVisible(manual_visible)
        for slider in self._modern_manual_sliders():
            slider.setEnabled(manual_visible)

    def _on_conversion_method_clicked(self, checked: bool, method: str) -> None:
        if not checked or self._updating:
            return
        normalized = normalize_conversion_method(method)
        self._set_conversion_method_ui(normalized)
        self.conversionMethodChanged.emit(normalized)

    def _set_modern_feature_toggles_ui(self, *, adaptive_thresholds_enabled: bool, input_normalization_enabled: bool, smart_overlap_stitching_enabled: bool, auto_calibration_enabled: bool) -> None:
        self._modern_adaptive_thresholds_enabled = bool(adaptive_thresholds_enabled)
        self._modern_input_normalization_enabled = bool(input_normalization_enabled)
        self._modern_smart_overlap_stitching_enabled = bool(smart_overlap_stitching_enabled)
        self._modern_auto_calibration_enabled = bool(auto_calibration_enabled)
        self.modern_adaptive_thresholds_checkbox.setChecked(self._modern_adaptive_thresholds_enabled)
        self.modern_input_normalization_checkbox.setChecked(self._modern_input_normalization_enabled)
        self.modern_smart_overlap_stitching_checkbox.setChecked(self._modern_smart_overlap_stitching_enabled)
        self.modern_auto_calibration_checkbox.setChecked(self._modern_auto_calibration_enabled)
        self._refresh_conversion_feature_toggles_ui()

    def _set_modern_manual_thresholds_ui(self, *, onset_threshold: float, offset_threshold: float, frame_threshold: float, pedal_offset_threshold: float) -> None:
        self._modern_manual_onset_threshold = self._clamp_threshold(onset_threshold, minimum=MODERN_ONSET_THRESHOLD_MIN, maximum=MODERN_ONSET_THRESHOLD_MAX, default=MODERN_ONSET_THRESHOLD_DEFAULT)
        self._modern_manual_offset_threshold = self._clamp_threshold(offset_threshold, minimum=MODERN_OFFSET_THRESHOLD_MIN, maximum=MODERN_OFFSET_THRESHOLD_MAX, default=MODERN_OFFSET_THRESHOLD_DEFAULT)
        self._modern_manual_frame_threshold = self._clamp_threshold(frame_threshold, minimum=MODERN_FRAME_THRESHOLD_MIN, maximum=MODERN_FRAME_THRESHOLD_MAX, default=MODERN_FRAME_THRESHOLD_DEFAULT)
        self._modern_manual_pedal_offset_threshold = self._clamp_threshold(pedal_offset_threshold, minimum=MODERN_PEDAL_OFFSET_THRESHOLD_MIN, maximum=MODERN_PEDAL_OFFSET_THRESHOLD_MAX, default=MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT)
        self.modern_onset_slider.setValue(self._threshold_to_slider_value(self._modern_manual_onset_threshold))
        self.modern_offset_slider.setValue(self._threshold_to_slider_value(self._modern_manual_offset_threshold))
        self.modern_frame_slider.setValue(self._threshold_to_slider_value(self._modern_manual_frame_threshold))
        self.modern_pedal_offset_slider.setValue(self._threshold_to_slider_value(self._modern_manual_pedal_offset_threshold))
        self.modern_onset_value.setText(f'{self._modern_manual_onset_threshold:.3f}')
        self.modern_offset_value.setText(f'{self._modern_manual_offset_threshold:.3f}')
        self.modern_frame_value.setText(f'{self._modern_manual_frame_threshold:.3f}')
        self.modern_pedal_offset_value.setText(f'{self._modern_manual_pedal_offset_threshold:.3f}')
        self._clear_pending_modern_thresholds()

    def _set_modern_calibration_scales_ui(self, *, cleanup_scale: float, pedal_cluster_scale: float, alignment_gate_scale: float) -> None:
        self._modern_cleanup_scale = self._clamp_scale(cleanup_scale)
        self._modern_pedal_cluster_scale = self._clamp_scale(pedal_cluster_scale)
        self._modern_alignment_gate_scale = self._clamp_scale(alignment_gate_scale)
        self.modern_cleanup_scale_slider.setValue(self._threshold_to_slider_value(self._modern_cleanup_scale))
        self.modern_pedal_cluster_scale_slider.setValue(self._threshold_to_slider_value(self._modern_pedal_cluster_scale))
        self.modern_alignment_gate_scale_slider.setValue(self._threshold_to_slider_value(self._modern_alignment_gate_scale))
        self.modern_cleanup_scale_value.setText(f'{self._modern_cleanup_scale:.3f}')
        self.modern_pedal_cluster_scale_value.setText(f'{self._modern_pedal_cluster_scale:.3f}')
        self.modern_alignment_gate_scale_value.setText(f'{self._modern_alignment_gate_scale:.3f}')
        self._clear_pending_modern_scale_values()

    def _on_modern_adaptive_thresholds_toggled(self, enabled: bool) -> None:
        self._modern_adaptive_thresholds_enabled = bool(enabled)
        if self._updating:
            return
        self.modernAdaptiveThresholdsChanged.emit(bool(enabled))

    def _on_modern_input_normalization_toggled(self, enabled: bool) -> None:
        self._modern_input_normalization_enabled = bool(enabled)
        if self._updating:
            return
        self.modernInputNormalizationChanged.emit(bool(enabled))

    def _on_modern_smart_overlap_stitching_toggled(self, enabled: bool) -> None:
        self._modern_smart_overlap_stitching_enabled = bool(enabled)
        if self._updating:
            return
        self.modernSmartOverlapStitchingChanged.emit(bool(enabled))

    def _on_modern_auto_calibration_toggled(self, enabled: bool) -> None:
        self._modern_auto_calibration_enabled = bool(enabled)
        self._refresh_conversion_feature_toggles_ui()
        if self._updating:
            return
        self.modernAutoCalibrationChanged.emit(bool(enabled))

    def _on_modern_onset_threshold_changed(self, slider_value: int) -> None:
        threshold = self._clamp_threshold(self._slider_value_to_threshold(slider_value), minimum=MODERN_ONSET_THRESHOLD_MIN, maximum=MODERN_ONSET_THRESHOLD_MAX, default=MODERN_ONSET_THRESHOLD_DEFAULT)
        self._modern_manual_onset_threshold = threshold
        self.modern_onset_value.setText(f'{threshold:.3f}')
        if self._updating:
            return
        self._pending_modern_onset_threshold = threshold

    def _on_modern_offset_threshold_changed(self, slider_value: int) -> None:
        threshold = self._clamp_threshold(self._slider_value_to_threshold(slider_value), minimum=MODERN_OFFSET_THRESHOLD_MIN, maximum=MODERN_OFFSET_THRESHOLD_MAX, default=MODERN_OFFSET_THRESHOLD_DEFAULT)
        self._modern_manual_offset_threshold = threshold
        self.modern_offset_value.setText(f'{threshold:.3f}')
        if self._updating:
            return
        self._pending_modern_offset_threshold = threshold

    def _on_modern_frame_threshold_changed(self, slider_value: int) -> None:
        threshold = self._clamp_threshold(self._slider_value_to_threshold(slider_value), minimum=MODERN_FRAME_THRESHOLD_MIN, maximum=MODERN_FRAME_THRESHOLD_MAX, default=MODERN_FRAME_THRESHOLD_DEFAULT)
        self._modern_manual_frame_threshold = threshold
        self.modern_frame_value.setText(f'{threshold:.3f}')
        if self._updating:
            return
        self._pending_modern_frame_threshold = threshold

    def _on_modern_pedal_offset_threshold_changed(self, slider_value: int) -> None:
        threshold = self._clamp_threshold(self._slider_value_to_threshold(slider_value), minimum=MODERN_PEDAL_OFFSET_THRESHOLD_MIN, maximum=MODERN_PEDAL_OFFSET_THRESHOLD_MAX, default=MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT)
        self._modern_manual_pedal_offset_threshold = threshold
        self.modern_pedal_offset_value.setText(f'{threshold:.3f}')
        if self._updating:
            return
        self._pending_modern_pedal_offset_threshold = threshold

    def _on_modern_cleanup_scale_changed(self, slider_value: int) -> None:
        value = self._clamp_scale(self._slider_value_to_threshold(slider_value))
        self._modern_cleanup_scale = value
        self.modern_cleanup_scale_value.setText(f'{value:.3f}')
        if self._updating:
            return
        self._pending_modern_cleanup_scale = value

    def _on_modern_pedal_cluster_scale_changed(self, slider_value: int) -> None:
        value = self._clamp_scale(self._slider_value_to_threshold(slider_value))
        self._modern_pedal_cluster_scale = value
        self.modern_pedal_cluster_scale_value.setText(f'{value:.3f}')
        if self._updating:
            return
        self._pending_modern_pedal_cluster_scale = value

    def _on_modern_alignment_gate_scale_changed(self, slider_value: int) -> None:
        value = self._clamp_scale(self._slider_value_to_threshold(slider_value))
        self._modern_alignment_gate_scale = value
        self.modern_alignment_gate_scale_value.setText(f'{value:.3f}')
        if self._updating:
            return
        self._pending_modern_alignment_gate_scale = value

    def _on_modern_onset_threshold_released(self) -> None:
        if self._updating:
            return
        if self._pending_modern_onset_threshold is None:
            return
        self.modernOnsetThresholdChanged.emit(self._pending_modern_onset_threshold)
        self._pending_modern_onset_threshold = None

    def _on_modern_offset_threshold_released(self) -> None:
        if self._updating:
            return
        if self._pending_modern_offset_threshold is None:
            return
        self.modernOffsetThresholdChanged.emit(self._pending_modern_offset_threshold)
        self._pending_modern_offset_threshold = None

    def _on_modern_frame_threshold_released(self) -> None:
        if self._updating:
            return
        if self._pending_modern_frame_threshold is None:
            return
        self.modernFrameThresholdChanged.emit(self._pending_modern_frame_threshold)
        self._pending_modern_frame_threshold = None

    def _on_modern_pedal_offset_threshold_released(self) -> None:
        if self._updating:
            return
        if self._pending_modern_pedal_offset_threshold is None:
            return
        self.modernPedalOffsetThresholdChanged.emit(self._pending_modern_pedal_offset_threshold)
        self._pending_modern_pedal_offset_threshold = None

    def _on_modern_cleanup_scale_released(self) -> None:
        if self._updating:
            return
        if self._pending_modern_cleanup_scale is None:
            return
        self.modernCleanupScaleChanged.emit(self._pending_modern_cleanup_scale)
        self._pending_modern_cleanup_scale = None

    def _on_modern_pedal_cluster_scale_released(self) -> None:
        if self._updating:
            return
        if self._pending_modern_pedal_cluster_scale is None:
            return
        self.modernPedalClusterScaleChanged.emit(self._pending_modern_pedal_cluster_scale)
        self._pending_modern_pedal_cluster_scale = None

    def _on_modern_alignment_gate_scale_released(self) -> None:
        if self._updating:
            return
        if self._pending_modern_alignment_gate_scale is None:
            return
        self.modernAlignmentGateScaleChanged.emit(self._pending_modern_alignment_gate_scale)
        self._pending_modern_alignment_gate_scale = None

    def _clear_pending_modern_thresholds(self) -> None:
        self._pending_modern_onset_threshold = None
        self._pending_modern_offset_threshold = None
        self._pending_modern_frame_threshold = None
        self._pending_modern_pedal_offset_threshold = None

    def _clear_pending_modern_scale_values(self) -> None:
        self._pending_modern_cleanup_scale = None
        self._pending_modern_pedal_cluster_scale = None
        self._pending_modern_alignment_gate_scale = None

    @staticmethod
    def _normalize_gpu_provider_preference(value: str | None) -> str:
        normalized = str(value or 'auto').strip().lower()
        if normalized in {'auto', 'cuda', 'dml'}:
            return normalized
        return 'auto'

    def _set_gpu_provider_preference_ui(self, preference: str) -> None:
        normalized = self._normalize_gpu_provider_preference(preference)
        self._gpu_provider_preference = normalized
        self.auto_provider_button.setChecked(normalized == 'auto')
        self.cuda_provider_button.setChecked(normalized == 'cuda')
        self.dml_provider_button.setChecked(normalized == 'dml')

    def _on_provider_clicked(self, checked: bool, preference: str) -> None:
        if not checked or self._updating:
            return
        normalized = self._normalize_gpu_provider_preference(preference)
        self._set_gpu_provider_preference_ui(normalized)
        self.gpuProviderPreferenceChanged.emit(normalized)

    def _on_batch_changed(self, value: int) -> None:
        self.batch_value.setText(str(value))
        if self._updating:
            return
        self.gpuBatchSizeChanged.emit(int(value))

    def _on_auto_updates_toggled(self, enabled: bool) -> None:
        if self._updating:
            return
        self.autoCheckUpdatesChanged.emit(bool(enabled))

    def _browse_download_location(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, 'Choose download location', self.download_location_edit.text().strip())
        if not selected:
            return
        self.download_location_edit.setText(selected)
        if self._updating:
            return
        self.downloadLocationChanged.emit(selected)

    def _on_ui_scale_changed(self, value: int) -> None:
        raw_value = max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, int(value)))
        snapped = self._normalize_ui_scale(raw_value)
        if snapped != raw_value:
            self.ui_scale_slider.blockSignals(True)
            self.ui_scale_slider.setValue(snapped)
            self.ui_scale_slider.blockSignals(False)
        self.ui_scale_value.setText(f'{snapped}%')
        if self._updating:
            return
        self._pending_ui_scale_percent = snapped

    def _on_ui_scale_slider_released(self) -> None:
        if self._updating:
            return
        if self._pending_ui_scale_percent is not None:
            self.uiScalePercentChanged.emit(self._pending_ui_scale_percent)
            self._pending_ui_scale_percent = None

    def set_values(self, *, device_preference: str, conversion_method: str, modern_adaptive_thresholds_enabled: bool, modern_input_normalization_enabled: bool, modern_smart_overlap_stitching_enabled: bool, modern_auto_calibration_enabled: bool, modern_cleanup_scale: float, modern_pedal_cluster_scale: float, modern_alignment_gate_scale: float, modern_manual_onset_threshold: float, modern_manual_offset_threshold: float, modern_manual_frame_threshold: float, modern_manual_pedal_offset_threshold: float, gpu_provider_preference: str, gpu_batch_size: int, ui_scale_percent: int, download_location: str, auto_check_updates: bool) -> None:
        self._updating = True
        pref = 'gpu' if str(device_preference).strip().lower() == 'gpu' else 'cpu'
        self._selected_device = pref
        self.cpu_button.setChecked(pref == 'cpu')
        self.gpu_button.setChecked(pref == 'gpu')
        self._set_conversion_method_ui(conversion_method)
        self._set_modern_feature_toggles_ui(adaptive_thresholds_enabled=modern_adaptive_thresholds_enabled, input_normalization_enabled=modern_input_normalization_enabled, smart_overlap_stitching_enabled=modern_smart_overlap_stitching_enabled, auto_calibration_enabled=modern_auto_calibration_enabled)
        self._set_modern_calibration_scales_ui(cleanup_scale=modern_cleanup_scale, pedal_cluster_scale=modern_pedal_cluster_scale, alignment_gate_scale=modern_alignment_gate_scale)
        self._set_modern_manual_thresholds_ui(onset_threshold=modern_manual_onset_threshold, offset_threshold=modern_manual_offset_threshold, frame_threshold=modern_manual_frame_threshold, pedal_offset_threshold=modern_manual_pedal_offset_threshold)
        self._set_gpu_provider_preference_ui(gpu_provider_preference)
        self.batch_slider.setValue(max(GPU_BATCH_SIZE_MIN, min(GPU_BATCH_SIZE_MAX, int(gpu_batch_size))))
        self.batch_value.setText(str(self.batch_slider.value()))
        ui_scale = self._normalize_ui_scale(ui_scale_percent)
        self.ui_scale_slider.setValue(ui_scale)
        self.ui_scale_value.setText(f'{ui_scale}%')
        self._pending_ui_scale_percent = None
        self.download_location_edit.setText(str(download_location or '').strip())
        self.auto_updates_checkbox.setChecked(bool(auto_check_updates))
        self._updating = False
        self._refresh_runtime_ui()

    def selected_device(self) -> str:
        return self._selected_device

    def set_runtime_status(self, *, checking: bool, runtime_available: bool, cuda_available: bool, dml_available: bool, active_provider: str) -> None:
        self._runtime_checking = bool(checking)
        self._runtime_available = bool(runtime_available)
        self._cuda_available = bool(cuda_available)
        self._dml_available = bool(dml_available)
        self._active_provider = str(active_provider or 'CPU')
        self._refresh_runtime_ui()

    def _refresh_runtime_ui(self) -> None:
        selected = self.selected_device()
        provider_ready = self._cuda_available or self._dml_available
        active_provider = str(self._active_provider or 'CPU')
        provider_controls_enabled = selected == 'gpu'
        self.auto_provider_button.setEnabled(provider_controls_enabled)
        self.cuda_provider_button.setEnabled(provider_controls_enabled)
        self.dml_provider_button.setEnabled(provider_controls_enabled)
        if self._runtime_checking:
            self.runtime_label.setText(RUNTIME_LABEL_CHECKING)
        elif not self._runtime_available:
            self.runtime_label.setText(RUNTIME_LABEL_MISSING)
        elif selected == 'gpu':
            if active_provider.strip().lower().startswith('cpu fallback:'):
                self.runtime_label.setText(f'GPU unavailable. Using {active_provider}')
            elif active_provider.strip().lower().startswith('validation pending'):
                self.runtime_label.setText(RUNTIME_LABEL_GPU_VALIDATING)
            elif not provider_ready:
                self.runtime_label.setText(RUNTIME_LABEL_GPU_NO_PACK)
            elif active_provider.strip().upper().startswith('CPU'):
                self.runtime_label.setText(RUNTIME_LABEL_GPU_UNAVAILABLE)
            else:
                self.runtime_label.setText(RUNTIME_LABEL_GPU_ACTIVE)
        elif provider_ready:
            self.runtime_label.setText(RUNTIME_LABEL_CPU_WITH_GPU_PACK)
        else:
            self.runtime_label.setText(RUNTIME_LABEL_CPU_NO_GPU_PACK)
        batch_enabled = selected == 'gpu' and provider_ready
        self.batch_slider.setEnabled(batch_enabled)
        self.batch_label.setEnabled(batch_enabled)
        self.batch_value.setEnabled(batch_enabled)
        self.batch_slider.update()
        self._apply_interaction_cursors()

