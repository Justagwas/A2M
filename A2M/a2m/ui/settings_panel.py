from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QSizePolicy, QVBoxLayout, QWidget

from a2m.core.config import ENGINE_UNIFORM_VELOCITY_DEFAULT, ENGINE_UNIFORM_VELOCITY_MAX, ENGINE_UNIFORM_VELOCITY_MIN, UI_SCALE_PERCENT_MAX, UI_SCALE_PERCENT_MIN
from a2m.core.messages import RUNTIME_LABEL_CHECKING, RUNTIME_LABEL_CPU_NO_GPU_PACK, RUNTIME_LABEL_CPU_WITH_GPU_PACK
from a2m.core.messages import RUNTIME_LABEL_CUDA_ACTIVE, RUNTIME_LABEL_DML_ACTIVE, RUNTIME_LABEL_GPU_NO_PACK, RUNTIME_LABEL_GPU_UNAVAILABLE
from a2m.core.messages import RUNTIME_LABEL_MISSING
from a2m.core.resource_service import gpu_memory_level_for_batch, gpu_memory_presets, normalize_gpu_batch_size, normalize_gpu_memory_max_batch, normalize_performance_mode
from .interaction import sync_pointer_cursor
from .theme import ThemePalette
from .widgets.custom_controls import RoundHandleSliderStyle, SquareCheckBoxStyle


class SettingsPanel(QFrame):
    devicePreferenceChanged = Signal(str)
    gpuProviderPreferenceChanged = Signal(str)
    enginePedalsEnabledChanged = Signal(bool)
    engineVelocityModeChanged = Signal(str)
    engineUniformVelocityChanged = Signal(int)
    transcriptionPerformanceModeChanged = Signal(str, str)
    gpuMemoryUsageChanged = Signal(int)
    resetEngineSettingsRequested = Signal()
    uiScalePercentChanged = Signal(int)
    downloadLocationChanged = Signal(str)
    autoCheckUpdatesChanged = Signal(bool)
    checkUpdatesNowRequested = Signal()
    resetSettingsRequested = Signal()
    _UI_SCALE_STEP = 5

    def __init__(self, theme: ThemePalette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setObjectName('settingsBody')
        self.setMinimumWidth(0)
        self._updating = False
        self._selected_device = 'cpu'
        self._gpu_provider_preference = 'dml'
        self._runtime_checking = True
        self._runtime_available = False
        self._cuda_available = False
        self._dml_available = False
        self._active_provider = 'CPU'
        self._engine_pedals_enabled = True
        self._engine_velocity_mode = 'expressive'
        self._cpu_performance_mode = 'balanced'
        self._gpu_performance_mode = 'balanced'
        self._gpu_batch_size = 2
        self._gpu_memory_max_batch = 4
        self._gpu_memory_presets = gpu_memory_presets(self._gpu_memory_max_batch)
        self._engine_controls_enabled = True
        self._device_controls_enabled = True
        self._pending_uniform_velocity: int | None = None
        self._pending_ui_scale_percent: int | None = None
        self._settings_card_layouts: list[QVBoxLayout] = []
        self._engine_choice_layouts: list[QHBoxLayout] = []
        self._scale_slider_style: RoundHandleSliderStyle | None = None
        self._velocity_slider_style: RoundHandleSliderStyle | None = None
        self._checkbox_style: SquareCheckBoxStyle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(7)
        self._root_layout = layout
        title = QLabel('Settings', self)
        title.setObjectName('title')
        layout.addWidget(title)
        self._build_device_card()
        self._build_engine_card()
        self._build_interface_card()
        self._build_downloads_card()
        self._build_updates_card()
        layout.addStretch(1)
        self._install_control_styles()
        self._refresh_performance_ui()
        self._refresh_engine_ui()
        self._refresh_runtime_ui()
        self.refresh_cursor_state()
        self.set_scale(1.0)

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
        return card, layout

    @staticmethod
    def _configure_mode_button(button: QPushButton) -> None:
        button.setObjectName('modeButton')
        button.setCheckable(True)
        button.setFocusPolicy(Qt.NoFocus)
        button.setMinimumWidth(0)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def _build_device_card(self) -> None:
        card, layout = self._create_settings_card('Device')
        label = QLabel('Device mode', card)
        label.setObjectName('settingsSubtext')
        layout.addWidget(label)
        holder = QFrame(card)
        holder.setObjectName('modeHolder')
        holder_layout = QHBoxLayout(holder)
        holder_layout.setContentsMargins(2, 2, 2, 2)
        holder_layout.setSpacing(2)
        self._mode_holder_layout = holder_layout
        self.mode_group = QButtonGroup(self)
        self.cpu_button = QPushButton('CPU', holder)
        self.gpu_button = QPushButton('GPU', holder)
        for button, mode in ((self.cpu_button, 'cpu'), (self.gpu_button, 'gpu')):
            self._configure_mode_button(button)
            button.clicked.connect(lambda checked, selected=mode: self._on_device_clicked(checked, selected))
            holder_layout.addWidget(button)
            self.mode_group.addButton(button)
        self.cpu_button.setChecked(True)
        layout.addWidget(holder)

        self.provider_label = QLabel('GPU provider', card)
        self.provider_label.setObjectName('settingsSubtext')
        layout.addWidget(self.provider_label)
        self.provider_holder = QFrame(card)
        self.provider_holder.setObjectName('modeHolder')
        provider_layout = QHBoxLayout(self.provider_holder)
        provider_layout.setContentsMargins(2, 2, 2, 2)
        provider_layout.setSpacing(2)
        self._provider_holder_layout = provider_layout
        self.provider_group = QButtonGroup(self)
        self.cuda_provider_button = QPushButton('CUDA', self.provider_holder)
        self.dml_provider_button = QPushButton('DirectML', self.provider_holder)
        for button, provider in (
            (self.cuda_provider_button, 'cuda'),
            (self.dml_provider_button, 'dml'),
        ):
            self._configure_mode_button(button)
            button.clicked.connect(lambda checked, selected=provider: self._on_provider_clicked(checked, selected))
            provider_layout.addWidget(button)
            self.provider_group.addButton(button)
        self.dml_provider_button.setChecked(True)
        layout.addWidget(self.provider_holder)
        self.runtime_label = QLabel(RUNTIME_LABEL_CHECKING, card)
        self.runtime_label.setObjectName('muted')
        self.runtime_label.setWordWrap(True)
        self.runtime_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.runtime_label)

    def _build_engine_card(self) -> None:
        card, layout = self._create_settings_card('Piano Engine')
        self.performance_label = QLabel('CPU usage', card)
        self.performance_label.setObjectName('settingsSubtext')
        layout.addWidget(self.performance_label)
        performance_holder = QFrame(card)
        performance_holder.setObjectName('modeHolder')
        performance_layout = QHBoxLayout(performance_holder)
        performance_layout.setContentsMargins(2, 2, 2, 2)
        performance_layout.setSpacing(2)
        self._engine_choice_layouts.append(performance_layout)
        self.performance_group = QButtonGroup(self)
        self.performance_buttons: dict[str, QPushButton] = {}
        for text, mode in (('Low', 'low'), ('Balanced', 'balanced'), ('Fast', 'high'), ('Max', 'maximum')):
            button = QPushButton(text, performance_holder)
            self._configure_mode_button(button)
            button.clicked.connect(lambda checked, selected=mode: self._on_performance_clicked(checked, selected))
            performance_layout.addWidget(button)
            self.performance_group.addButton(button)
            self.performance_buttons[mode] = button
        self.performance_buttons['balanced'].setChecked(True)
        layout.addWidget(performance_holder)

        self.gpu_memory_container = QFrame(card)
        self.gpu_memory_container.setObjectName('cardContent')
        gpu_memory_layout = QVBoxLayout(self.gpu_memory_container)
        gpu_memory_layout.setContentsMargins(0, 0, 0, 0)
        gpu_memory_layout.setSpacing(5)
        gpu_memory_label = QLabel('GPU memory usage', self.gpu_memory_container)
        gpu_memory_label.setObjectName('settingsSubtext')
        gpu_memory_layout.addWidget(gpu_memory_label)
        gpu_memory_holder = QFrame(self.gpu_memory_container)
        gpu_memory_holder.setObjectName('modeHolder')
        gpu_memory_choice_layout = QHBoxLayout(gpu_memory_holder)
        gpu_memory_choice_layout.setContentsMargins(2, 2, 2, 2)
        gpu_memory_choice_layout.setSpacing(2)
        self._engine_choice_layouts.append(gpu_memory_choice_layout)
        self.gpu_memory_group = QButtonGroup(self)
        self.gpu_memory_buttons: dict[str, QPushButton] = {}
        for text, level in (('Low', 'low'), ('Balanced', 'balanced'), ('High', 'high'), ('Max', 'maximum')):
            button = QPushButton(text, gpu_memory_holder)
            self._configure_mode_button(button)
            button.clicked.connect(
                lambda checked, selected=level: self._on_gpu_memory_clicked(checked, selected)
            )
            gpu_memory_choice_layout.addWidget(button, 2 if level == 'balanced' else 1)
            self.gpu_memory_group.addButton(button)
            self.gpu_memory_buttons[level] = button
        self.gpu_memory_buttons['balanced'].setChecked(True)
        gpu_memory_layout.addWidget(gpu_memory_holder)
        layout.addWidget(self.gpu_memory_container)

        pedals_label = QLabel('Pedal events', card)
        pedals_label.setObjectName('settingsSubtext')
        layout.addWidget(pedals_label)
        pedals_holder = QFrame(card)
        pedals_holder.setObjectName('modeHolder')
        pedals_layout = QHBoxLayout(pedals_holder)
        pedals_layout.setContentsMargins(2, 2, 2, 2)
        pedals_layout.setSpacing(2)
        self._engine_choice_layouts.append(pedals_layout)
        self.pedals_group = QButtonGroup(self)
        self.pedals_off_button = QPushButton('Off', pedals_holder)
        self.pedals_on_button = QPushButton('On', pedals_holder)
        for button, enabled in ((self.pedals_off_button, False), (self.pedals_on_button, True)):
            self._configure_mode_button(button)
            button.clicked.connect(lambda checked, selected=enabled: self._on_pedals_clicked(checked, selected))
            pedals_layout.addWidget(button)
            self.pedals_group.addButton(button)
        self.pedals_on_button.setChecked(True)
        layout.addWidget(pedals_holder)

        dynamics_label = QLabel('Dynamics', card)
        dynamics_label.setObjectName('settingsSubtext')
        layout.addWidget(dynamics_label)
        dynamics_holder = QFrame(card)
        dynamics_holder.setObjectName('modeHolder')
        dynamics_layout = QHBoxLayout(dynamics_holder)
        dynamics_layout.setContentsMargins(2, 2, 2, 2)
        dynamics_layout.setSpacing(2)
        self._engine_choice_layouts.append(dynamics_layout)
        self.velocity_group = QButtonGroup(self)
        self.expressive_velocity_button = QPushButton('Expressive', dynamics_holder)
        self.uniform_velocity_button = QPushButton('Uniform', dynamics_holder)
        for button, mode in ((self.expressive_velocity_button, 'expressive'), (self.uniform_velocity_button, 'uniform')):
            self._configure_mode_button(button)
            button.clicked.connect(lambda checked, selected=mode: self._on_velocity_mode_clicked(checked, selected))
            dynamics_layout.addWidget(button)
            self.velocity_group.addButton(button)
        self.expressive_velocity_button.setChecked(True)
        layout.addWidget(dynamics_holder)

        velocity_row = QHBoxLayout()
        self.uniform_velocity_label = QLabel('Uniform velocity', card)
        self.uniform_velocity_label.setObjectName('settingsSubtext')
        self.uniform_velocity_value = QLabel(str(ENGINE_UNIFORM_VELOCITY_DEFAULT), card)
        self.uniform_velocity_value.setObjectName('muted')
        self.uniform_velocity_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        velocity_row.addWidget(self.uniform_velocity_label)
        velocity_row.addStretch(1)
        velocity_row.addWidget(self.uniform_velocity_value)
        layout.addLayout(velocity_row)
        self.uniform_velocity_slider = QSlider(Qt.Horizontal, card)
        self.uniform_velocity_slider.setRange(ENGINE_UNIFORM_VELOCITY_MIN, ENGINE_UNIFORM_VELOCITY_MAX)
        self.uniform_velocity_slider.setValue(ENGINE_UNIFORM_VELOCITY_DEFAULT)
        self.uniform_velocity_slider.valueChanged.connect(self._on_uniform_velocity_changed)
        self.uniform_velocity_slider.sliderReleased.connect(self._on_uniform_velocity_released)
        layout.addWidget(self.uniform_velocity_slider)

        self.reset_engine_btn = QPushButton('Restore Piano Engine defaults', card)
        self.reset_engine_btn.setObjectName('settingsActionButton')
        self.reset_engine_btn.clicked.connect(self.resetEngineSettingsRequested.emit)
        layout.addWidget(self.reset_engine_btn)

    def _build_interface_card(self) -> None:
        card, layout = self._create_settings_card('Interface')
        row = QHBoxLayout()
        label = QLabel('UI size', card)
        label.setObjectName('settingsSubtext')
        self.ui_scale_value = QLabel('100%', card)
        self.ui_scale_value.setObjectName('muted')
        self.ui_scale_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(self.ui_scale_value)
        layout.addLayout(row)
        self.ui_scale_slider = QSlider(Qt.Horizontal, card)
        self.ui_scale_slider.setRange(UI_SCALE_PERCENT_MIN, UI_SCALE_PERCENT_MAX)
        self.ui_scale_slider.setSingleStep(self._UI_SCALE_STEP)
        self.ui_scale_slider.setPageStep(self._UI_SCALE_STEP)
        self.ui_scale_slider.setValue(100)
        self.ui_scale_slider.valueChanged.connect(self._on_ui_scale_changed)
        self.ui_scale_slider.sliderReleased.connect(self._on_ui_scale_slider_released)
        layout.addWidget(self.ui_scale_slider)

    def _build_downloads_card(self) -> None:
        card, layout = self._create_settings_card('Output Location')
        hint = QLabel('Choose where converted MIDI files are saved.', card)
        hint.setObjectName('settingsSubtext')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.download_location_edit = QLineEdit(card)
        self.download_location_edit.setObjectName('settingsReadOnlyInput')
        self.download_location_edit.setReadOnly(True)
        layout.addWidget(self.download_location_edit)
        self.download_path_browse_btn = QPushButton('Change save folder', card)
        self.download_path_browse_btn.setObjectName('settingsActionButton')
        self.download_path_browse_btn.clicked.connect(self._browse_download_location)
        layout.addWidget(self.download_path_browse_btn)

    def _build_updates_card(self) -> None:
        card, layout = self._create_settings_card('Updates')
        self.auto_updates_checkbox = QCheckBox('Automatic updates', card)
        self.auto_updates_checkbox.toggled.connect(self._on_auto_updates_toggled)
        layout.addWidget(self.auto_updates_checkbox)
        self.check_updates_btn = QPushButton('Check for updates now', card)
        self.check_updates_btn.setObjectName('settingsActionButton')
        self.check_updates_btn.clicked.connect(self.checkUpdatesNowRequested.emit)
        layout.addWidget(self.check_updates_btn)
        self.reset_settings_btn = QPushButton('Reset all settings', card)
        self.reset_settings_btn.setObjectName('settingsActionButton')
        self.reset_settings_btn.clicked.connect(self.resetSettingsRequested.emit)
        layout.addWidget(self.reset_settings_btn)

    def _interactive_controls(self) -> tuple[QWidget, ...]:
        return (
            self.cpu_button, self.gpu_button, self.cuda_provider_button, self.dml_provider_button,
            *self.performance_buttons.values(), self.pedals_off_button, self.pedals_on_button,
            *self.gpu_memory_buttons.values(), self.expressive_velocity_button, self.uniform_velocity_button, self.uniform_velocity_slider,
            self.reset_engine_btn, self.ui_scale_slider, self.download_path_browse_btn,
            self.auto_updates_checkbox, self.check_updates_btn, self.reset_settings_btn,
        )

    def refresh_cursor_state(self) -> None:
        self.setCursor(Qt.ArrowCursor)
        for widget in self._interactive_controls():
            sync_pointer_cursor(widget)

    def _install_control_styles(self) -> None:
        self._scale_slider_style = RoundHandleSliderStyle(
            handle_color=self.theme.text_primary,
            border_color=self.theme.border,
            groove_color=self.theme.border,
            fill_color=self.theme.accent,
            handle_size=18,
            groove_height=6,
            parent=self.ui_scale_slider,
        )
        self.ui_scale_slider.setStyle(self._scale_slider_style)
        self._velocity_slider_style = RoundHandleSliderStyle(
            handle_color=self.theme.text_primary,
            border_color=self.theme.border,
            groove_color=self.theme.border,
            fill_color=self.theme.accent,
            handle_size=18,
            groove_height=6,
            parent=self.uniform_velocity_slider,
        )
        self.uniform_velocity_slider.setStyle(self._velocity_slider_style)
        self._checkbox_style = SquareCheckBoxStyle(
            border_color=self.theme.border,
            fill_color=self.theme.accent,
            check_color=self.theme.text_primary,
            size=20,
            radius=4,
            parent=self.auto_updates_checkbox,
        )
        self.auto_updates_checkbox.setStyle(self._checkbox_style)

    @staticmethod
    def _scaled(value: int, scale: float, minimum: int = 1) -> int:
        return max(minimum, int(round(value * scale)))

    @classmethod
    def _normalize_ui_scale(cls, value: int | str) -> int:
        normalized = max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, int(value)))
        normalized = int(round(normalized / cls._UI_SCALE_STEP) * cls._UI_SCALE_STEP)
        return max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, normalized))

    def set_scale(self, scale: float) -> None:
        scale = max(0.5, min(2.0, float(scale)))
        margin = self._scaled(14, scale, 8)
        self._root_layout.setContentsMargins(margin, margin, margin, margin)
        self._root_layout.setSpacing(self._scaled(7, scale, 4))
        for layout in self._settings_card_layouts:
            layout.setContentsMargins(self._scaled(8, scale, 5), self._scaled(7, scale, 4), self._scaled(8, scale, 5), self._scaled(7, scale, 4))
            layout.setSpacing(self._scaled(6, scale, 4))
        pad = self._scaled(2, scale, 1)
        for layout in (self._mode_holder_layout, self._provider_holder_layout, *self._engine_choice_layouts):
            layout.setContentsMargins(pad, pad, pad, pad)
            layout.setSpacing(pad)
        if self._scale_slider_style is not None:
            self._scale_slider_style.set_metrics(handle_size=self._scaled(18, scale, 14), groove_height=self._scaled(6, scale, 4))
        if self._velocity_slider_style is not None:
            self._velocity_slider_style.set_metrics(handle_size=self._scaled(18, scale, 14), groove_height=self._scaled(6, scale, 4))
        if self._checkbox_style is not None:
            self._checkbox_style.set_metrics(size=self._scaled(20, scale, 14), radius=self._scaled(4, scale, 2))
        self.download_location_edit.setMinimumHeight(self._scaled(30, scale, 22))
        action_height = self._scaled(32, scale, 24)
        for button in (self.reset_engine_btn, self.download_path_browse_btn, self.check_updates_btn, self.reset_settings_btn):
            button.setMinimumHeight(action_height)
        for level, button in self.gpu_memory_buttons.items():
            base_width = 76 if level == 'balanced' else 52
            button.setMinimumWidth(self._scaled(base_width, scale, 40))
        width = QFontMetrics(self.ui_scale_value.font()).horizontalAdvance(f'{UI_SCALE_PERCENT_MAX}%')
        self.ui_scale_value.setMinimumWidth(width + self._scaled(12, scale, 8))

    def set_theme(self, theme: ThemePalette) -> None:
        self.theme = theme
        if self._scale_slider_style is not None:
            self._scale_slider_style.set_colors(handle_color=theme.text_primary, border_color=theme.border, groove_color=theme.border, fill_color=theme.accent)
        if self._velocity_slider_style is not None:
            self._velocity_slider_style.set_colors(handle_color=theme.text_primary, border_color=theme.border, groove_color=theme.border, fill_color=theme.accent)
        if self._checkbox_style is not None:
            self._checkbox_style.set_colors(border_color=theme.border, fill_color=theme.accent, check_color=theme.text_primary)
        self.ui_scale_slider.update()
        self.uniform_velocity_slider.update()
        self.auto_updates_checkbox.update()

    @staticmethod
    def _normalize_gpu_provider_preference(value: str | None) -> str:
        normalized = str(value or 'dml').strip().lower()
        return normalized if normalized in {'cuda', 'dml'} else 'dml'

    def _set_provider_ui(self, preference: str) -> None:
        normalized = self._normalize_gpu_provider_preference(preference)
        self._gpu_provider_preference = normalized
        self.cuda_provider_button.setChecked(normalized == 'cuda')
        self.dml_provider_button.setChecked(normalized == 'dml')

    def _on_device_clicked(self, checked: bool, preference: str) -> None:
        if not checked or self._updating:
            return
        self._selected_device = 'gpu' if preference == 'gpu' else 'cpu'
        self._refresh_performance_ui()
        self.devicePreferenceChanged.emit(self._selected_device)
        self._refresh_runtime_ui()

    def _on_provider_clicked(self, checked: bool, preference: str) -> None:
        if not checked or self._updating:
            return
        self._set_provider_ui(preference)
        self.gpuProviderPreferenceChanged.emit(self._gpu_provider_preference)

    def _set_performance_mode(self, device: str, mode: str | None) -> None:
        selected_device = 'gpu' if str(device or '').strip().lower() == 'gpu' else 'cpu'
        normalized = normalize_performance_mode(mode)
        if selected_device == 'gpu':
            self._gpu_performance_mode = normalized
        else:
            self._cpu_performance_mode = normalized
        self._refresh_performance_ui()

    def _performance_mode_for_selected_device(self) -> str:
        if self._selected_device == 'gpu':
            return self._gpu_performance_mode
        return self._cpu_performance_mode

    def _refresh_performance_ui(self) -> None:
        self.performance_label.setText('CPU usage')
        active_mode = self._performance_mode_for_selected_device()
        for button_mode, button in self.performance_buttons.items():
            button.setChecked(button_mode == active_mode)
        gpu_selected = self._selected_device == 'gpu'
        self.gpu_memory_container.setVisible(gpu_selected)
        active_level = gpu_memory_level_for_batch(self._gpu_batch_size, self._gpu_memory_max_batch)
        for level, button in self.gpu_memory_buttons.items():
            button.setChecked(level == active_level)
            button.setEnabled(gpu_selected and self._engine_controls_enabled)

    def _on_performance_clicked(self, checked: bool, mode: str) -> None:
        if not checked or self._updating:
            return
        device = self._selected_device
        self._set_performance_mode(device, mode)
        self.transcriptionPerformanceModeChanged.emit(device, self._performance_mode_for_selected_device())

    def _on_gpu_memory_clicked(self, checked: bool, level: str) -> None:
        if not checked or self._updating:
            return
        normalized = normalize_gpu_batch_size(self._gpu_memory_presets.get(level, self._gpu_memory_presets['balanced']))
        self._gpu_batch_size = normalized
        self._refresh_performance_ui()
        self.gpuMemoryUsageChanged.emit(normalized)

    def _on_pedals_clicked(self, checked: bool, enabled: bool) -> None:
        if not checked or self._updating:
            return
        self._engine_pedals_enabled = bool(enabled)
        self.enginePedalsEnabledChanged.emit(self._engine_pedals_enabled)

    def _on_velocity_mode_clicked(self, checked: bool, mode: str) -> None:
        if not checked or self._updating:
            return
        self._engine_velocity_mode = 'uniform' if str(mode).strip().lower() == 'uniform' else 'expressive'
        self._refresh_engine_ui()
        self.engineVelocityModeChanged.emit(self._engine_velocity_mode)

    def _on_uniform_velocity_changed(self, value: int) -> None:
        normalized = max(ENGINE_UNIFORM_VELOCITY_MIN, min(ENGINE_UNIFORM_VELOCITY_MAX, int(value)))
        self.uniform_velocity_value.setText(str(normalized))
        if not self._updating:
            self._pending_uniform_velocity = normalized

    def _on_uniform_velocity_released(self) -> None:
        if not self._updating and self._pending_uniform_velocity is not None:
            self.engineUniformVelocityChanged.emit(self._pending_uniform_velocity)
            self._pending_uniform_velocity = None

    def _refresh_engine_ui(self) -> None:
        enabled = bool(self._engine_controls_enabled)
        for button in self.performance_buttons.values():
            button.setEnabled(enabled)
        for button in self.gpu_memory_buttons.values():
            button.setEnabled(enabled and self._selected_device == 'gpu')
        for button in (
            self.pedals_off_button,
            self.pedals_on_button,
            self.expressive_velocity_button,
            self.uniform_velocity_button,
            self.reset_engine_btn,
        ):
            button.setEnabled(enabled)
        uniform_enabled = enabled and self._engine_velocity_mode == 'uniform'
        self.uniform_velocity_label.setEnabled(uniform_enabled)
        self.uniform_velocity_value.setEnabled(uniform_enabled)
        self.uniform_velocity_slider.setEnabled(uniform_enabled)
        self.refresh_cursor_state()

    def set_engine_controls_enabled(self, enabled: bool) -> None:
        self._engine_controls_enabled = bool(enabled)
        self._refresh_engine_ui()

    def set_device_controls_enabled(self, enabled: bool) -> None:
        self._device_controls_enabled = bool(enabled)
        self._refresh_runtime_ui()

    def _on_auto_updates_toggled(self, enabled: bool) -> None:
        if not self._updating:
            self.autoCheckUpdatesChanged.emit(bool(enabled))

    def _browse_download_location(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, 'Choose download location', self.download_location_edit.text().strip())
        if selected:
            self.download_location_edit.setText(selected)
            if not self._updating:
                self.downloadLocationChanged.emit(selected)

    def _on_ui_scale_changed(self, value: int) -> None:
        snapped = self._normalize_ui_scale(value)
        if snapped != value:
            self.ui_scale_slider.blockSignals(True)
            self.ui_scale_slider.setValue(snapped)
            self.ui_scale_slider.blockSignals(False)
        self.ui_scale_value.setText(f'{snapped}%')
        if not self._updating:
            self._pending_ui_scale_percent = snapped

    def _on_ui_scale_slider_released(self) -> None:
        if not self._updating and self._pending_ui_scale_percent is not None:
            self.uiScalePercentChanged.emit(self._pending_ui_scale_percent)
            self._pending_ui_scale_percent = None

    def set_values(
        self,
        *,
        device_preference: str,
        gpu_provider_preference: str,
        engine_pedals_enabled: bool,
        engine_velocity_mode: str,
        engine_uniform_velocity: int,
        cpu_performance_mode: str,
        gpu_performance_mode: str,
        gpu_batch_size: int,
        gpu_memory_max_batch: int,
        ui_scale_percent: int,
        download_location: str,
        auto_check_updates: bool,
    ) -> None:
        self._updating = True
        self._selected_device = 'gpu' if str(device_preference).strip().lower() == 'gpu' else 'cpu'
        self.cpu_button.setChecked(self._selected_device == 'cpu')
        self.gpu_button.setChecked(self._selected_device == 'gpu')
        self._set_provider_ui(gpu_provider_preference)
        self._engine_pedals_enabled = bool(engine_pedals_enabled)
        self.pedals_off_button.setChecked(not self._engine_pedals_enabled)
        self.pedals_on_button.setChecked(self._engine_pedals_enabled)
        self._engine_velocity_mode = 'uniform' if str(engine_velocity_mode).strip().lower() == 'uniform' else 'expressive'
        self.expressive_velocity_button.setChecked(self._engine_velocity_mode == 'expressive')
        self.uniform_velocity_button.setChecked(self._engine_velocity_mode == 'uniform')
        velocity = max(ENGINE_UNIFORM_VELOCITY_MIN, min(ENGINE_UNIFORM_VELOCITY_MAX, int(engine_uniform_velocity)))
        self.uniform_velocity_slider.setValue(velocity)
        self.uniform_velocity_value.setText(str(velocity))
        self._pending_uniform_velocity = None
        self._cpu_performance_mode = normalize_performance_mode(cpu_performance_mode)
        self._gpu_performance_mode = normalize_performance_mode(gpu_performance_mode)
        self._gpu_memory_max_batch = normalize_gpu_memory_max_batch(gpu_memory_max_batch)
        self._gpu_memory_presets = gpu_memory_presets(self._gpu_memory_max_batch)
        self._gpu_batch_size = normalize_gpu_batch_size(gpu_batch_size)
        self._refresh_performance_ui()
        scale = self._normalize_ui_scale(ui_scale_percent)
        self.ui_scale_slider.setValue(scale)
        self.ui_scale_value.setText(f'{scale}%')
        self._pending_ui_scale_percent = None
        self.download_location_edit.setText(str(download_location or '').strip())
        self.auto_updates_checkbox.setChecked(bool(auto_check_updates))
        self._updating = False
        self._refresh_engine_ui()
        self._refresh_runtime_ui()

    def set_runtime_status(self, *, checking: bool, runtime_available: bool, cuda_available: bool, dml_available: bool, active_provider: str) -> None:
        self._runtime_checking = bool(checking)
        self._runtime_available = bool(runtime_available)
        self._cuda_available = bool(cuda_available)
        self._dml_available = bool(dml_available)
        self._active_provider = str(active_provider or 'CPU')
        self._refresh_runtime_ui()

    def _provider_ready(self) -> bool:
        if self._gpu_provider_preference == 'cuda':
            return self._cuda_available
        if self._gpu_provider_preference == 'dml':
            return self._dml_available
        return False

    def _missing_provider_label(self) -> str:
        if self._gpu_provider_preference == 'cuda':
            return 'GPU selected, but the CUDA runtime pack is not installed or active.'
        if self._gpu_provider_preference == 'dml':
            return 'GPU selected, but the DirectML runtime pack is not installed or active.'
        return RUNTIME_LABEL_GPU_NO_PACK

    def _provider_display_name(self) -> str:
        return 'CUDA' if self._gpu_provider_preference == 'cuda' else 'DirectML'

    def _refresh_runtime_ui(self) -> None:
        provider_ready = self._provider_ready()
        gpu_selected = self._selected_device == 'gpu'
        device_controls_enabled = self._device_controls_enabled and not self._runtime_checking
        self.cpu_button.setEnabled(device_controls_enabled)
        self.gpu_button.setEnabled(device_controls_enabled)
        provider_controls_enabled = gpu_selected and device_controls_enabled
        self.provider_label.setEnabled(provider_controls_enabled)
        self.provider_holder.setEnabled(provider_controls_enabled)
        for button in (self.cuda_provider_button, self.dml_provider_button):
            button.setEnabled(provider_controls_enabled)
        if self._runtime_checking:
            text = RUNTIME_LABEL_CHECKING
        elif not self._runtime_available:
            text = RUNTIME_LABEL_MISSING
        elif gpu_selected:
            active = self._active_provider.strip()
            if active.lower().startswith('validation pending'):
                text = f'Validating {self._provider_display_name()} acceleration support...' if provider_ready else self._missing_provider_label()
            elif not provider_ready:
                text = self._missing_provider_label()
            elif active.upper().startswith('CPU'):
                text = RUNTIME_LABEL_GPU_UNAVAILABLE
            elif active.lower() == 'cuda':
                text = RUNTIME_LABEL_CUDA_ACTIVE
            elif active.lower() in {'dml', 'directml'}:
                text = RUNTIME_LABEL_DML_ACTIVE
            else:
                text = f'{self._provider_display_name()} acceleration is active.'
        elif self._cuda_available or self._dml_available:
            text = RUNTIME_LABEL_CPU_WITH_GPU_PACK
        else:
            text = RUNTIME_LABEL_CPU_NO_GPU_PACK
        self.runtime_label.setText(text)
        self.refresh_cursor_state()
