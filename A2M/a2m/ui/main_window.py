from __future__ import annotations
import re
from collections.abc import Callable
from pathlib import Path
from PySide6.QtCore import QEvent, QEasingCurve, QPropertyAnimation, QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QGuiApplication, QIcon, QPainter, QPalette, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget
from a2m.core.constants import APP_NAME, APP_VERSION, UI_SCALE_PERCENT_MAX, UI_SCALE_PERCENT_MIN
from .interaction import is_pointer_control, sync_pointer_cursor
from .settings_panel import SettingsPanel
from .theme import DARK_THEME, ThemePalette, build_stylesheet
from .windows_titlebar import apply_windows_titlebar_theme

class MainWindow(QMainWindow):
    chooseFileRequested = Signal()
    convertRequested = Signal()
    stopRequested = Signal()
    openDownloadsRequested = Signal()
    officialPageRequested = Signal()
    themeModeChanged = Signal(str)
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

    def __init__(self, theme: ThemePalette=DARK_THEME, icon_path: Path | None=None, theme_mode: str='dark', ui_scale_percent: int=100) -> None:
        super().__init__()
        self.theme = theme
        self._theme_mode = 'light' if theme_mode == 'light' else 'dark'
        self._ui_scale_percent = self._normalize_ui_scale_percent(ui_scale_percent)
        self._close_handler: Callable[[], bool] | None = None
        self._file_name_full = 'No file chosen'
        self._file_path_full = ''
        self._file_path_placeholder = 'No file path selected'
        self._model_status_full = 'Transcription model: checking...'
        self._settings_visible = False
        self._render_scale = 1.0
        self._base_width = 608
        self._base_height = 400
        self._base_settings_width = 284
        self._settings_min_width = 260
        self._settings_target_width = self._base_settings_width
        self._settings_animation_expected_end_width: int | None = None
        self.setWindowTitle(APP_NAME)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        if hasattr(Qt, 'MSWindowsFixedSizeDialogHint'):
            self.setWindowFlag(Qt.MSWindowsFixedSizeDialogHint, True)
        if icon_path and icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self._build_ui()
        self._apply_window_layout()
        self.set_theme_mode(self._theme_mode)
        self._apply_interaction_cursors()
        self._install_wheel_guards()

    def _build_ui(self) -> None:
        root, outer, content_row = self._build_root_layout()
        self._build_main_column(root, content_row)
        self._build_settings_panel(root, content_row)
        self._build_footer(root, outer)
        self._connect_settings_signals()
        self._refresh_file_labels()

    def _build_root_layout(self) -> tuple[QWidget, QVBoxLayout, QHBoxLayout]:
        root = QWidget(self)
        root.setObjectName('a2mRoot')
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 12, 6, 6)
        outer.setSpacing(8)
        self._outer_layout = outer
        content_row = QHBoxLayout()
        content_row.setContentsMargins(12, 0, 12, 0)
        content_row.setSpacing(8)
        self._content_row_layout = content_row
        outer.addLayout(content_row, 1)
        return (root, outer, content_row)

    def _build_main_column(self, root: QWidget, content_row: QHBoxLayout) -> None:
        self.main_column = QWidget(root)
        self.main_column.setObjectName('mainColumn')
        main_layout = QVBoxLayout(self.main_column)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        self._main_layout = main_layout
        content_row.addWidget(self.main_column, 1)
        header = QFrame(self.main_column)
        header.setObjectName('card')
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(2)
        self._header_layout = header_layout
        self.title_label = QLabel(APP_NAME, header)
        self.title_label.setObjectName('title')
        self.subtitle_label = QLabel('Transcribe a local audio file into MIDI.', header)
        self.subtitle_label.setObjectName('subtitle')
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)
        main_layout.addWidget(header)
        file_card = QFrame(self.main_column)
        file_card.setObjectName('card')
        file_layout = QHBoxLayout(file_card)
        file_layout.setContentsMargins(12, 10, 12, 10)
        file_layout.setSpacing(8)
        self._file_layout = file_layout
        file_text = QWidget(file_card)
        file_text.setObjectName('cardContent')
        file_text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        file_text.setMinimumWidth(0)
        self.file_text_container = file_text
        file_text_layout = QVBoxLayout(file_text)
        file_text_layout.setContentsMargins(0, 0, 0, 0)
        file_text_layout.setSpacing(4)
        self._file_text_layout = file_text_layout
        file_title = QLabel('Audio file', file_text)
        file_title.setObjectName('caption')
        self.file_label = QLabel(self._file_name_full, file_text)
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.file_label.setMinimumWidth(0)
        self.file_path_label = QLabel(self._file_path_full, file_text)
        self.file_path_label.setObjectName('muted')
        self.file_path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.file_path_label.setMinimumWidth(0)
        self.model_status_label = QLabel(self._model_status_full, file_text)
        self.model_status_label.setObjectName('muted')
        self.model_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.model_status_label.setMinimumWidth(0)
        file_text_layout.addWidget(file_title)
        file_text_layout.addWidget(self.file_label)
        file_text_layout.addWidget(self.file_path_label)
        file_text_layout.addWidget(self.model_status_label)
        file_layout.addWidget(file_text, 1)
        self.file_choose_btn = QPushButton('Choose Audio', file_card)
        self.file_choose_btn.setObjectName('actionButton')
        self.file_choose_btn.setMinimumWidth(120)
        self.file_choose_btn.setMinimumHeight(60)
        self.file_choose_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.file_choose_btn.clicked.connect(self.chooseFileRequested.emit)
        file_layout.addWidget(self.file_choose_btn, 0)
        main_layout.addWidget(file_card)
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self._action_row_layout = action_row
        self.convert_btn = QPushButton('Convert to MIDI', self.main_column)
        self.convert_btn.setObjectName('convertButton')
        self.convert_btn.clicked.connect(self.convertRequested.emit)
        self.stop_btn = QPushButton('STOP', self.main_column)
        self.stop_btn.setObjectName('stopButton')
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        self.stop_btn.setEnabled(False)
        action_row.addWidget(self.convert_btn, 1)
        action_row.addWidget(self.stop_btn)
        main_layout.addLayout(action_row)
        self.progress_bar = QProgressBar(self.main_column)
        self.progress_bar.setObjectName('downloadProgress')
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('0.00%')
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setFixedHeight(24)
        main_layout.addWidget(self.progress_bar)
        self.console = QPlainTextEdit(self.main_column)
        self.console.setObjectName('consoleOutput')
        self.console.setReadOnly(True)
        self.console.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.console.setPlaceholderText('Console output')
        self._apply_console_placeholder_color()
        self.console.setMinimumHeight(86)
        self.console.setMaximumHeight(112)
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(self.console, 0)

    def _build_settings_panel(self, root: QWidget, content_row: QHBoxLayout) -> None:
        self.settings_container = QFrame(root)
        self.settings_container.setObjectName('settingsPanel')
        settings_container_layout = QVBoxLayout(self.settings_container)
        settings_container_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_scroll = QScrollArea(self.settings_container)
        self.settings_scroll.setObjectName('settingsScroll')
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.settings_panel = SettingsPanel(self.theme, self.settings_scroll)
        self.settings_scroll.setWidget(self.settings_panel)
        settings_container_layout.addWidget(self.settings_scroll)
        self.settings_container.setMaximumWidth(0)
        self.settings_container.setMinimumWidth(0)
        content_row.addWidget(self.settings_container, 0)
        self.settings_animation = QPropertyAnimation(self.settings_container, b'maximumWidth', self)
        self.settings_animation.setDuration(150)
        self.settings_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.settings_animation.valueChanged.connect(self._on_settings_animation_value_changed)
        self.settings_animation.finished.connect(self._on_settings_animation_finished)

    def _build_footer(self, root: QWidget, outer: QVBoxLayout) -> None:
        footer = QFrame(root)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(2, 0, 2, 0)
        footer_layout.setSpacing(8)
        self._footer_layout = footer_layout
        self.theme_toggle_btn = QPushButton('', footer)
        self.theme_toggle_btn.setObjectName('footerIcon')
        self.theme_toggle_btn.setFlat(True)
        self.theme_toggle_btn.setIconSize(QSize(20, 20))
        self.theme_toggle_btn.clicked.connect(self._on_theme_toggle_clicked)
        footer_layout.addWidget(self.theme_toggle_btn, 0, Qt.AlignLeft)
        self.settings_toggle_btn = QPushButton('Show settings', footer)
        self.settings_toggle_btn.setObjectName('footerLink')
        self.settings_toggle_btn.setFlat(True)
        self.settings_toggle_btn.clicked.connect(lambda: self.set_settings_visible(not self._settings_visible))
        footer_layout.addWidget(self.settings_toggle_btn, 0, Qt.AlignLeft)
        footer_layout.addStretch(1)
        self.downloads_btn = QPushButton('Open downloads folder', footer)
        self.downloads_btn.setObjectName('footerLink')
        self.downloads_btn.setFlat(True)
        self.downloads_btn.clicked.connect(self.openDownloadsRequested.emit)
        footer_layout.addWidget(self.downloads_btn, 0, Qt.AlignRight)
        self.official_btn = QPushButton('Official website', footer)
        self.official_btn.setObjectName('footerLink')
        self.official_btn.setFlat(True)
        self.official_btn.clicked.connect(self.officialPageRequested.emit)
        footer_layout.addWidget(self.official_btn, 0, Qt.AlignRight)
        self.version_label = QLabel(f'v{APP_VERSION}', footer)
        self.version_label.setObjectName('footerVersion')
        footer_layout.addWidget(self.version_label, 0, Qt.AlignRight)
        outer.addWidget(footer, 0)

    def _connect_settings_signals(self) -> None:
        self.settings_panel.devicePreferenceChanged.connect(self.devicePreferenceChanged.emit)
        self.settings_panel.conversionMethodChanged.connect(self.conversionMethodChanged.emit)
        self.settings_panel.modernAdaptiveThresholdsChanged.connect(self.modernAdaptiveThresholdsChanged.emit)
        self.settings_panel.modernInputNormalizationChanged.connect(self.modernInputNormalizationChanged.emit)
        self.settings_panel.modernSmartOverlapStitchingChanged.connect(self.modernSmartOverlapStitchingChanged.emit)
        self.settings_panel.modernAutoCalibrationChanged.connect(self.modernAutoCalibrationChanged.emit)
        self.settings_panel.modernOnsetThresholdChanged.connect(self.modernOnsetThresholdChanged.emit)
        self.settings_panel.modernOffsetThresholdChanged.connect(self.modernOffsetThresholdChanged.emit)
        self.settings_panel.modernFrameThresholdChanged.connect(self.modernFrameThresholdChanged.emit)
        self.settings_panel.modernPedalOffsetThresholdChanged.connect(self.modernPedalOffsetThresholdChanged.emit)
        self.settings_panel.modernCleanupScaleChanged.connect(self.modernCleanupScaleChanged.emit)
        self.settings_panel.modernPedalClusterScaleChanged.connect(self.modernPedalClusterScaleChanged.emit)
        self.settings_panel.modernAlignmentGateScaleChanged.connect(self.modernAlignmentGateScaleChanged.emit)
        self.settings_panel.gpuProviderPreferenceChanged.connect(self.gpuProviderPreferenceChanged.emit)
        self.settings_panel.gpuBatchSizeChanged.connect(self.gpuBatchSizeChanged.emit)
        self.settings_panel.uiScalePercentChanged.connect(self.uiScalePercentChanged.emit)
        self.settings_panel.downloadLocationChanged.connect(self.downloadLocationChanged.emit)
        self.settings_panel.autoCheckUpdatesChanged.connect(self.autoCheckUpdatesChanged.emit)
        self.settings_panel.checkUpdatesNowRequested.connect(self.checkUpdatesNowRequested.emit)
        self.settings_panel.resetSettingsRequested.connect(self.resetSettingsRequested.emit)

    @staticmethod
    def _set_widget_cursor(widget: QWidget) -> None:
        sync_pointer_cursor(widget)

    def _iter_main_action_controls(self) -> tuple[QWidget, ...]:
        return (
            self.file_choose_btn,
            self.convert_btn,
            self.stop_btn,
            self.theme_toggle_btn,
            self.settings_toggle_btn,
            self.downloads_btn,
            self.official_btn,
        )

    def _apply_interaction_cursors(self) -> None:
        for widget in self._iter_main_action_controls():
            self._set_widget_cursor(widget)

    def refresh_cursor_state(self) -> None:
        self.setCursor(Qt.ArrowCursor)
        root = self.centralWidget()
        if root is not None:
            root.setCursor(Qt.ArrowCursor)
        self._apply_interaction_cursors()
        self.settings_panel.refresh_cursor_state()

    def _install_wheel_guards(self) -> None:
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

    @staticmethod
    def _is_interactive_control(watched: object) -> bool:
        return is_pointer_control(watched)

    def eventFilter(self, watched, event):
        if event.type() in {QEvent.EnabledChange, QEvent.Show, QEvent.Hide, QEvent.Enter, QEvent.HoverEnter, QEvent.HoverMove, QEvent.StyleChange, QEvent.Polish}:
            if isinstance(watched, QWidget) and self._is_interactive_control(watched):
                self._set_widget_cursor(watched)
        if event.type() == QEvent.Wheel:
            if isinstance(watched, QSlider):
                scroll_bar = self.settings_scroll.verticalScrollBar() if hasattr(self, 'settings_scroll') else None
                if scroll_bar is not None:
                    delta_y = 0
                    try:
                        delta_y = int(event.angleDelta().y())
                    except Exception:
                        delta_y = 0
                    if delta_y == 0:
                        try:
                            delta_y = int(event.pixelDelta().y())
                        except Exception:
                            delta_y = 0
                    if delta_y != 0:
                        step = max(1, int(scroll_bar.singleStep() or 20))
                        lines = float(delta_y) / 120.0
                        scroll_offset = int(round(lines * step * 3))
                        scroll_bar.setValue(int(scroll_bar.value() - scroll_offset))
                        event.accept()
                        return True
        return super().eventFilter(watched, event)

    def _restore_cursor_state_after_modal_deferred(self) -> None:
        self._clear_override_cursors()
        self.setCursor(Qt.ArrowCursor)
        self.unsetCursor()
        self.refresh_cursor_state()

    def restore_cursor_state_after_modal(self) -> None:
        self._clear_override_cursors()
        self.setCursor(Qt.ArrowCursor)
        self.unsetCursor()
        self.refresh_cursor_state()
        QTimer.singleShot(0, self._restore_cursor_state_after_modal_deferred)

    @staticmethod
    def _clear_override_cursors() -> None:
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    def _build_theme_icon(self, mode: str) -> QIcon:
        size = max(14, int(self.theme_toggle_btn.iconSize().width()))
        screen = QGuiApplication.primaryScreen()
        dpr = float(screen.devicePixelRatio()) if screen is not None else 1.0
        px = int(round(size * dpr))
        icon = QPixmap(px, px)
        icon.setDevicePixelRatio(dpr)
        icon.fill(Qt.transparent)
        icon_color = QColor(self.theme.text_primary)
        painter = QPainter(icon)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(icon_color, max(1.1, size * 0.1), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        center = QPointF(size * 0.5, size * 0.5)
        if mode == 'sun':
            orbit_radius = size * 0.22
            inner_ray = size * 0.34
            outer_ray = size * 0.46
            painter.drawEllipse(QPointF(center.x(), center.y()), orbit_radius, orbit_radius)
            for direction in (QPointF(1.0, 0.0), QPointF(-1.0, 0.0), QPointF(0.0, 1.0), QPointF(0.0, -1.0), QPointF(0.707, 0.707), QPointF(-0.707, -0.707), QPointF(0.707, -0.707), QPointF(-0.707, 0.707)):
                start = QPointF(center.x() + direction.x() * inner_ray, center.y() + direction.y() * inner_ray)
                end = QPointF(center.x() + direction.x() * outer_ray, center.y() + direction.y() * outer_ray)
                painter.drawLine(start, end)
        else:
            moon_radius = size * 0.38
            painter.setBrush(icon_color)
            painter.drawEllipse(center, moon_radius, moon_radius)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.setPen(Qt.NoPen)
            painter.setBrush(Qt.transparent)
            painter.drawEllipse(QPointF(center.x() + size * 0.17, center.y() - size * 0.1), size * 0.34, size * 0.34)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.end()
        return QIcon(icon)

    def _refresh_theme_toggle_icon(self) -> None:
        if self._theme_mode == 'dark':
            self.theme_toggle_btn.setIcon(self._build_theme_icon('moon'))
            self.theme_toggle_btn.setToolTip('Switch to light mode')
        else:
            self.theme_toggle_btn.setIcon(self._build_theme_icon('sun'))
            self.theme_toggle_btn.setToolTip('Switch to dark mode')

    @staticmethod
    def _available_screen_geometry():
        screen = QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else None

    @staticmethod
    def _scaled(value: int, scale: float, minimum: int=1) -> int:
        return max(minimum, int(round(value * scale)))

    @staticmethod
    def _normalize_ui_scale_percent(value: int | str | None) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = 100
        return max(UI_SCALE_PERCENT_MIN, min(UI_SCALE_PERCENT_MAX, parsed))

    @staticmethod
    def _fit_factor_for_bounds(width: int, height: int, max_width: int, max_height: int) -> float:
        if width <= 0 or height <= 0:
            return 1.0
        if max_width <= 0 or max_height <= 0:
            return 1.0
        return min(1.0, max_width / width, max_height / height)

    def _apply_scaled_metrics(self, scale: float) -> None:
        side_margin = self._scaled(6, scale, 3)
        self._outer_layout.setContentsMargins(side_margin, self._scaled(12, scale, 6), side_margin, self._scaled(6, scale, 3))
        self._outer_layout.setSpacing(self._scaled(8, scale, 4))
        content_side_margin = self._scaled(12, scale, 6)
        self._content_row_layout.setContentsMargins(content_side_margin, 0, content_side_margin, 0)
        self._content_row_layout.setSpacing(self._scaled(8, scale, 4))
        self._main_layout.setSpacing(self._scaled(8, scale, 4))
        margin_x = self._scaled(12, scale, 6)
        margin_y = self._scaled(10, scale, 5)
        self._header_layout.setContentsMargins(margin_x, margin_y, margin_x, margin_y)
        self._header_layout.setSpacing(self._scaled(2, scale, 1))
        self._file_layout.setContentsMargins(margin_x, margin_y, margin_x, margin_y)
        self._file_layout.setSpacing(self._scaled(8, scale, 4))
        self._file_text_layout.setSpacing(self._scaled(4, scale, 2))
        self._action_row_layout.setSpacing(self._scaled(6, scale, 3))
        self._footer_layout.setContentsMargins(self._scaled(2, scale, 1), 0, self._scaled(2, scale, 1), 0)
        self._footer_layout.setSpacing(self._scaled(8, scale, 4))
        self.file_choose_btn.setMinimumWidth(self._scaled(120, scale, 96))
        self.file_choose_btn.setMinimumHeight(self._scaled(60, scale, 46))
        self.console.setMinimumHeight(self._scaled(86, scale, 68))
        self.console.setMaximumHeight(self._scaled(112, scale, 88))
        self.progress_bar.setFixedHeight(self._scaled(26, scale, 18))
        icon_px = self._scaled(20, scale, 18)
        self.theme_toggle_btn.setIconSize(QSize(icon_px, icon_px))
        footer_pt = max(7.5, round(9.5 * scale, 1))
        self.version_label.setStyleSheet(f'color: {self.theme.text_secondary}; font: 650 {footer_pt:.1f}pt "Segoe UI";')
        self.settings_panel.setMinimumWidth(self._scaled(208, scale, 162))
        self.settings_panel.set_scale(scale)
        self._settings_min_width = max(self.settings_panel.minimumSizeHint().width(), self.settings_panel.minimumWidth())

    def _compute_dimensions(self, scale: float) -> tuple[int, int]:
        total_width = int(round(self._base_width * scale))
        height = int(round(self._base_height * scale))
        return (total_width, height)

    def _compute_settings_target_width(self, scale: float, window_width: int) -> int:
        desired = max(int(round(self._base_settings_width * scale)), self._settings_min_width)
        outer_margins = self._outer_layout.contentsMargins()
        row_margins = self._content_row_layout.contentsMargins()
        content_width = max(1, window_width - outer_margins.left() - outer_margins.right() - row_margins.left() - row_margins.right())
        reserve_main = self._scaled(235, scale, 150)
        max_overlay_by_reserve = max(0, content_width - reserve_main)
        max_overlay_by_ratio = int(round(content_width * 0.86))
        max_overlay = max(self._settings_min_width, min(max_overlay_by_reserve, max_overlay_by_ratio))
        return min(desired, max_overlay)

    def _set_settings_container_width(self, width: int) -> None:
        clamped = max(0, int(width))
        self.settings_container.setMinimumWidth(clamped)
        self.settings_container.setMaximumWidth(clamped)

    def _on_settings_animation_finished(self) -> None:
        if self._settings_animation_expected_end_width is None:
            return
        end_width = int(self._settings_animation_expected_end_width)
        self._settings_animation_expected_end_width = None
        if self._settings_visible:
            self._set_settings_container_width(end_width)
        else:
            self._set_settings_container_width(0)

    def _on_settings_animation_value_changed(self, value: object) -> None:
        try:
            width = max(0, int(round(float(value))))
        except Exception:
            width = max(0, int(self.settings_container.maximumWidth()))
        self.settings_container.setMinimumWidth(width)
        self._refresh_file_labels()

    def _on_theme_toggle_clicked(self) -> None:
        next_mode = 'light' if self._theme_mode == 'dark' else 'dark'
        self.themeModeChanged.emit(next_mode)

    def set_theme(self, theme: ThemePalette, theme_mode: str) -> None:
        self.theme = theme
        self.settings_panel.set_theme(theme)
        self._apply_window_layout()
        self.set_theme_mode(theme_mode)
        self._apply_interaction_cursors()
        self._apply_console_placeholder_color()

    def _apply_console_placeholder_color(self) -> None:
        palette = self.console.palette()
        palette.setColor(QPalette.PlaceholderText, QColor(self.theme.text_secondary))
        self.console.setPalette(palette)

    def _resolve_render_scale(self) -> float:
        requested_scale = self._normalize_ui_scale_percent(self._ui_scale_percent) / 100.0
        geometry = self._available_screen_geometry()
        render_scale = requested_scale
        if geometry is not None:
            max_width = max(1, int(geometry.width() * 0.92))
            max_height = max(1, int(geometry.height() * 0.92))
            target_width, target_height = self._compute_dimensions(render_scale)
            fit = self._fit_factor_for_bounds(target_width, target_height, max_width, max_height)
            render_scale = max(0.55, render_scale * fit)
        return render_scale

    def _apply_window_layout(self) -> None:
        geometry = self._available_screen_geometry()
        self._render_scale = self._resolve_render_scale()
        self.setStyleSheet(build_stylesheet(self.theme, self._render_scale))
        self._apply_scaled_metrics(self._render_scale)
        width, height = self._compute_dimensions(self._render_scale)
        if geometry is not None:
            width = min(width, max(1, int(geometry.width() * 0.92)))
            height = min(height, max(1, int(geometry.height() * 0.92)))
        width = max(1, width)
        height = max(1, height)
        self.setFixedSize(width, height)
        self.resize(width, height)
        self._settings_target_width = self._compute_settings_target_width(self._render_scale, self.width())
        self._settings_animation_expected_end_width = None
        if self._settings_visible:
            self.settings_animation.stop()
            self._set_settings_container_width(self._settings_target_width)
        else:
            self._set_settings_container_width(0)

    def set_theme_mode(self, mode: str) -> None:
        self._theme_mode = 'light' if mode == 'light' else 'dark'
        self.theme_toggle_btn.setText('')
        self._refresh_theme_toggle_icon()
        self._apply_windows_titlebar_theme()

    def _apply_windows_titlebar_theme(self) -> None:
        apply_windows_titlebar_theme(self, dark=self._theme_mode != 'light')

    def current_theme_mode(self) -> str:
        return self._theme_mode

    def set_close_handler(self, handler: Callable[[], bool]) -> None:
        self._close_handler = handler

    def closeEvent(self, event) -> None:
        if self._close_handler and (not self._close_handler()):
            event.ignore()
            return
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_windows_titlebar_theme)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_file_labels()

    def _refresh_file_labels(self) -> None:
        name_metrics = QFontMetrics(self.file_label.font())
        path_metrics = QFontMetrics(self.file_path_label.font())
        model_metrics = QFontMetrics(self.model_status_label.font())
        container_width = max(self.file_label.width(), self.file_path_label.width())
        container_width = max(container_width, self.file_text_container.width())
        width = max(120, container_width - 8)
        self.file_label.setText(self._elide_text(name_metrics, self._file_name_full, Qt.ElideRight, width))
        full_path = self._file_path_full.strip() or self._file_path_placeholder
        path_text = self._elide_text(path_metrics, full_path, Qt.ElideMiddle, width)
        self.file_path_label.setText(path_text)
        self.file_path_label.setVisible(True)
        status_lines = self._model_status_full.splitlines() or ['']
        status_text = '\n'.join((self._elide_text(model_metrics, line, Qt.ElideRight, width) for line in status_lines))
        self.model_status_label.setText(status_text)

    @staticmethod
    def _elide_text(metrics: QFontMetrics, text: str, mode: Qt.TextElideMode, width: int) -> str:
        return metrics.elidedText(str(text or ''), mode, int(width))

    def set_file_info(self, display_name: str, full_path: str) -> None:
        self._file_name_full = display_name or 'No file chosen'
        self._file_path_full = full_path or ''
        self._refresh_file_labels()

    def set_model_status(self, *, ready: bool) -> None:
        if ready:
            self._model_status_full = 'Transcription model: Ready'
        else:
            self._model_status_full = 'Transcription model: REQUIRED (download once)'
        self.model_status_label.setText(self._model_status_full)
        self._refresh_file_labels()
        self.model_status_label.update()
        QTimer.singleShot(0, self._refresh_file_labels)

    def set_console_text(self, text: str) -> None:
        self.console.setPlainText(str(text or ''))
        self.console.moveCursor(QTextCursor.End)

    def set_progress(self, percent: int | float, text: str | None=None) -> None:
        value = max(0.0, min(100.0, float(percent)))
        self.progress_bar.setValue(int(round(value)))
        if text:
            formatted = re.sub(r'(\d+(?:\.\d+)?)%', lambda m: f"{float(m.group(1)):.2f}%", str(text))
            self.progress_bar.setFormat(formatted)
        else:
            self.progress_bar.setFormat(f'{value:.2f}%')

    def set_controls_state(self, *, file_enabled: bool, convert_enabled: bool, stop_enabled: bool) -> None:
        self.file_choose_btn.setEnabled(file_enabled)
        self.convert_btn.setEnabled(convert_enabled)
        self.stop_btn.setEnabled(stop_enabled)
        for button in (self.file_choose_btn, self.convert_btn, self.stop_btn):
            self._set_widget_cursor(button)

    def set_settings_values(self, *, device_preference: str, conversion_method: str, modern_adaptive_thresholds_enabled: bool, modern_input_normalization_enabled: bool, modern_smart_overlap_stitching_enabled: bool, modern_auto_calibration_enabled: bool, modern_cleanup_scale: float, modern_pedal_cluster_scale: float, modern_alignment_gate_scale: float, modern_manual_onset_threshold: float, modern_manual_offset_threshold: float, modern_manual_frame_threshold: float, modern_manual_pedal_offset_threshold: float, gpu_provider_preference: str, gpu_batch_size: int, ui_scale_percent: int, download_location: str, auto_check_updates: bool) -> None:
        self.settings_panel.set_values(device_preference=device_preference, conversion_method=conversion_method, modern_adaptive_thresholds_enabled=modern_adaptive_thresholds_enabled, modern_input_normalization_enabled=modern_input_normalization_enabled, modern_smart_overlap_stitching_enabled=modern_smart_overlap_stitching_enabled, modern_auto_calibration_enabled=modern_auto_calibration_enabled, modern_cleanup_scale=modern_cleanup_scale, modern_pedal_cluster_scale=modern_pedal_cluster_scale, modern_alignment_gate_scale=modern_alignment_gate_scale, modern_manual_onset_threshold=modern_manual_onset_threshold, modern_manual_offset_threshold=modern_manual_offset_threshold, modern_manual_frame_threshold=modern_manual_frame_threshold, modern_manual_pedal_offset_threshold=modern_manual_pedal_offset_threshold, gpu_provider_preference=gpu_provider_preference, gpu_batch_size=gpu_batch_size, ui_scale_percent=ui_scale_percent, download_location=download_location, auto_check_updates=auto_check_updates)

    def set_ui_scale_percent(self, value: int) -> None:
        normalized = self._normalize_ui_scale_percent(value)
        if normalized == self._ui_scale_percent:
            return
        self._ui_scale_percent = normalized
        self._apply_window_layout()
        self._refresh_file_labels()

    def set_runtime_status(self, *, checking: bool, runtime_available: bool, cuda_available: bool, dml_available: bool, active_provider: str) -> None:
        self.settings_panel.set_runtime_status(checking=checking, runtime_available=runtime_available, cuda_available=cuda_available, dml_available=dml_available, active_provider=active_provider)

    def set_settings_visible(self, visible: bool, *, animated: bool=True) -> None:
        visible = bool(visible)
        self._settings_visible = visible
        self.settings_toggle_btn.setText('Hide settings' if visible else 'Show settings')
        self._settings_animation_expected_end_width = None
        self.settings_animation.stop()
        self._settings_target_width = self._compute_settings_target_width(self._render_scale, self.width())
        end_width = self._settings_target_width if visible else 0
        self.settings_container.setMinimumWidth(0)
        if animated:
            self.settings_animation.setStartValue(self.settings_container.maximumWidth())
            self.settings_animation.setEndValue(end_width)
            self._settings_animation_expected_end_width = int(end_width)
            self.settings_animation.start()
        else:
            self._set_settings_container_width(end_width)
        self._refresh_file_labels()

    def is_settings_visible(self) -> bool:
        return self._settings_visible
