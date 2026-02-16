from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ThemePalette:
    mode: str
    app_bg: str
    panel_bg: str
    border: str
    text_primary: str
    text_secondary: str
    accent: str
    accent_hover: str
    danger: str
    danger_hover: str
    success: str
    disabled_bg: str
    disabled_fg: str
DARK_THEME = ThemePalette(mode='dark', app_bg='#0A0A0B', panel_bg='#141416', border='#2A2A2D', text_primary='#F4F4F5', text_secondary='#B7B7BC', accent='#D20F39', accent_hover='#F03A5F', danger='#C51E3A', danger_hover='#D94A63', success='#22C55E', disabled_bg='#202024', disabled_fg='#8C8C93')
LIGHT_THEME = ThemePalette(mode='light', app_bg='#ECEDEF', panel_bg='#FAFAFB', border='#D1D3D8', text_primary='#1B1F2A', text_secondary='#4B5161', accent='#C51E3A', accent_hover='#D94A63', danger='#B71C38', danger_hover='#CD4A63', success='#1E9A4B', disabled_bg='#E6E8ED', disabled_fg='#7A8090')

def get_theme(mode: str | None) -> ThemePalette:
    if str(mode or '').strip().lower() == 'light':
        return LIGHT_THEME
    return DARK_THEME

def _scaled(value: float, scale: float, minimum: int=1) -> int:
    return max(minimum, int(round(value * scale)))

def _scaled_pt(value: float, scale: float, minimum: float=7.0) -> float:
    return max(minimum, round(value * scale, 1))

def build_stylesheet(theme: ThemePalette, ui_scale: float=1.0) -> str:
    scale = max(0.5, min(2.0, float(ui_scale)))
    frame_radius = _scaled(8, scale, 4)
    button_radius = _scaled(6, scale, 3)
    widget_font = _scaled_pt(10.0, scale, 8.0)
    title_font = _scaled_pt(11.0, scale, 8.5)
    subtitle_font = _scaled_pt(8.5, scale, 7.0)
    button_font = _scaled_pt(9.5, scale, 7.5)
    console_font = _scaled_pt(9.1, scale, 7.2)
    button_bold_font = _scaled_pt(10.0, scale, 8.0)
    footer_font = _scaled_pt(9.5, scale, 7.5)
    settings_button_font = _scaled_pt(9.4, scale, 7.4)
    card_title_font = _scaled_pt(9.2, scale, 7.2)
    button_pad_v = _scaled(3, scale, 2)
    button_pad_h = _scaled(7, scale, 4)
    action_min_h = _scaled(24, scale, 20)
    convert_min_h = _scaled(36, scale, 28)
    stop_min_w = _scaled(116, scale, 90)
    checkbox_spacing = _scaled(8, scale, 5)
    icon_box = _scaled(22, scale, 18)
    progress_h = _scaled(26, scale, 18)
    return f"""
QMainWindow {{
    background: {theme.app_bg};
}}
QWidget#a2mRoot, QWidget#mainColumn {{
    background: {theme.app_bg};
}}
QWidget#cardContent {{
    background: transparent;
    border: none;
}}
QFrame#card {{
    background: {theme.panel_bg};
    border: 1px solid {theme.border};
    border-radius: {frame_radius}px;
}}
QFrame#settingsPanel, QFrame#settingsCard, QFrame#modeHolder {{
    background: {theme.panel_bg};
    border: 2px solid {theme.border};
    border-radius: {frame_radius}px;
}}
QFrame#footerDivider {{
    background: {theme.border};
    border: none;
}}
QScrollArea#settingsScroll {{
    background: transparent;
    border: none;
}}
QScrollArea#settingsScroll QWidget#qt_scrollarea_viewport {{
    background: transparent;
    border: none;
}}
QWidget#settingsBody {{
    background: transparent;
}}
QLabel {{
    color: {theme.text_primary};
    background: transparent;
    font-family: "Segoe UI";
    font-size: {widget_font:.1f}pt;
}}
QLabel#title {{
    color: {theme.text_primary};
    font: 700 {title_font:.1f}pt "Segoe UI";
}}
QLabel#subtitle, QLabel#caption, QLabel#muted {{
    color: {theme.text_secondary};
    font: 600 {subtitle_font:.1f}pt "Segoe UI";
}}
QLabel#settingsCardTitle {{
    color: {theme.text_primary};
    font: 700 {card_title_font:.1f}pt "Segoe UI";
}}
QLabel#settingsSubtext {{
    color: {theme.text_secondary};
    font: 650 {button_font:.1f}pt "Segoe UI";
    padding-top: {_scaled(1, scale, 1)}px;
    padding-bottom: {_scaled(1, scale, 1)}px;
}}
QLabel:disabled, QLabel#subtitle:disabled, QLabel#caption:disabled, QLabel#muted:disabled {{
    color: {theme.disabled_fg};
}}
QLineEdit#settingsReadOnlyInput {{
    background: {theme.app_bg};
    color: {theme.text_primary};
    border: 1px solid {theme.border};
    border-radius: {button_radius}px;
    padding: {button_pad_v}px {button_pad_h}px;
    font: 600 {button_font:.1f}pt "Segoe UI";
}}
QLineEdit#settingsReadOnlyInput:disabled {{
    background: {theme.disabled_bg};
    color: {theme.disabled_fg};
    border-color: {theme.border};
}}
QPushButton {{
    background: {theme.panel_bg};
    color: {theme.text_primary};
    border: 1px solid {theme.border};
    border-radius: {button_radius}px;
    padding: {button_pad_v}px {button_pad_h}px;
    font: 600 {button_font:.1f}pt "Segoe UI";
}}
QPushButton:hover {{
    background: {theme.accent};
    color: {theme.text_primary};
}}
QPushButton:disabled {{
    background: {theme.disabled_bg};
    color: {theme.disabled_fg};
    border-color: {theme.border};
}}
QPushButton#actionButton {{
    min-height: {action_min_h}px;
}}
QFrame#settingsPanel QPushButton#settingsActionButton {{
    min-height: {_scaled(32, scale, 24)}px;
    font: 700 {settings_button_font:.1f}pt "Segoe UI";
}}
QPushButton#convertButton {{
    background: {theme.panel_bg};
    color: {theme.text_primary};
    border: 1px solid {theme.border};
    min-height: {convert_min_h}px;
    font: 700 {button_bold_font:.1f}pt "Segoe UI";
}}
QPushButton#convertButton:hover {{
    background: {theme.accent};
}}
QPushButton#convertButton:disabled {{
    background: {theme.disabled_bg};
    color: {theme.disabled_fg};
    border-color: {theme.border};
}}
QPushButton#stopButton {{
    background: {theme.danger};
    border: 1px solid {theme.danger};
    min-height: {convert_min_h}px;
    min-width: {stop_min_w}px;
    font: 700 {button_bold_font:.1f}pt "Segoe UI";
}}
QPushButton#stopButton:hover {{
    background: {theme.danger_hover};
    border-color: {theme.danger_hover};
}}
QPushButton#footerLink {{
    background: transparent;
    color: {theme.accent};
    border: none;
    padding: 2px;
    font: 700 {footer_font:.1f}pt "Segoe UI";
    text-align: left;
}}
QPushButton#footerLink:hover {{
    color: {theme.text_primary};
    background: transparent;
}}
QPushButton#footerIcon {{
    background: transparent;
    color: {theme.text_primary};
    border: none;
    min-width: {icon_box}px;
    min-height: {icon_box}px;
    max-width: {icon_box}px;
    max-height: {icon_box}px;
    padding: 0;
    margin: 0;
}}
QPushButton#footerIcon:hover {{
    background: transparent;
}}
QPushButton#modeButton {{
    background: transparent;
    color: {theme.text_secondary};
    border: none;
    border-radius: {button_radius}px;
    padding: {_scaled(4, scale, 2)}px {_scaled(10, scale, 6)}px;
    font: 600 {button_font:.1f}pt "Segoe UI";
}}
QPushButton#modeButton:hover {{
    color: {theme.text_primary};
    background: {theme.panel_bg};
}}
QPushButton#modeButton:checked {{
    color: {theme.text_primary};
    background: {theme.accent};
}}
QTextEdit#consoleOutput, QPlainTextEdit#consoleOutput {{
    background: {theme.app_bg};
    color: {theme.text_primary};
    border: 1px solid {theme.border};
    border-radius: {button_radius}px;
    padding: {_scaled(2, scale, 2)}px {_scaled(7, scale, 4)}px;
    font: 600 {console_font:.1f}pt "Segoe UI";
    selection-background-color: {theme.accent};
}}
QProgressBar#downloadProgress {{
    background: {theme.app_bg};
    border: 1px solid {theme.border};
    border-radius: {_scaled(4, scale, 2)}px;
    min-height: {progress_h}px;
    max-height: {progress_h}px;
    text-align: center;
    color: {theme.text_primary};
    font: 700 {_scaled_pt(8.8, scale, 6.4):.1f}pt "Segoe UI";
}}
QProgressBar#downloadProgress::chunk {{
    background: {theme.accent};
    border-radius: {_scaled(4, scale, 2)}px;
}}
QCheckBox {{
    color: {theme.text_primary};
    font: 600 {button_font:.1f}pt "Segoe UI";
    spacing: {checkbox_spacing}px;
}}
"""




