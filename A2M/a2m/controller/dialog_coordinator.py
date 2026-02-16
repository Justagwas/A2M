from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QMessageBox, QWidget


class DialogCoordinator:

    def __init__(self, window: QWidget, *, apply_theme: Callable[[QWidget], None], exec_dialog: Callable[[QWidget], int]) -> None:
        self._window = window
        self._apply_theme = apply_theme
        self._exec_dialog = exec_dialog

    def build_message_box(self, *, icon: QMessageBox.Icon, title: str, text: str, buttons: QMessageBox.StandardButtons=QMessageBox.Ok, default_button: QMessageBox.StandardButton=QMessageBox.NoButton) -> QMessageBox:
        box = QMessageBox(self._window)
        box.setOption(QMessageBox.DontUseNativeDialog, True)
        box.setIcon(icon)
        box.setWindowTitle(str(title or 'A2M'))
        box.setText(str(text or ''))
        box.setStandardButtons(buttons)
        if default_button != QMessageBox.NoButton:
            box.setDefaultButton(default_button)
        icon_obj = self._window.windowIcon()
        if not icon_obj.isNull():
            box.setWindowIcon(icon_obj)
        self._apply_theme(box)
        return box

    def show_info(self, title: str, text: str) -> QMessageBox.StandardButton:
        return self._exec_dialog(self.build_message_box(icon=QMessageBox.Information, title=title, text=text))

    def show_warning(self, title: str, text: str) -> QMessageBox.StandardButton:
        return self._exec_dialog(self.build_message_box(icon=QMessageBox.Warning, title=title, text=text))

    def show_error(self, title: str, text: str) -> QMessageBox.StandardButton:
        return self._exec_dialog(self.build_message_box(icon=QMessageBox.Critical, title=title, text=text))

    def ask_question(self, title: str, text: str, *, default_button: QMessageBox.StandardButton=QMessageBox.NoButton) -> QMessageBox.StandardButton:
        return self._exec_dialog(self.build_message_box(icon=QMessageBox.Question, title=title, text=text, buttons=QMessageBox.Yes | QMessageBox.No, default_button=default_button))

