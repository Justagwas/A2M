from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QAbstractItemView, QComboBox, QSlider, QWidget


def is_pointer_control(widget: object) -> bool:
    return isinstance(widget, (QAbstractButton, QSlider, QAbstractItemView, QComboBox))


def sync_pointer_cursor(widget: QWidget) -> None:
    if widget.isEnabled():
        widget.setCursor(Qt.PointingHandCursor)
    else:
        widget.unsetCursor()
