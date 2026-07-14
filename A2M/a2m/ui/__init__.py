from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_window import MainWindow

__all__ = ['MainWindow']


def __getattr__(name: str):
    if name == 'MainWindow':
        from .main_window import MainWindow
        return MainWindow
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
