from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app_controller import AppController

__all__ = ['AppController']


def __getattr__(name: str):
    if name == 'AppController':
        from .app_controller import AppController
        return AppController
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
