from __future__ import annotations
import os


def apply_windows_titlebar_theme(widget, *, dark: bool) -> None:
    if os.name != 'nt':
        return
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = int(widget.winId())
        if hwnd == 0:
            return
        value = ctypes.c_int(1 if dark else 0)
        size = ctypes.sizeof(value)
        dwm = ctypes.windll.dwmapi
        for attribute in (20, 19):
            result = dwm.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                ctypes.c_uint(attribute),
                ctypes.byref(value),
                ctypes.c_uint(size),
            )
            if result == 0:
                break
    except Exception:
        return
