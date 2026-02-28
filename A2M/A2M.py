"""
A2M - Audio to MIDI Converter

Copyright 2026 Justagwas

This program is licensed under the GNU General Public License v3.0
See the LICENSE file in the project root for the full license text.

Official Project Page:
https://justagwas.com/projects/a2m

Official Source Code:
https://github.com/justagwas/a2m
https://sourceforge.net/projects/a2m

SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations
import ctypes
import os
import sys
from a2m.core.config import APP_NAME, APP_SHORT_NAME
from a2m.core.gpu_helper import run_gpu_helper_cli

MUTEX_NAME = 'A2MMutex'


class SingleInstanceGuard:
    def __init__(self, mutex_name: str) -> None:
        self._mutex_name = str(mutex_name or '').strip() or MUTEX_NAME
        self._handle = None

    def acquire(self) -> bool:
        if os.name != 'nt':
            return True
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
            kernel32.CreateMutexW.restype = ctypes.c_void_p
            handle = kernel32.CreateMutexW(None, 0, self._mutex_name)
            if not handle:
                return False
            error_already_exists = 183
            if kernel32.GetLastError() == error_already_exists:
                kernel32.CloseHandle(handle)
                return False
            self._handle = handle
            return True
        except Exception as exc:
            print(f'[A2M] Warning: single-instance guard failed: {exc}', file=sys.stderr)
            return True

    def release(self) -> None:
        if os.name != 'nt':
            return
        if self._handle is None:
            return
        try:
            ctypes.windll.kernel32.CloseHandle(self._handle)
        except Exception as exc:
            print(f'[A2M] Warning: failed to release single-instance guard: {exc}', file=sys.stderr)
        self._handle = None


def _build_loading_splash():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPixmap
    from PySide6.QtWidgets import QSplashScreen
    screen = QGuiApplication.primaryScreen()
    dpi_scale = 1.0
    dpr = 1.0
    if screen is not None:
        try:
            logical_dpi = float(screen.logicalDotsPerInch())
            if logical_dpi > 0:
                dpi_scale = max(1.0, min(3.0, logical_dpi / 96.0))
        except Exception:
            dpi_scale = 1.0
        try:
            dpr = max(1.0, float(screen.devicePixelRatio()))
        except Exception:
            dpr = 1.0

    width = int(round(460 * dpi_scale))
    height = int(round(180 * dpi_scale))
    px_width = max(1, int(round(width * dpr)))
    px_height = max(1, int(round(height * dpr)))

    pixmap = QPixmap(px_width, px_height)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(QColor('#0A0A0B'))
    painter = QPainter(pixmap)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor('#141416'))
    border_px = max(1, int(round(1 * dpi_scale)))
    painter.drawRect(0, 0, width, height)
    painter.setPen(QColor('#D20F39'))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(border_px, border_px, width - (2 * border_px), height - (2 * border_px))
    title_font = QFont('Segoe UI')
    title_font.setBold(True)
    title_font.setPointSizeF(max(11.0, 16.0 * dpi_scale))
    subtitle_font = QFont('Segoe UI')
    subtitle_font.setWeight(QFont.DemiBold)
    subtitle_font.setPointSizeF(max(8.0, 10.0 * dpi_scale))
    painter.setFont(title_font)
    painter.setPen(QColor('#F4F4F5'))
    x_margin = max(14, int(round(24 * dpi_scale)))
    title_y = max(40, int(round(78 * dpi_scale)))
    subtitle_y = max(58, int(round(108 * dpi_scale)))
    painter.drawText(x_margin, title_y, 'A2M is loading')
    painter.setFont(subtitle_font)
    painter.setPen(QColor('#B7B7BC'))
    painter.drawText(x_margin, subtitle_y, 'Initializing components...')
    painter.end()
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    message_font = QFont('Segoe UI')
    message_font.setWeight(QFont.DemiBold)
    message_font.setPointSizeF(max(7.0, 9.0 * dpi_scale))
    splash.setFont(message_font)
    return splash


def _run_gui() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication(sys.argv)
    app.setApplicationName(APP_SHORT_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_SHORT_NAME)
    guard = SingleInstanceGuard(MUTEX_NAME)
    if not guard.acquire():
        QMessageBox.information(
            None,
            APP_NAME,
            f'{APP_NAME} is already running.',
        )
        return 0
    splash = _build_loading_splash()
    splash.show()
    splash.showMessage(
        'A2M is loading...',
        Qt.AlignBottom | Qt.AlignHCenter,
        QColor('#B7B7BC'),
    )
    app.processEvents()
    try:
        from a2m.app_controller import AppController
        splash.showMessage(
            'Loading main window...',
            Qt.AlignBottom | Qt.AlignHCenter,
            QColor('#B7B7BC'),
        )
        app.processEvents()
        controller = AppController(app)
        controller.run()
        splash.finish(controller.window)
        return app.exec()
    finally:
        splash.close()
        guard.release()


def main() -> int:
    if '--gpu-helper' in sys.argv:
        return int(run_gpu_helper_cli(sys.argv[1:]))
    return _run_gui()


if __name__ == '__main__':
    raise SystemExit(main())

