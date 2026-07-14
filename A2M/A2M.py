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
import subprocess
import sys
import threading
import time
from pathlib import Path
from a2m.core.config import APP_NAME, APP_RESTART_EXIT_CODE, APP_SHORT_NAME, APP_VERSION
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

    class LoadingSplash(QSplashScreen):
        def mousePressEvent(self, event) -> None:
            event.ignore()

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
    painter.setRenderHint(QPainter.Antialiasing, False)
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
    splash = LoadingSplash(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    message_font = QFont('Segoe UI')
    message_font.setWeight(QFont.DemiBold)
    message_font.setPointSizeF(max(7.0, 9.0 * dpi_scale))
    splash.setFont(message_font)
    return splash


def _initialize_startup(app, splash, *, splash_shown_at: float):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from a2m.core import gpu_runtime_service, resource_service, runtime_inventory_service, runtime_pack_service
    from a2m.core.config_service import load_config, save_config

    config = load_config()
    initialize_hardware_defaults = not config.hardware_defaults_initialized

    result: dict[str, object] = {}
    pending_statuses = [
        'Checking processor and system memory...'
        if initialize_hardware_defaults
        else 'Checking available transcription backends...'
    ]
    status_lock = threading.Lock()
    provider = str(config.gpu_provider_preference or 'dml')

    def add_status(message: str) -> None:
        with status_lock:
            pending_statuses.append(str(message))

    def initialize_components() -> None:
        try:
            if initialize_hardware_defaults:
                processor_count = resource_service.logical_processor_count()
                total_memory_mib = resource_service.total_physical_memory_mib()
                total_memory_gib = max(1, int(round(total_memory_mib / 1024.0))) if total_memory_mib else 0
                memory_text = f' and {total_memory_gib} GB of system memory' if total_memory_gib else ''
                add_status(f'Detected {processor_count} logical processor threads{memory_text}.')
                add_status('Checking graphics memory...')
                gpu_memory_mib = gpu_runtime_service.get_gpu_memory_mib(provider)
                if gpu_memory_mib > 0:
                    add_status(f'Detected about {max(1, int(round(gpu_memory_mib / 1024.0)))} GB of graphics memory.')
                else:
                    add_status('Graphics memory was not reported; using conservative limits.')
                add_status('Choosing safe CPU and GPU usage defaults...')
                defaults = resource_service.recommended_hardware_defaults(
                    processor_count=processor_count,
                    total_memory_mib=total_memory_mib,
                    gpu_memory_mib=gpu_memory_mib,
                )
                result['defaults'] = defaults
                cpu_name = resource_service.performance_display_name(defaults.cpu_mode)
                gpu_name = resource_service.gpu_memory_level_name(
                    defaults.gpu_batch_size,
                    defaults.gpu_memory_max_batch,
                )
                add_status(f'Selected {cpu_name} CPU usage and {gpu_name} GPU memory usage.')
            inventory = runtime_inventory_service.inspect_runtime_inventory(
                configured_provider=config.gpu_provider_preference,
                configured_runtime_path=(config.gpu_runtime_path if config.gpu_runtime_enabled else ''),
                status_callback=add_status,
            )
            result['runtime_inventory'] = inventory
            ready_labels = [
                runtime_pack_service.provider_display_name(name)
                for name in ('dml', 'cuda')
                if inventory.provider_status(name).available
            ]
            if ready_labels:
                add_status(f'GPU support ready: {", ".join(ready_labels)}.')
            elif inventory.cpu_available:
                add_status('CPU transcription support is ready; optional GPU components were not found.')
            else:
                add_status('Transcription runtime support could not be verified.')
        except Exception as exc:
            result['error'] = str(exc)
            add_status('Some system checks were unavailable; A2M will use safe settings.')

    worker = threading.Thread(target=initialize_components, daemon=True)
    worker.start()
    next_status_at = splash_shown_at
    message_color = QColor('#B7B7BC')
    while True:
        now = time.monotonic()
        status_message = ''
        if now >= next_status_at:
            with status_lock:
                if pending_statuses:
                    status_message = pending_statuses.pop(0)
        if status_message:
            splash.showMessage(
                status_message,
                Qt.AlignBottom | Qt.AlignHCenter,
                message_color,
            )
            next_status_at = now + 0.38
        app.processEvents()
        with status_lock:
            statuses_waiting = bool(pending_statuses)
        if (
            not worker.is_alive()
            and not statuses_waiting
            and (now - splash_shown_at) >= 2.0
        ):
            break
        time.sleep(0.02)
    worker.join()

    if initialize_hardware_defaults:
        defaults = result.get('defaults')
        if not isinstance(defaults, resource_service.HardwareDefaults):
            defaults = resource_service.recommended_hardware_defaults(gpu_memory_mib=0)
        config.cpu_performance_mode = defaults.cpu_mode
        config.gpu_performance_mode = defaults.gpu_cpu_mode
        config.gpu_memory_max_batch = defaults.gpu_memory_max_batch
        config.gpu_batch_size = defaults.gpu_batch_size
        config.hardware_defaults_initialized = True
        splash.showMessage(
            'Saving first-time settings...',
            Qt.AlignBottom | Qt.AlignHCenter,
            message_color,
        )
        app.processEvents()
        save_config(config)
    splash.showMessage(
        'Setup complete. Loading A2M...',
        Qt.AlignBottom | Qt.AlignHCenter,
        message_color,
    )
    app.processEvents()
    return result.get('runtime_inventory')


def _run_gui() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication(sys.argv)
    app.setApplicationName(APP_SHORT_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_SHORT_NAME)
    splash = _build_loading_splash()
    splash.show()
    splash.showMessage(
        'A2M is loading...',
        Qt.AlignBottom | Qt.AlignHCenter,
        QColor('#B7B7BC'),
    )
    app.processEvents()
    splash_shown_at = time.monotonic()
    guard = SingleInstanceGuard(MUTEX_NAME)
    if not guard.acquire():
        splash.close()
        QMessageBox.information(
            None,
            APP_NAME,
            f'{APP_NAME} is already running.',
        )
        return 0
    try:
        runtime_inventory = _initialize_startup(app, splash, splash_shown_at=splash_shown_at)
        from a2m.app_controller import AppController
        splash.showMessage(
            'Loading main window...',
            Qt.AlignBottom | Qt.AlignHCenter,
            QColor('#B7B7BC'),
        )
        app.processEvents()
        try:
            controller = AppController(app, startup_runtime_inventory=runtime_inventory)
        except RuntimeError as exc:
            splash.close()
            QMessageBox.critical(None, APP_NAME, str(exc))
            return 1
        controller.run()
        splash.finish(controller.window)
        return app.exec()
    finally:
        splash.close()
        guard.release()


def main() -> int:
    if '--gpu-helper' in sys.argv:
        return int(run_gpu_helper_cli(sys.argv[1:]))
    exit_code = _run_gui()
    if exit_code != APP_RESTART_EXIT_CODE:
        return int(exit_code)
    if getattr(sys, 'frozen', False):
        launch_args = [str(sys.executable), *sys.argv[1:]]
        working_dir = str(Path(sys.executable).resolve().parent)
    else:
        script_path = str(Path(sys.argv[0]).resolve())
        launch_args = [str(sys.executable), script_path, *sys.argv[1:]]
        working_dir = str(Path(script_path).resolve().parent)
    try:
        subprocess.Popen(launch_args, cwd=working_dir)
    except Exception as exc:
        print(f'[A2M] Restart failed: {exc}', file=sys.stderr)
        if os.name == 'nt':
            try:
                ctypes.windll.user32.MessageBoxW(
                    None,
                    f'A2M could not restart automatically.\n\n{exc}',
                    APP_NAME,
                    0x10,
                )
            except Exception:
                pass
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
