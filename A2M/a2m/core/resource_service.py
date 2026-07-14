from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass

from .config import GPU_BATCH_SIZE_MAX, GPU_BATCH_SIZE_MIN


PERFORMANCE_MODES = ('low', 'balanced', 'high', 'maximum')
_PERFORMANCE_MODE = 'balanced'


@dataclass(slots=True, frozen=True)
class HardwareDefaults:
    processor_count: int
    total_memory_mib: int
    gpu_memory_mib: int
    cpu_mode: str
    gpu_cpu_mode: str
    gpu_memory_max_batch: int
    gpu_batch_size: int


def normalize_performance_mode(value: str | None) -> str:
    normalized = str(value or 'balanced').strip().lower()
    return normalized if normalized in PERFORMANCE_MODES else 'balanced'


def logical_processor_count() -> int:
    process_cpu_count = getattr(os, 'process_cpu_count', None)
    try:
        detected = process_cpu_count() if callable(process_cpu_count) else os.cpu_count()
    except Exception:
        detected = os.cpu_count()
    return max(1, int(detected or 1))


def recommended_performance_mode(processor_count: int | None = None) -> str:
    count = logical_processor_count() if processor_count is None else max(1, int(processor_count))
    if count <= 2:
        return 'low'
    if count <= 8:
        return 'balanced'
    return 'high'


def total_physical_memory_mib() -> int:
    if os.name == 'nt':
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(0, int(status.ullTotalPhys // (1024 * 1024)))
        except Exception:
            pass
    try:
        page_size = int(os.sysconf('SC_PAGE_SIZE'))
        page_count = int(os.sysconf('SC_PHYS_PAGES'))
        return max(0, int((page_size * page_count) // (1024 * 1024)))
    except Exception:
        return 0


def worker_count_for_mode(mode: str | None, processor_count: int | None = None) -> int:
    count = logical_processor_count() if processor_count is None else max(1, int(processor_count))
    normalized = normalize_performance_mode(mode)
    proportions = {
        'low': 0.25,
        'balanced': 0.5,
        'high': 0.75,
        'maximum': 1.0,
    }
    return max(1, min(count, int(math.ceil(count * proportions[normalized]))))


def set_performance_mode(mode: str | None) -> str:
    global _PERFORMANCE_MODE
    _PERFORMANCE_MODE = normalize_performance_mode(mode)
    return _PERFORMANCE_MODE


def active_worker_count() -> int:
    return worker_count_for_mode(_PERFORMANCE_MODE)


def normalize_gpu_batch_size(value: int | str | None) -> int:
    try:
        batch_size = int(value)
    except Exception:
        batch_size = 2
    return max(GPU_BATCH_SIZE_MIN, min(GPU_BATCH_SIZE_MAX, batch_size))


def normalize_gpu_memory_max_batch(value: int | str | None) -> int:
    try:
        requested = int(value)
    except Exception:
        requested = 4
    tiers = (4, 6, 8, 12, 16)
    requested = max(tiers[0], min(tiers[-1], requested))
    return min(tiers, key=lambda tier: (abs(tier - requested), tier))


def gpu_memory_presets(max_batch: int | str | None = 4) -> dict[str, int]:
    normalized = normalize_gpu_memory_max_batch(max_batch)
    return {
        4: {'low': 1, 'balanced': 2, 'high': 3, 'maximum': 4},
        6: {'low': 1, 'balanced': 2, 'high': 4, 'maximum': 6},
        8: {'low': 1, 'balanced': 3, 'high': 5, 'maximum': 8},
        12: {'low': 1, 'balanced': 4, 'high': 8, 'maximum': 12},
        16: {'low': 1, 'balanced': 6, 'high': 11, 'maximum': 16},
    }[normalized].copy()


def gpu_memory_level_for_batch(batch_size: int | str | None, max_batch: int | str | None = 4) -> str:
    batch = normalize_gpu_batch_size(batch_size)
    presets = gpu_memory_presets(max_batch)
    return min(presets, key=lambda level: (abs(presets[level] - batch), presets[level]))


def gpu_memory_level_name(batch_size: int | str | None, max_batch: int | str | None = 4) -> str:
    return {
        'low': 'Low',
        'balanced': 'Balanced',
        'high': 'High',
        'maximum': 'Max',
    }[gpu_memory_level_for_batch(batch_size, max_batch)]


def recommended_gpu_memory_max_batch(
    *,
    total_memory_mib: int | None = None,
    gpu_memory_mib: int | None = None,
) -> int:
    system_mib = total_physical_memory_mib() if total_memory_mib is None else max(0, int(total_memory_mib))
    if system_mib < 8 * 1024:
        system_limit = 4
    elif system_mib < 16 * 1024:
        system_limit = 6
    elif system_mib < 24 * 1024:
        system_limit = 8
    elif system_mib < 48 * 1024:
        system_limit = 12
    else:
        system_limit = 16

    graphics_mib = max(0, int(gpu_memory_mib or 0))
    if graphics_mib <= 0:
        return min(system_limit, 8)
    if graphics_mib < 5 * 1024:
        graphics_limit = 4
    elif graphics_mib < 7 * 1024:
        graphics_limit = 6
    elif graphics_mib < 10 * 1024:
        graphics_limit = 8
    elif graphics_mib < 14 * 1024:
        graphics_limit = 12
    else:
        graphics_limit = 16
    return min(system_limit, graphics_limit)


def recommended_hardware_defaults(
    *,
    processor_count: int | None = None,
    total_memory_mib: int | None = None,
    gpu_memory_mib: int | None = None,
) -> HardwareDefaults:
    processors = logical_processor_count() if processor_count is None else max(1, int(processor_count))
    system_mib = total_physical_memory_mib() if total_memory_mib is None else max(0, int(total_memory_mib))
    graphics_mib = max(0, int(gpu_memory_mib or 0))
    max_batch = recommended_gpu_memory_max_batch(
        total_memory_mib=system_mib,
        gpu_memory_mib=graphics_mib,
    )
    presets = gpu_memory_presets(max_batch)
    return HardwareDefaults(
        processor_count=processors,
        total_memory_mib=system_mib,
        gpu_memory_mib=graphics_mib,
        cpu_mode=recommended_performance_mode(processors),
        gpu_cpu_mode='low' if processors <= 2 else 'balanced',
        gpu_memory_max_batch=max_batch,
        gpu_batch_size=presets['balanced'],
    )


def gpu_output_memory_mib(batch_size: int | str | None, frame_count: int = 691) -> int:
    batch = normalize_gpu_batch_size(batch_size)
    frames = max(1, int(frame_count))
    output_bytes = batch * (
        frames * frames * 90 * 4
        + 90 * frames * 256 * 4
        + frames * 90 * 4
    )
    return int(round(output_bytes / (1024.0 * 1024.0)))


def gpu_memory_description(batch_size: int | str | None) -> str:
    batch = normalize_gpu_batch_size(batch_size)
    section_word = 'section' if batch == 1 else 'sections'
    return f'Processes up to {batch} audio {section_word} at a time.'


def gpu_memory_resource_description(batch_size: int | str | None) -> str:
    batch = normalize_gpu_batch_size(batch_size)
    output_mib = gpu_output_memory_mib(batch)
    return f'About {output_mib} MiB for model output, plus working memory. A2M reduces the group if needed.'


def performance_display_name(mode: str | None) -> str:
    normalized = normalize_performance_mode(mode)
    return {
        'low': 'Low',
        'balanced': 'Balanced',
        'high': 'Fast',
        'maximum': 'Max',
    }[normalized]


def normalize_performance_device(device: str | None) -> str:
    return 'gpu' if str(device or '').strip().lower() == 'gpu' else 'cpu'


def performance_description(mode: str | None, device: str | None = 'cpu') -> str:
    normalized = normalize_performance_mode(mode)
    return {
        'low': 'Leaves most CPU power available for other apps.',
        'balanced': 'Balances transcription speed with other apps.',
        'high': 'Uses most of the CPU for faster transcription while leaving some capacity for other apps.',
        'maximum': 'Uses all available CPU threads for maximum speed.',
    }[normalized]


def performance_resource_description(
    mode: str | None,
    processor_count: int | None = None,
    *,
    device: str | None = 'cpu',
) -> str:
    count = logical_processor_count() if processor_count is None else max(1, int(processor_count))
    workers = worker_count_for_mode(mode, count)
    percentage = int(round(workers / count * 100.0))
    return f'Up to {workers} of {count} logical CPU threads ({percentage}%).'
