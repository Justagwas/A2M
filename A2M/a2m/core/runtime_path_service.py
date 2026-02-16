from __future__ import annotations

import os
import sys
from pathlib import Path

from .runtime_artifacts import runtime_search_paths


def _normalized_path_key(path_value: str | Path) -> str:
    try:
        return str(Path(path_value).resolve()).lower()
    except Exception:
        return str(path_value).lower()


def _prepend_env_path_once(path_str: str, *, tracked_prefixes: set[str] | None=None, track_existing: bool=False) -> None:
    normalized = _normalized_path_key(path_str)
    if not normalized:
        return
    if tracked_prefixes is not None and normalized in tracked_prefixes:
        return
    existing = [entry for entry in os.environ.get('PATH', '').split(os.pathsep) if entry]
    existing_normalized = {_normalized_path_key(entry) for entry in existing}
    if normalized in existing_normalized:
        if tracked_prefixes is not None and track_existing:
            tracked_prefixes.add(normalized)
        return
    os.environ['PATH'] = path_str + os.pathsep + os.environ.get('PATH', '')
    if tracked_prefixes is not None:
        tracked_prefixes.add(normalized)


def prime_runtime_path(
    runtime_root: Path,
    *,
    tracked_sys_paths: set[str] | None=None,
    tracked_env_prefixes: set[str] | None=None,
    track_existing_env_prefixes: bool=False,
    dll_dir_handles: dict[str, object] | list[object] | None=None,
) -> None:
    if not runtime_root.exists():
        return
    root_str = str(runtime_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        if tracked_sys_paths is not None:
            tracked_sys_paths.add(root_str)
    for candidate in runtime_search_paths(runtime_root):
        _prepend_env_path_once(
            str(candidate),
            tracked_prefixes=tracked_env_prefixes,
            track_existing=track_existing_env_prefixes,
        )
    if not hasattr(os, 'add_dll_directory'):
        return
    for candidate in runtime_search_paths(runtime_root):
        candidate_str = str(candidate)
        if isinstance(dll_dir_handles, dict) and candidate_str in dll_dir_handles:
            continue
        try:
            handle = os.add_dll_directory(candidate_str)
        except Exception:
            continue
        if isinstance(dll_dir_handles, dict):
            dll_dir_handles[candidate_str] = handle
        elif isinstance(dll_dir_handles, list):
            dll_dir_handles.append(handle)
