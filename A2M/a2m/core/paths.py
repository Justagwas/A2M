from __future__ import annotations
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from .config import CONFIG_FILENAME

def app_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]

def app_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return app_root_dir()

def bundle_dir() -> Path:
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', None)
        if base:
            return Path(base)
    return app_dir()

def icon_path() -> Path:
    bundled = bundle_dir() / 'icon.ico'
    if bundled.exists():
        return bundled
    local = app_dir() / 'icon.ico'
    if local.exists():
        return local
    return bundled

def config_path() -> Path:
    return localappdata_dir() / 'A2M' / CONFIG_FILENAME

def legacy_config_paths() -> list[Path]:
    appdata = os.environ.get('APPDATA')
    roaming_root = Path(appdata) if appdata else Path.home() / 'AppData' / 'Roaming'
    return dedupe_paths(
        (
            app_dir() / CONFIG_FILENAME,
            roaming_root / 'A2M' / CONFIG_FILENAME,
            # A2M v1 stored its configuration directly in the user profile.
            # Keep this last so a newer v2 configuration wins when both exist.
            Path.home() / '.a2m_config.json',
        ),
        resolve_if_exists=False,
    )

def localappdata_dir() -> Path:
    root = os.environ.get('LOCALAPPDATA')
    if root:
        return Path(root)
    return Path.home() / 'AppData' / 'Local'


def normalized_path_key(path: Path | str, *, resolve_if_exists: bool=True) -> str:
    candidate = Path(path)
    try:
        if resolve_if_exists and candidate.exists():
            return str(candidate.resolve()).lower()
    except Exception:
        pass
    return str(candidate).strip().lower()


def dedupe_paths(paths: Iterable[Path | str], *, existing_only: bool=False, resolve_if_exists: bool=True) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        candidate = Path(raw)
        if not str(candidate):
            continue
        if existing_only and (not candidate.exists()):
            continue
        key = normalized_path_key(candidate, resolve_if_exists=resolve_if_exists)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped
