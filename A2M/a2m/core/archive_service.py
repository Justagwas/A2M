from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from threading import Event

from .config import ARCHIVE_EXTRACT_MAX_BYTES


def assert_not_stopped(stop_event: Event | None, *, message: str) -> None:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError(message)


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)


def safe_extract_zip(
    archive_path: Path,
    target_dir: Path,
    *,
    stop_event: Event | None = None,
    stop_message: str = 'Operation stopped by user.',
    unsafe_entry_prefix: str = 'Unsafe archive entry rejected',
    empty_message: str = 'Archive is empty.',
    max_total_uncompressed_bytes: int | None = ARCHIVE_EXTRACT_MAX_BYTES,
) -> None:
    extracted_any = False
    extracted_bytes = 0
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_target = target_dir.resolve()
    with zipfile.ZipFile(archive_path, 'r') as archive:
        for info in archive.infolist():
            assert_not_stopped(stop_event, message=stop_message)
            member_name = str(info.filename or '').replace('\\', '/').lstrip('/')
            if not member_name:
                continue
            destination = (target_dir / member_name).resolve()
            try:
                destination.relative_to(resolved_target)
            except Exception as exc:
                raise RuntimeError(f'{unsafe_entry_prefix}: {member_name}') from exc
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            declared_size = max(0, int(getattr(info, 'file_size', 0) or 0))
            if max_total_uncompressed_bytes is not None and extracted_bytes + declared_size > int(max_total_uncompressed_bytes):
                raise RuntimeError('Archive is larger than the allowed extraction limit.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, 'r') as src, open(destination, 'wb') as dst:
                while True:
                    assert_not_stopped(stop_event, message=stop_message)
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    extracted_bytes += len(chunk)
                    if max_total_uncompressed_bytes is not None and extracted_bytes > int(max_total_uncompressed_bytes):
                        raise RuntimeError('Archive is larger than the allowed extraction limit.')
                    dst.write(chunk)
            extracted_any = True
    if not extracted_any:
        raise RuntimeError(empty_message)
