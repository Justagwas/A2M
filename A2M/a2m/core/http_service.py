from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Event
from typing import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ProgressCallback = Callable[[float], None]


def _ensure_not_stopped(stop_event: Event | None, *, message: str) -> None:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError(message)


def fetch_text(url: str, *, headers_profiles: tuple[dict[str, str], ...] | list[dict[str, str]], timeout_seconds: int | float, stop_event: Event | None=None, stop_message: str='Request stopped.') -> tuple[str, str]:
    target = str(url or '').strip()
    if not target:
        raise RuntimeError('Missing update source URL.')
    errors: list[str] = []
    for headers in headers_profiles:
        _ensure_not_stopped(stop_event, message=stop_message)
        try:
            request = Request(target, headers=headers)
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read().decode('utf-8-sig', errors='replace')
                final_url = target
                final_url_getter = getattr(response, 'geturl', None)
                if callable(final_url_getter):
                    try:
                        final_url = str(final_url_getter() or target)
                    except Exception:
                        final_url = target
            return (payload, final_url)
        except InterruptedError:
            raise
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(errors[-1])
    raise RuntimeError('Unable to fetch update source.')


def download_file_with_retries(url: str, dest_path: Path | str, *, headers_profiles: tuple[dict[str, str], ...] | list[dict[str, str]], timeout_seconds: int | float, retries_per_header: int, retry_backoff_seconds: int | float, progress_callback: ProgressCallback | None=None, stop_event: Event | None=None, stop_message: str='Download stopped by user.', error_prefix: str='Failed to download file', chunk_bytes: int=1024 * 1024) -> None:
    destination = Path(dest_path)
    tmp_path = destination.with_suffix(destination.suffix + '.part')
    if tmp_path.exists():
        tmp_path.unlink()
    attempt_errors: list[str] = []
    for profile_index, headers in enumerate(headers_profiles, start=1):
        for attempt in range(1, max(1, int(retries_per_header)) + 1):
            _ensure_not_stopped(stop_event, message=stop_message)
            if tmp_path.exists():
                tmp_path.unlink()
            try:
                request = Request(str(url or '').strip(), headers=headers)
                with urlopen(request, timeout=timeout_seconds) as response, open(tmp_path, 'wb') as out_file:
                    total_header = response.headers.get('Content-Length')
                    total_size = int(total_header) if total_header and total_header.isdigit() else None
                    downloaded = 0
                    while True:
                        _ensure_not_stopped(stop_event, message=stop_message)
                        chunk = response.read(max(1, int(chunk_bytes)))
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size and progress_callback:
                            percent = (float(downloaded) * 100.0) / float(total_size)
                            progress_callback(min(percent, 100.0))
                os.replace(tmp_path, destination)
                return
            except InterruptedError:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise
            except Exception as exc:
                if isinstance(exc, HTTPError):
                    err_text = f'HTTP {exc.code}: {exc.reason}'
                else:
                    err_text = str(exc)
                attempt_errors.append(f'profile {profile_index} attempt {attempt}: {err_text}')
                if attempt < max(1, int(retries_per_header)):
                    time.sleep(float(retry_backoff_seconds) * attempt)
    if tmp_path.exists():
        tmp_path.unlink()
    error_summary = '; '.join(attempt_errors) if attempt_errors else 'Unknown error'
    raise RuntimeError(f'{error_prefix}: {error_summary}')
