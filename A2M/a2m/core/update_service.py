from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread

from .config import (
    APP_NAME,
    APP_SHORT_NAME,
    APP_VERSION,
    DOWNLOAD_HEADER_PROFILES,
    DOWNLOAD_RETRIES_PER_HEADER,
    DOWNLOAD_RETRY_BACKOFF_SECONDS,
    INNO_SETUP_APP_ID,
    OFFICIAL_PAGE_URL,
    UPDATE_CHECK_TIMEOUT_SECONDS,
    UPDATE_GITHUB_DOWNLOAD_URL,
    UPDATE_MANIFEST_URL,
)
from .paths import app_dir, localappdata_dir
from .self_updater import (
    PreparedUpdateInstall,
    SelfUpdater,
    UpdateCheckData,
    is_newer_version,
    normalize_version,
    parse_semver,
)

_UPDATER: SelfUpdater | None = None
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def parse_version_tuple(version_text: str) -> tuple[int, ...] | None:
    return parse_semver(normalize_version(version_text))


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _sanitize_notes(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _runtime_storage_dir() -> Path:
    root = localappdata_dir() / APP_SHORT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _get_updater() -> SelfUpdater:
    global _UPDATER
    if _UPDATER is None:
        executable_name = f"{APP_SHORT_NAME}.exe" if sys.platform == "win32" else APP_SHORT_NAME
        _UPDATER = SelfUpdater(
            app_name=APP_NAME,
            app_version=APP_VERSION,
            manifest_url=UPDATE_MANIFEST_URL,
            page_url=OFFICIAL_PAGE_URL,
            setup_url=UPDATE_GITHUB_DOWNLOAD_URL,
            installer_app_id=INNO_SETUP_APP_ID,
            executable_name=executable_name,
            install_dir=app_dir(),
            runtime_storage_dir=_runtime_storage_dir(),
            timeout_seconds=float(UPDATE_CHECK_TIMEOUT_SECONDS),
            request_retries=max(1, int(DOWNLOAD_RETRIES_PER_HEADER)),
            manifest_request_retries=1,
            retry_backoff_seconds=max(0.0, float(DOWNLOAD_RETRY_BACKOFF_SECONDS)),
            header_profiles=DOWNLOAD_HEADER_PROFILES,
        )
        Thread(target=_UPDATER.recover_pending_update, daemon=True).start()
    return _UPDATER


def check_for_updates(current_version: str, *, stop_event: Event | None = None) -> UpdateCheckData:
    return _get_updater().check_for_updates(current_version, stop_event=stop_event)


def fetch_update_manifest(stop_event: Event | None = None) -> tuple[str, str]:
    check = check_for_updates(APP_VERSION, stop_event=stop_event)
    return str(check.latest_version or ""), str(check.page_url or OFFICIAL_PAGE_URL)


def _build_check_data_from_payload(payload: dict[str, object]) -> UpdateCheckData:
    current_version = normalize_version(str(payload.get("current_version") or APP_VERSION)) or "0.0.0"
    latest_version = normalize_version(str(payload.get("latest") or "")) or ""
    current_tuple = parse_semver(current_version)
    latest_tuple = parse_semver(latest_version)
    if current_tuple is None:
        raise RuntimeError(f"Current version is not valid semver: {current_version!r}")
    if latest_tuple is None:
        raise RuntimeError(f"Latest version is not valid semver: {latest_version!r}")
    if not bool(payload.get("update_available", False)):
        raise RuntimeError("Update payload is not marked as update-available.")
    if latest_tuple <= current_tuple:
        raise RuntimeError(
            f"Update payload does not contain a newer version (current={current_version}, latest={latest_version})."
        )

    setup_url = str(payload.get("setup_url") or "").strip()
    setup_sha256 = str(payload.get("setup_sha256") or "").strip().lower()
    setup_size = max(0, _safe_int(payload.get("setup_size"), 0))
    if not setup_url:
        raise RuntimeError("Update payload does not include a setup installer URL.")
    if not _SHA256_RE.fullmatch(setup_sha256):
        raise RuntimeError("Update payload does not include a valid setup SHA256.")
    if setup_size <= 0:
        raise RuntimeError("Update payload does not include a valid setup size.")

    return UpdateCheckData(
        update_available=True,
        current_version=current_version,
        latest_version=latest_version,
        page_url=str(payload.get("url") or OFFICIAL_PAGE_URL),
        setup_url=setup_url,
        setup_sha256=setup_sha256,
        setup_size=setup_size,
        released=str(payload.get("released") or ""),
        notes=_sanitize_notes(payload.get("notes")),
        source="latest.json",
        channel=str(payload.get("channel") or "stable"),
        minimum_supported_version=normalize_version(str(payload.get("minimum_supported_version") or "1.0.0")) or "1.0.0",
        requires_manual_update=bool(payload.get("requires_manual_update", False)),
        setup_managed_install=bool(payload.get("setup_managed_install", False)),
    )


def prepare_update_from_payload(
    payload: dict[str, object],
    *,
    stop_event: Event | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> PreparedUpdateInstall:
    return _get_updater().prepare_update(
        _build_check_data_from_payload(payload),
        stop_event=stop_event,
        progress_callback=progress_callback,
    )


def launch_prepared_update(
    prepared: PreparedUpdateInstall,
    *,
    restart_after_update: bool,
) -> None:
    _get_updater().launch_prepared_update(
        prepared,
        restart_after_update=bool(restart_after_update),
    )


def discard_prepared_update(prepared: PreparedUpdateInstall) -> None:
    _get_updater().discard_prepared_update(prepared)


def install_update_from_payload(
    payload: dict[str, object],
    *,
    stop_event: Event | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> str:
    check_data = _build_check_data_from_payload(payload)
    return str(
        _get_updater().install_update(
            check_data,
            stop_event=stop_event,
            progress_callback=progress_callback,
        )
        or check_data.latest_version
        or ""
    )


__all__ = [
    "check_for_updates",
    "discard_prepared_update",
    "fetch_update_manifest",
    "install_update_from_payload",
    "is_newer_version",
    "launch_prepared_update",
    "normalize_version",
    "prepare_update_from_payload",
    "parse_semver",
    "parse_version_tuple",
]
