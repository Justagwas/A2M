from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import winreg  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - non-Windows
    winreg = None  # type: ignore[assignment]

_SEMVER_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_ALLOWED_HOSTS = {
    "justagwas.com",
    "www.justagwas.com",
}
_UPDATE_ALLOWED_HOSTS = {
    "github.com",
    "www.github.com",
    "sourceforge.net",
    "www.sourceforge.net",
    "justagwas.com",
    "www.justagwas.com",
    "downloads.justagwas.com",
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
}
_VALID_UPDATE_CHANNELS = {"stable", "nightly"}
_UPDATE_STAGING_PREFIX = "a2m-update-"


def normalize_version(version_text: str) -> str:
    text = str(version_text or "").strip()
    if text.lower().startswith("v"):
        text = text[1:]
    return text


def parse_semver(value: str) -> tuple[int, int, int] | None:
    match = _SEMVER_RE.search(str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def is_newer_version(latest_version: str, current_version: str) -> bool:
    latest = parse_semver(normalize_version(latest_version))
    current = parse_semver(normalize_version(current_version))
    if latest is None or current is None:
        return False
    return latest > current


def _ensure_not_stopped(stop_event: Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Update operation stopped.")


def _safe_int(value: object) -> int:
    try:
        parsed = int(value)
    except Exception:
        return 0
    return max(0, parsed)


def _sanitize_notes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    notes: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            notes.append(text)
    return notes


def _url_allowed(url_text: str, *, allowed_hosts: set[str]) -> bool:
    parsed = urlparse(str(url_text or "").strip())
    if parsed.scheme.lower() != "https":
        return False
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return False
    return host in allowed_hosts


def _sanitize_url(url_text: object, *, allowed_hosts: set[str]) -> str:
    candidate = str(url_text or "").strip()
    if candidate and _url_allowed(candidate, allowed_hosts=allowed_hosts):
        return candidate
    return ""


def _normalize_semver(value: str) -> str:
    parsed = parse_semver(normalize_version(value))
    if parsed is None:
        raise RuntimeError("did not contain a valid semantic version")
    return f"{parsed[0]}.{parsed[1]}.{parsed[2]}"


def _normalize_channel(value: object, *, default: str = "stable") -> str:
    candidate = str(value or "").strip().lower()
    if candidate in _VALID_UPDATE_CHANNELS:
        return candidate
    return default


def _sanitize_sha256(value: object) -> str:
    candidate = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(candidate):
        return candidate
    return ""


@dataclass(slots=True)
class UpdateManifest:
    version: str
    released: str = ""
    url: str = ""
    setup_url: str = ""
    setup_sha256: str = ""
    setup_size: int = 0
    notes: list[str] = field(default_factory=list)
    channel: str = "stable"
    minimum_supported_version: str = "1.0.0"


@dataclass(slots=True)
class UpdateCheckData:
    update_available: bool
    current_version: str
    latest_version: str
    page_url: str
    setup_url: str = ""
    setup_sha256: str = ""
    setup_size: int = 0
    released: str = ""
    notes: list[str] = field(default_factory=list)
    source: str = "latest.json"
    channel: str = "stable"
    minimum_supported_version: str = "1.0.0"
    requires_manual_update: bool = False
    setup_managed_install: bool = False

    @property
    def install_supported(self) -> bool:
        return bool(
            self.update_available
            and self.setup_url
            and self.setup_sha256
            and self.setup_size > 0
            and (not self.requires_manual_update)
            and self.setup_managed_install
            and sys.platform == "win32"
            and getattr(sys, "frozen", False)
        )


@dataclass(slots=True)
class PreparedUpdateInstall:
    latest_version: str
    setup_path: Path
    staging_root: Path
    requires_elevation: bool = False


class SelfUpdater:
    def __init__(
        self,
        *,
        app_name: str,
        app_version: str,
        manifest_url: str,
        page_url: str,
        setup_url: str,
        installer_app_id: str = "",
        executable_name: str,
        install_dir: Path,
        runtime_storage_dir: Path,
        timeout_seconds: float = 12.0,
        default_channel: str = "stable",
        request_retries: int = 3,
        manifest_request_retries: int | None = None,
        retry_backoff_seconds: float = 1.0,
        header_profiles: tuple[dict[str, str], ...] | None = None,
    ) -> None:
        self._app_name = str(app_name or "").strip() or "App"
        self._app_version = str(app_version or "").strip() or "0.0.0"
        self._manifest_url = str(manifest_url or "").strip()
        self._page_url = str(page_url or "").strip()
        self._default_setup_url = str(setup_url or "").strip()
        self._installer_app_id = str(installer_app_id or "").strip()
        self._exe_name = str(executable_name or "").strip() or f"{self._app_name}.exe"
        self._install_dir = Path(install_dir).resolve()
        self._runtime_storage_dir = Path(runtime_storage_dir).resolve()
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._request_retries = max(1, int(request_retries))
        if manifest_request_retries is None:
            self._manifest_request_retries = self._request_retries
        else:
            self._manifest_request_retries = max(1, int(manifest_request_retries))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._default_channel = _normalize_channel(
            os.environ.get("JUSTAGWAS_UPDATE_CHANNEL", default_channel),
            default="stable",
        )
        self._header_profiles = self._build_header_profiles(header_profiles)

    def recover_pending_update(self) -> None:
        try:
            self._cleanup_old_payloads()
        except Exception:
            return

    def check_for_updates(self, current_version: str, *, stop_event: Event | None = None) -> UpdateCheckData:
        current_normalized = normalize_version(current_version) or "0.0.0"
        if parse_semver(current_normalized) is None:
            current_normalized = "0.0.0"

        manifest = self._fetch_manifest(stop_event=stop_event)
        update_available = is_newer_version(manifest.version, current_normalized)
        minimum_supported = normalize_version(manifest.minimum_supported_version) or "1.0.0"
        minimum_tuple = parse_semver(minimum_supported)
        current_tuple = parse_semver(current_normalized)
        requires_manual = bool(minimum_tuple and current_tuple and current_tuple < minimum_tuple)
        return UpdateCheckData(
            update_available=update_available,
            current_version=current_normalized,
            latest_version=manifest.version,
            page_url=manifest.url or self._page_url,
            setup_url=manifest.setup_url,
            setup_sha256=manifest.setup_sha256,
            setup_size=manifest.setup_size,
            released=manifest.released,
            notes=list(manifest.notes),
            source="latest.json",
            channel=manifest.channel,
            minimum_supported_version=minimum_supported,
            requires_manual_update=requires_manual,
            setup_managed_install=self._is_setup_managed_install(),
        )

    def install_update(
        self,
        check_data: UpdateCheckData,
        *,
        stop_event: Event | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        prepared = self.prepare_update(
            check_data,
            stop_event=stop_event,
            progress_callback=progress_callback,
        )
        self.launch_prepared_update(prepared, restart_after_update=True)
        return str(check_data.latest_version or "")

    def prepare_update(
        self,
        check_data: UpdateCheckData,
        *,
        stop_event: Event | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> PreparedUpdateInstall:
        _ensure_not_stopped(stop_event)
        if sys.platform != "win32":
            raise RuntimeError("Self-update is currently supported on Windows only.")
        if not getattr(sys, "frozen", False):
            raise RuntimeError("Self-update is available only in packaged (frozen) builds.")
        if not check_data.setup_managed_install:
            raise RuntimeError("In-app update install is available only for setup-managed installs.")
        if check_data.requires_manual_update:
            raise RuntimeError("This version is below the minimum supported auto-update version.")
        if not check_data.setup_url:
            raise RuntimeError("No setup installer URL was provided in latest.json.")
        if not check_data.setup_sha256 or check_data.setup_size <= 0:
            raise RuntimeError("Setup installer metadata is incomplete in latest.json.")

        self._emit_progress(progress_callback, 0.0, "Preparing update...")
        target_root = self._create_staging_root()
        try:
            setup_name = self._setup_filename_from_url(check_data.setup_url, check_data.latest_version)
            setup_path = target_root / setup_name
            mode_arg = self._installer_mode_arg()

            self._download_setup(
                url=check_data.setup_url,
                sha256=check_data.setup_sha256,
                expected_size=check_data.setup_size,
                destination=setup_path,
                stop_event=stop_event,
                progress_callback=progress_callback,
            )
            _ensure_not_stopped(stop_event)
            self._emit_progress(progress_callback, 96.0, "Verifying update package...")
            self._emit_progress(progress_callback, 98.0, "Ready to install update.")
        except Exception:
            self._remove_tree(target_root)
            raise
        return PreparedUpdateInstall(
            latest_version=str(check_data.latest_version or ""),
            setup_path=setup_path,
            staging_root=target_root,
            requires_elevation=(mode_arg == "/ALLUSERS"),
        )

    def launch_prepared_update(
        self,
        prepared: PreparedUpdateInstall,
        *,
        restart_after_update: bool,
    ) -> None:
        try:
            self._launch_installer(
                prepared.setup_path,
                restart_after_update=bool(restart_after_update),
            )
        except Exception:
            self.discard_prepared_update(prepared)
            raise

    def discard_prepared_update(self, prepared: PreparedUpdateInstall) -> None:
        try:
            self._remove_tree(Path(prepared.staging_root))
        except Exception:
            return

    def _create_staging_root(self) -> Path:
        self._runtime_storage_dir.mkdir(parents=True, exist_ok=True)
        return Path(
            tempfile.mkdtemp(
                prefix=_UPDATE_STAGING_PREFIX,
                dir=str(self._runtime_storage_dir),
            )
        )

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[float, str], None] | None,
        percent: float,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        try:
            bounded = max(0.0, min(100.0, float(percent)))
            progress_callback(bounded, str(message or "").strip())
        except Exception:
            return

    def _fetch_manifest(self, *, stop_event: Event | None = None) -> UpdateManifest:
        if not _url_allowed(self._manifest_url, allowed_hosts=_MANIFEST_ALLOWED_HOSTS):
            raise RuntimeError("Manifest URL is missing or untrusted.")
        payload, _ = self._request_text(
            self._manifest_url,
            stop_event=stop_event,
            allowed_hosts=_MANIFEST_ALLOWED_HOSTS,
        )
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise RuntimeError("latest.json did not return a JSON object")

        requested_channel = _normalize_channel(data.get("channel"), default=self._default_channel)
        selected_channel = requested_channel
        source = data
        channels_payload = data.get("channels")
        if isinstance(channels_payload, dict):
            normalized_channels: dict[str, object] = {}
            for key, value in channels_payload.items():
                normalized_key = str(key or "").strip().lower()
                if normalized_key not in _VALID_UPDATE_CHANNELS:
                    continue
                normalized_channels[normalized_key] = value
            channel_candidate = normalized_channels.get(selected_channel)
            if not isinstance(channel_candidate, dict):
                selected_channel = "stable"
                channel_candidate = normalized_channels.get("stable")
            if isinstance(channel_candidate, dict):
                source = channel_candidate

        if isinstance(source, dict):
            selected_channel = _normalize_channel(source.get("channel"), default=selected_channel)

        def _pick_value(*keys: str) -> object:
            for key in keys:
                if key in source and source.get(key) not in (None, ""):
                    return source.get(key)
            for key in keys:
                if key in data and data.get(key) not in (None, ""):
                    return data.get(key)
            return ""

        version_candidate = source.get("version") or data.get("version") or ""
        if not isinstance(version_candidate, str):
            raise RuntimeError("latest.json did not contain a semantic version string")
        version = _normalize_semver(version_candidate)

        page_url = _sanitize_url(
            _pick_value("url"),
            allowed_hosts=_UPDATE_ALLOWED_HOSTS,
        ) or self._page_url
        setup_url = _sanitize_url(
            _pick_value("setup_url"),
            allowed_hosts=_UPDATE_ALLOWED_HOSTS,
        ) or _sanitize_url(self._default_setup_url, allowed_hosts=_UPDATE_ALLOWED_HOSTS)
        setup_sha256 = _sanitize_sha256(_pick_value("setup_sha256"))
        setup_size = _safe_int(_pick_value("setup_size"))
        released = str(_pick_value("released") or "").strip()
        notes = _sanitize_notes(_pick_value("notes"))
        minimum_supported = normalize_version(
            str(_pick_value("minimum_supported_version") or "1.0.0")
        ) or "1.0.0"
        if parse_semver(minimum_supported) is None:
            minimum_supported = "1.0.0"

        return UpdateManifest(
            version=version,
            released=released,
            url=page_url,
            setup_url=setup_url,
            setup_sha256=setup_sha256,
            setup_size=setup_size,
            notes=notes,
            channel=selected_channel,
            minimum_supported_version=minimum_supported,
        )

    @staticmethod
    def _sleep_with_stop(stop_event: Event | None, delay_seconds: float) -> None:
        remaining = max(0.0, float(delay_seconds))
        while remaining > 0.0:
            _ensure_not_stopped(stop_event)
            step = min(0.10, remaining)
            time.sleep(step)
            remaining -= step

    def _build_header_profiles(self, profiles: tuple[dict[str, str], ...] | None) -> tuple[dict[str, str], ...]:
        fallback = {
            "User-Agent": f"{self._app_name}/{self._app_version}",
            "Accept": "*/*",
        }
        sanitized: list[dict[str, str]] = []
        if isinstance(profiles, tuple):
            for raw_profile in profiles:
                if not isinstance(raw_profile, dict):
                    continue
                current: dict[str, str] = {}
                for key, value in raw_profile.items():
                    header_key = str(key or "").strip()
                    header_value = str(value or "").strip()
                    if header_key and header_value:
                        current[header_key] = header_value
                if current:
                    current.setdefault("User-Agent", fallback["User-Agent"])
                    current.setdefault("Accept", fallback["Accept"])
                    sanitized.append(current)
        if not sanitized:
            sanitized.append(dict(fallback))
        elif not any((profile.get("User-Agent") == fallback["User-Agent"] for profile in sanitized)):
            sanitized.append(dict(fallback))
        return tuple(sanitized)

    def _request_text(
        self,
        url: str,
        *,
        stop_event: Event | None = None,
        allowed_hosts: set[str] | None = None,
    ) -> tuple[str, str]:
        _ensure_not_stopped(stop_event)
        trusted_hosts = allowed_hosts if allowed_hosts is not None else _UPDATE_ALLOWED_HOSTS
        manifest_request = bool(
            str(url or "").strip() == self._manifest_url
            and trusted_hosts == _MANIFEST_ALLOWED_HOSTS
        )
        request_retries = self._manifest_request_retries if manifest_request else self._request_retries
        header_profiles = self._header_profiles[:1] if manifest_request else self._header_profiles
        last_error: Exception | None = None
        for attempt in range(request_retries):
            for headers in header_profiles:
                _ensure_not_stopped(stop_event)
                request = Request(url=url, headers=headers, method="GET")
                try:
                    with urlopen(request, timeout=self._timeout_seconds) as response:
                        final_url = str(response.geturl() or url)
                        if not _url_allowed(final_url, allowed_hosts=trusted_hosts):
                            raise RuntimeError("Update endpoint redirected to an untrusted host.")
                        body = response.read().decode("utf-8-sig", errors="replace")
                    _ensure_not_stopped(stop_event)
                    return body, final_url
                except InterruptedError:
                    raise
                except Exception as exc:
                    last_error = exc
            if attempt < request_retries - 1:
                self._sleep_with_stop(stop_event, self._retry_backoff_seconds * (2 ** attempt))
        if last_error is not None:
            raise RuntimeError(f"Update request failed: {last_error}")
        raise RuntimeError("Update request failed.")

    def _download_setup(
        self,
        *,
        url: str,
        sha256: str,
        expected_size: int,
        destination: Path,
        stop_event: Event | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        _ensure_not_stopped(stop_event)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        last_error: Exception | None = None
        self._emit_progress(progress_callback, 1.0, "Downloading update...")
        for attempt in range(self._request_retries):
            for headers in self._header_profiles:
                _ensure_not_stopped(stop_event)
                request = Request(url=url, headers=headers, method="GET")
                try:
                    with urlopen(request, timeout=max(5.0, self._timeout_seconds * 2.0)) as response:
                        final_url = str(response.geturl() or url)
                        if not _url_allowed(final_url, allowed_hosts=_UPDATE_ALLOWED_HOSTS):
                            raise RuntimeError("Update download redirected to an untrusted host.")
                        downloaded = 0
                        with tmp_path.open("wb") as handle:
                            while True:
                                _ensure_not_stopped(stop_event)
                                chunk = response.read(256 * 1024)
                                if not chunk:
                                    break
                                downloaded += len(chunk)
                                handle.write(chunk)
                                if expected_size > 0:
                                    fraction = max(0.0, min(1.0, downloaded / float(expected_size)))
                                    self._emit_progress(
                                        progress_callback,
                                        max(1.0, min(92.0, round(fraction * 92.0, 1))),
                                        f"Downloading update... {int(round(fraction * 100.0))}%",
                                    )
                    _ensure_not_stopped(stop_event)
                    actual_size = tmp_path.stat().st_size
                    if actual_size != expected_size:
                        raise RuntimeError(f"Downloaded size mismatch. Expected {expected_size}, got {actual_size}.")
                    actual_sha256 = self._sha256_file(tmp_path)
                    if actual_sha256.lower() != sha256.lower():
                        raise RuntimeError("SHA256 verification failed for downloaded setup installer.")
                    os.replace(tmp_path, destination)
                    self._emit_progress(progress_callback, 94.0, "Download complete.")
                    return
                except InterruptedError:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        destination.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise
                except Exception as exc:
                    last_error = exc
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        destination.unlink(missing_ok=True)
                    except Exception:
                        pass
            if attempt < self._request_retries - 1:
                self._sleep_with_stop(stop_event, self._retry_backoff_seconds * (2 ** attempt))
        if last_error is not None:
            raise RuntimeError(f"Update download failed: {last_error}")
        raise RuntimeError("Update download failed.")

    def _launch_installer(self, setup_path: Path, *, restart_after_update: bool) -> None:
        if not setup_path.exists():
            raise RuntimeError("Downloaded setup installer was not found.")
        mode_arg = self._installer_mode_arg()
        installer_args = [
            "/SILENT",
            "/SP-",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/CLOSEAPPLICATIONS",
            "/FORCECLOSEAPPLICATIONS",
            mode_arg,
            f"/DIR={str(self._install_dir)}",
        ]
        if not restart_after_update:
            installer_args.append("/SKIPLAUNCH")
        if mode_arg == "/ALLUSERS":
            self._launch_installer_elevated(setup_path, installer_args)
            return

        args = [str(setup_path), *installer_args]
        creationflags = 0
        creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        creationflags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
        try:
            subprocess.Popen(args, close_fds=True, creationflags=creationflags)
        except Exception as exc:
            raise RuntimeError(f"Unable to launch setup installer. {exc}") from exc

    def _launch_installer_elevated(self, setup_path: Path, installer_args: list[str]) -> None:
        try:
            import ctypes  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "Unable to request administrator privileges for all-users update."
            ) from exc

        arg_line = subprocess.list2cmdline(installer_args)
        operation = "runas"
        show_normal = 1
        try:
            result = int(
                ctypes.windll.shell32.ShellExecuteW(  # type: ignore[attr-defined]
                    None,
                    operation,
                    str(setup_path),
                    arg_line,
                    str(setup_path.parent),
                    show_normal,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to request administrator privileges for all-users update."
            ) from exc
        if result <= 32:
            if result in {5, 1223}:
                raise RuntimeError(
                    "Administrator permission was denied. Update canceled for all-users installation."
                )
            raise RuntimeError(f"Failed to start elevated installer (ShellExecuteW code {result}).")

    def _installer_mode_arg(self) -> str:
        registry_mode = self._installer_mode_arg_from_registry()
        if registry_mode is not None:
            return registry_mode

        install_dir_norm = self._normalize_path_for_compare(self._install_dir)
        if not install_dir_norm:
            return "/CURRENTUSER"

        all_users_roots = (
            os.environ.get("ProgramData", ""),
            os.environ.get("ProgramFiles", ""),
            os.environ.get("ProgramFiles(x86)", ""),
            os.environ.get("ProgramW6432", ""),
        )
        for root in all_users_roots:
            if self._path_is_within_root(install_dir_norm, root):
                return "/ALLUSERS"

        current_user_roots = (
            os.environ.get("LOCALAPPDATA", ""),
            os.environ.get("APPDATA", ""),
            os.environ.get("USERPROFILE", ""),
        )
        for root in current_user_roots:
            if self._path_is_within_root(install_dir_norm, root):
                return "/CURRENTUSER"

        return "/CURRENTUSER"

    @staticmethod
    def _path_is_within_root(path_norm: str, root: str) -> bool:
        root_norm = SelfUpdater._normalize_path_for_compare(root)
        if not root_norm:
            return False
        return path_norm == root_norm or path_norm.startswith(root_norm + os.sep)

    def _installer_mode_arg_from_registry(self) -> str | None:
        if sys.platform != "win32":
            return None
        if winreg is None:
            return None
        app_id = str(self._installer_app_id or "").strip()
        if not app_id:
            return None
        install_dir_norm = self._normalize_path_for_compare(self._install_dir)
        if not install_dir_norm:
            return None

        uninstall_base = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
        key_names: list[str] = [app_id]
        if not app_id.lower().endswith("_is1"):
            key_names.append(f"{app_id}_is1")

        hkey_current_user = getattr(winreg, "HKEY_CURRENT_USER", None)
        hkey_local_machine = getattr(winreg, "HKEY_LOCAL_MACHINE", None)
        hives = (hkey_current_user, hkey_local_machine)
        views = [0]
        wow64_64 = int(getattr(winreg, "KEY_WOW64_64KEY", 0))
        wow64_32 = int(getattr(winreg, "KEY_WOW64_32KEY", 0))
        if wow64_64:
            views.append(wow64_64)
        if wow64_32:
            views.append(wow64_32)

        key_read = int(getattr(winreg, "KEY_READ", 0))
        for hive in hives:
            if hive is None:
                continue
            for view in views:
                access = key_read | int(view)
                for key_name in key_names:
                    key_path = rf"{uninstall_base}\{key_name}"
                    values = self._read_uninstall_values(hive, key_path, access)
                    if values is None:
                        continue
                    if not self._registry_values_match_install_dir(values, install_dir_norm):
                        continue
                    if hive == hkey_local_machine:
                        return "/ALLUSERS"
                    if hive == hkey_current_user:
                        return "/CURRENTUSER"
        return None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(256 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest().lower()

    def _setup_filename_from_url(self, url: str, version: str) -> str:
        candidate = Path(urlparse(str(url or "").strip()).path).name
        if not candidate:
            candidate = f"{self._app_name}Setup-{normalize_version(version) or 'latest'}.exe"
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", candidate).strip("._")
        if not cleaned.lower().endswith(".exe"):
            cleaned = f"{cleaned}.exe"
        if not cleaned:
            cleaned = f"{self._app_name}Setup.exe"
        return cleaned

    def _cleanup_old_payloads(self) -> None:
        if not self._runtime_storage_dir.exists():
            return
        for child in self._runtime_storage_dir.iterdir():
            if not child.is_dir() or not child.name.startswith(_UPDATE_STAGING_PREFIX):
                continue
            self._remove_tree(child)

    @staticmethod
    def _remove_tree(root: Path) -> None:
        try:
            for walk_root, dirs, files in os.walk(root, topdown=False):
                for file_name in files:
                    try:
                        Path(walk_root, file_name).unlink(missing_ok=True)
                    except Exception:
                        pass
                for dir_name in dirs:
                    try:
                        Path(walk_root, dir_name).rmdir()
                    except Exception:
                        pass
            root.rmdir()
        except Exception:
            return

    def _is_setup_managed_install(self) -> bool:
        try:
            if not self._install_dir.exists():
                return False
            for pattern in ("unins*.exe", "unins*.dat"):
                for _item in self._install_dir.glob(pattern):
                    return True
            if self._is_setup_managed_install_registry():
                return True
            return False
        except Exception:
            return False

    def _is_setup_managed_install_registry(self) -> bool:
        return self._installer_mode_arg_from_registry() is not None

    def _read_uninstall_values(
        self,
        hive: object,
        key_path: str,
        access: int,
    ) -> dict[str, str] | None:
        try:
            with winreg.OpenKey(hive, key_path, 0, access) as key:  # type: ignore[arg-type,union-attr]
                return {
                    "InstallLocation": self._read_registry_string(key, "InstallLocation"),
                    "AppPath": self._read_registry_string(key, "Inno Setup: App Path"),
                    "DisplayIcon": self._read_registry_string(key, "DisplayIcon"),
                    "UninstallString": self._read_registry_string(key, "UninstallString"),
                }
        except OSError:
            return None

    @staticmethod
    def _read_registry_string(key: object, name: str) -> str:
        try:
            value, _ = winreg.QueryValueEx(key, name)  # type: ignore[union-attr]
        except OSError:
            return ""
        return str(value or "").strip()

    def _registry_values_match_install_dir(self, values: dict[str, str], install_dir_norm: str) -> bool:
        for candidate in (values.get("InstallLocation", ""), values.get("AppPath", "")):
            normalized = self._normalize_path_for_compare(candidate)
            if normalized and normalized == install_dir_norm:
                return True

        for candidate in (values.get("DisplayIcon", ""), values.get("UninstallString", "")):
            path_value = self._extract_path_from_command(candidate)
            if not path_value:
                continue
            parent_norm = self._normalize_path_for_compare(Path(path_value).parent)
            if parent_norm and parent_norm == install_dir_norm:
                return True
        return False

    @staticmethod
    def _extract_path_from_command(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith('"'):
            end_quote = text.find('"', 1)
            if end_quote > 1:
                return text[1:end_quote]
            return ""
        first_token = text.split(" ", 1)[0]
        return first_token.strip()

    @staticmethod
    def _normalize_path_for_compare(value: str | Path) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            resolved = Path(text).resolve()
            return os.path.normcase(os.path.normpath(str(resolved)))
        except Exception:
            return os.path.normcase(os.path.normpath(text))
