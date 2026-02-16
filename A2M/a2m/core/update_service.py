from __future__ import annotations
import json
import re
from threading import Event
from xml.etree import ElementTree as ET
from .constants import DOWNLOAD_HEADER_PROFILES, OFFICIAL_PAGE_URL, UPDATE_CHECK_TIMEOUT_SECONDS, UPDATE_GITHUB_DOWNLOAD_URL, UPDATE_GITHUB_LATEST_URL, UPDATE_MANIFEST_URL, UPDATE_SOURCEFORGE_RSS_URL
from .http_service import fetch_text

def _pick_string_field(payload: object, *keys: str) -> str:
    if not isinstance(payload, dict):
        return ''
    normalized: dict[str, object] = {}
    for key, value in payload.items():
        normalized[str(key or '').strip().lower()] = value
    for key in keys:
        value = normalized.get(str(key or '').strip().lower())
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ''

def _normalize_download_url(url_text: str) -> str:
    text = str(url_text or '').strip()
    if text:
        return text
    fallback = str(UPDATE_GITHUB_DOWNLOAD_URL or '').strip()
    if fallback:
        return fallback
    return str(OFFICIAL_PAGE_URL or '').strip()

def _fetch_text(url: str, *, stop_event: Event | None=None) -> tuple[str, str]:
    return fetch_text(
        url,
        headers_profiles=DOWNLOAD_HEADER_PROFILES,
        timeout_seconds=UPDATE_CHECK_TIMEOUT_SECONDS,
        stop_event=stop_event,
        stop_message='Update check stopped.',
    )

def _fetch_json(url: str, *, stop_event: Event | None=None) -> dict | None:
    payload, _ = _fetch_text(url, stop_event=stop_event)
    data = json.loads(payload)
    return data if isinstance(data, dict) else None

def _extract_version(value: object) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    candidate = text
    match = re.search('(\\d+(?:\\.\\d+)+)', text)
    if match:
        candidate = match.group(1)
    parsed = parse_version_tuple(candidate)
    if not parsed:
        return ''
    return '.'.join((str(part) for part in parsed))

def normalize_version(version_text: str) -> str:
    text = str(version_text or '').strip()
    if text.lower().startswith('v'):
        text = text[1:]
    return text

def parse_version_tuple(version_text: str) -> tuple[int, ...] | None:
    normalized = normalize_version(version_text)
    if not normalized or not re.match('^\\d+(\\.\\d+)*$', normalized):
        return None
    return tuple((int(part) for part in normalized.split('.')))

def is_newer_version(latest_version: str, current_version: str) -> bool:
    latest_tuple = parse_version_tuple(latest_version)
    current_tuple = parse_version_tuple(current_version)
    if not latest_tuple or not current_tuple:
        return False
    width = max(len(latest_tuple), len(current_tuple))
    latest_tuple = latest_tuple + (0,) * (width - len(latest_tuple))
    current_tuple = current_tuple + (0,) * (width - len(current_tuple))
    return latest_tuple > current_tuple

def fetch_update_manifest(stop_event: Event | None=None) -> tuple[str, str]:
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError('Update check stopped.')
    errors: list[str] = []
    try:
        data = _fetch_json(UPDATE_MANIFEST_URL, stop_event=stop_event)
        if isinstance(data, dict):
            nested = data.get('latest')
            manifest = nested if isinstance(nested, dict) else data
            latest_version = _pick_string_field(manifest, 'version', 'latest', 'app_version', 'latest_version')
            download_url = _pick_string_field(manifest, 'download_url', 'url', 'download')
            normalized_version = _extract_version(latest_version)
            if normalized_version:
                return (normalized_version, _normalize_download_url(download_url))
        errors.append('Manifest is missing a valid version.')
    except InterruptedError:
        raise
    except Exception as exc:
        errors.append(str(exc))
    github_latest_url = str(UPDATE_GITHUB_LATEST_URL or '').strip()
    if github_latest_url:
        try:
            body, final_url = _fetch_text(github_latest_url, stop_event=stop_event)
            for source in (final_url, body):
                match = re.search('/tag/v?(\\d+(?:\\.\\d+)+)', source or '')
                if not match:
                    match = re.search('/releases/tag/v?(\\d+(?:\\.\\d+)+)', source or '')
                if match:
                    return (match.group(1), _normalize_download_url(''))
            errors.append('Could not parse version from GitHub latest release.')
        except InterruptedError:
            raise
        except Exception as exc:
            errors.append(str(exc))
    sourceforge_rss_url = str(UPDATE_SOURCEFORGE_RSS_URL or '').strip()
    if sourceforge_rss_url:
        try:
            rss_text, _ = _fetch_text(sourceforge_rss_url, stop_event=stop_event)
            xml_root = ET.fromstring(rss_text)
            titles = xml_root.findall('.//item/title')
            for title in titles:
                normalized_version = _extract_version(title.text or '')
                if normalized_version:
                    return (normalized_version, _normalize_download_url(''))
            errors.append('Could not parse version from SourceForge RSS.')
        except InterruptedError:
            raise
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(errors[-1])
    raise RuntimeError('Could not parse latest version from update sources.')
