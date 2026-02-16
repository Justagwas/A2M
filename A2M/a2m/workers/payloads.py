from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelDownloadPayload:
    model_path: str


@dataclass(slots=True)
class RuntimePackDownloadPayload:
    provider: str
    runtime_path: str


@dataclass(slots=True)
class CudnnInstallPayload:
    bin_dir: str
    process_path_changed: bool
    user_path_changed: bool


@dataclass(slots=True)
class ConversionPayload:
    midi_path: str


@dataclass(slots=True)
class UpdateCheckPayload:
    manual: bool
    latest_version: str
    latest_display: str
    download_url: str
    update_available: bool
