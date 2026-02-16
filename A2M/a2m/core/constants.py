from __future__ import annotations
import json
from pathlib import Path
from typing import Any


_DEFAULT_APP_METADATA: dict[str, Any] = {
    'app_name': 'A2M - Audio To MIDI',
    'app_short_name': 'A2M',
    'app_version': '2.0.0',
    'official_page_url': 'https://justagwas.com/projects/a2m',
    'model_url': 'https://downloads.justagwas.com/a2m/PianoModel.onnx',
    'model_filename': 'PianoModel.onnx',
    'model_min_bytes': 20000000,
    'update_manifest_url': 'https://www.justagwas.com/projects/a2m/latest.json',
    'update_github_latest_url': 'https://github.com/Justagwas/A2M/releases/latest',
    'update_github_download_url': 'https://www.justagwas.com/projects/a2m/download',
    'update_sourceforge_rss_url': 'https://sourceforge.net/projects/a2m/rss?path=/',
    'update_check_timeout_seconds': 12,
    'cuda_download_url': 'https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Windows&target_arch=x86_64',
    'cudnn_download_url': 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.19.0.56_cuda12-archive.zip',
    'onnx_cuda_runtime_pack_url': 'https://downloads.justagwas.com/a2m/a2m-onnx-cuda.zip',
    'onnx_dml_runtime_pack_url': 'https://downloads.justagwas.com/a2m/a2m-onnx-dml.zip',
    'download_timeout_seconds': 45,
    'download_retries_per_header': 3,
    'download_retry_backoff_seconds': 1.2,
}


def _coerce_str(value: object, default: str) -> str:
    text = str(value or '').strip()
    return text or default


def _coerce_int(value: object, default: int, *, minimum: int=1) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    if parsed < minimum:
        return default
    return parsed


def _coerce_float(value: object, default: float, *, minimum: float=0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if parsed < minimum:
        return default
    return parsed


def _metadata_path() -> Path:
    return Path(__file__).resolve().with_name('app_config.json')


def load_app_metadata(path: Path | None=None) -> dict[str, Any]:
    source_path = path if path is not None else _metadata_path()
    merged = dict(_DEFAULT_APP_METADATA)
    try:
        payload = json.loads(source_path.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            merged.update(payload)
    except Exception:
        pass
    return {
        'app_name': _coerce_str(merged.get('app_name'), str(_DEFAULT_APP_METADATA['app_name'])),
        'app_short_name': _coerce_str(merged.get('app_short_name'), str(_DEFAULT_APP_METADATA['app_short_name'])),
        'app_version': _coerce_str(merged.get('app_version'), str(_DEFAULT_APP_METADATA['app_version'])),
        'official_page_url': _coerce_str(merged.get('official_page_url'), str(_DEFAULT_APP_METADATA['official_page_url'])),
        'model_url': _coerce_str(merged.get('model_url'), str(_DEFAULT_APP_METADATA['model_url'])),
        'model_filename': _coerce_str(merged.get('model_filename'), str(_DEFAULT_APP_METADATA['model_filename'])),
        'model_min_bytes': _coerce_int(merged.get('model_min_bytes'), int(_DEFAULT_APP_METADATA['model_min_bytes']), minimum=1),
        'update_manifest_url': _coerce_str(merged.get('update_manifest_url'), str(_DEFAULT_APP_METADATA['update_manifest_url'])),
        'update_github_latest_url': _coerce_str(merged.get('update_github_latest_url'), str(_DEFAULT_APP_METADATA['update_github_latest_url'])),
        'update_github_download_url': _coerce_str(merged.get('update_github_download_url'), str(_DEFAULT_APP_METADATA['update_github_download_url'])),
        'update_sourceforge_rss_url': _coerce_str(merged.get('update_sourceforge_rss_url'), str(_DEFAULT_APP_METADATA['update_sourceforge_rss_url'])),
        'update_check_timeout_seconds': _coerce_int(merged.get('update_check_timeout_seconds'), int(_DEFAULT_APP_METADATA['update_check_timeout_seconds']), minimum=1),
        'cuda_download_url': _coerce_str(merged.get('cuda_download_url'), str(_DEFAULT_APP_METADATA['cuda_download_url'])),
        'cudnn_download_url': _coerce_str(merged.get('cudnn_download_url'), str(_DEFAULT_APP_METADATA['cudnn_download_url'])),
        'onnx_cuda_runtime_pack_url': _coerce_str(merged.get('onnx_cuda_runtime_pack_url'), str(_DEFAULT_APP_METADATA['onnx_cuda_runtime_pack_url'])),
        'onnx_dml_runtime_pack_url': _coerce_str(merged.get('onnx_dml_runtime_pack_url'), str(_DEFAULT_APP_METADATA['onnx_dml_runtime_pack_url'])),
        'download_timeout_seconds': _coerce_int(merged.get('download_timeout_seconds'), int(_DEFAULT_APP_METADATA['download_timeout_seconds']), minimum=1),
        'download_retries_per_header': _coerce_int(merged.get('download_retries_per_header'), int(_DEFAULT_APP_METADATA['download_retries_per_header']), minimum=1),
        'download_retry_backoff_seconds': _coerce_float(merged.get('download_retry_backoff_seconds'), float(_DEFAULT_APP_METADATA['download_retry_backoff_seconds']), minimum=0.0),
    }


_APP_METADATA = load_app_metadata()
APP_NAME = str(_APP_METADATA['app_name'])
APP_SHORT_NAME = str(_APP_METADATA['app_short_name'])
APP_VERSION = str(_APP_METADATA['app_version'])
OFFICIAL_PAGE_URL = str(_APP_METADATA['official_page_url'])
CONFIG_FILENAME = 'a2m_config.json'
CONFIG_SCHEMA_VERSION = 2
GPU_BATCH_SIZE_MIN = 1
GPU_BATCH_SIZE_MAX = 32
UI_SCALE_PERCENT_MIN = 75
UI_SCALE_PERCENT_MAX = 200
MODERN_ONSET_THRESHOLD_MIN = 0.05
MODERN_ONSET_THRESHOLD_MAX = 0.90
MODERN_ONSET_THRESHOLD_DEFAULT = 0.30
MODERN_OFFSET_THRESHOLD_MIN = 0.05
MODERN_OFFSET_THRESHOLD_MAX = 0.90
MODERN_OFFSET_THRESHOLD_DEFAULT = 0.30
MODERN_FRAME_THRESHOLD_MIN = 0.01
MODERN_FRAME_THRESHOLD_MAX = 0.50
MODERN_FRAME_THRESHOLD_DEFAULT = 0.10
MODERN_PEDAL_OFFSET_THRESHOLD_MIN = 0.05
MODERN_PEDAL_OFFSET_THRESHOLD_MAX = 0.50
MODERN_PEDAL_OFFSET_THRESHOLD_DEFAULT = 0.20
MODEL_URL = str(_APP_METADATA['model_url'])
MODEL_FILENAME = str(_APP_METADATA['model_filename'])
MODEL_MIN_BYTES = int(_APP_METADATA['model_min_bytes'])
UPDATE_MANIFEST_URL = str(_APP_METADATA['update_manifest_url'])
UPDATE_GITHUB_LATEST_URL = str(_APP_METADATA['update_github_latest_url'])
UPDATE_GITHUB_DOWNLOAD_URL = str(_APP_METADATA['update_github_download_url'])
UPDATE_SOURCEFORGE_RSS_URL = str(_APP_METADATA['update_sourceforge_rss_url'])
UPDATE_CHECK_TIMEOUT_SECONDS = int(_APP_METADATA['update_check_timeout_seconds'])
CUDA_DOWNLOAD_URL = str(_APP_METADATA['cuda_download_url'])
CUDNN_DOWNLOAD_URL = str(_APP_METADATA['cudnn_download_url'])
ONNX_CUDA_RUNTIME_PACK_URL = str(_APP_METADATA['onnx_cuda_runtime_pack_url'])
ONNX_DML_RUNTIME_PACK_URL = str(_APP_METADATA['onnx_dml_runtime_pack_url'])
DOWNLOAD_TIMEOUT_SECONDS = int(_APP_METADATA['download_timeout_seconds'])
DOWNLOAD_RETRIES_PER_HEADER = int(_APP_METADATA['download_retries_per_header'])
DOWNLOAD_RETRY_BACKOFF_SECONDS = float(_APP_METADATA['download_retry_backoff_seconds'])
DOWNLOAD_HEADER_PROFILES = (
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Referer': 'https://justagwas.com/',
    },
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0',
        'Accept': '*/*',
        'Referer': 'https://justagwas.com/',
    },
    {
        'User-Agent': f'{APP_SHORT_NAME}/{APP_VERSION}',
        'Accept': '*/*',
    },
)
DOWNLOADS_DIR = Path.home() / 'Downloads'
OUTPUT_MIDI_DIR = DOWNLOADS_DIR / 'A2M'
SUPPORTED_AUDIO_FILTER = 'Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma *.aiff *.aif);;All Files (*.*)'
