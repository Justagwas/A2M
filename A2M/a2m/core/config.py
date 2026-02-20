from __future__ import annotations
from pathlib import Path

APP_NAME = 'A2M - Audio To MIDI'
APP_SHORT_NAME = 'A2M'
APP_VERSION = '2.0.1'
OFFICIAL_PAGE_URL = 'https://www.justagwas.com/projects/a2m/download'
INNO_SETUP_APP_ID = 'A2MJustagwas'
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
MODEL_URL = 'https://downloads.justagwas.com/a2m/PianoModel.onnx'
MODEL_FILENAME = 'PianoModel.onnx'
MODEL_MIN_BYTES = 20000000
UPDATE_MANIFEST_URL = 'https://www.justagwas.com/projects/a2m/latest.json'
UPDATE_GITHUB_LATEST_URL = 'https://github.com/Justagwas/A2M/releases/latest/download/A2MSetup.exe'
UPDATE_GITHUB_DOWNLOAD_URL = 'https://downloads.justagwas.com/a2m/A2MSetup.exe'
UPDATE_SOURCEFORGE_RSS_URL = 'https://sourceforge.net/projects/a2m/rss?path=/'
UPDATE_CHECK_TIMEOUT_SECONDS = 12
CUDA_DOWNLOAD_URL = 'https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Windows&target_arch=x86_64'
CUDNN_DOWNLOAD_URL = 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.19.0.56_cuda12-archive.zip'
ONNX_CUDA_RUNTIME_PACK_URL = 'https://downloads.justagwas.com/a2m/a2m-onnx-cuda.zip'
ONNX_DML_RUNTIME_PACK_URL = 'https://downloads.justagwas.com/a2m/a2m-onnx-dml.zip'
DOWNLOAD_TIMEOUT_SECONDS = 45
DOWNLOAD_RETRIES_PER_HEADER = 3
DOWNLOAD_RETRY_BACKOFF_SECONDS = 1.2
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
