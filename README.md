# A2M

<div align="center">

[![Code: GitHub](https://img.shields.io/badge/Code-GitHub-111827.svg?style=flat&logo=github&logoColor=white)](https://github.com/Justagwas/A2M)
[![Website](https://img.shields.io/badge/Website-A2M-0ea5e9.svg?style=flat&logo=google-chrome&logoColor=white)](https://Justagwas.com/projects/a2m)
[![Mirror: SourceForge](https://img.shields.io/badge/Mirror-SourceForge-ff6600.svg?style=flat&logo=sourceforge&logoColor=white)](https://sourceforge.net/projects/a2m/)

</div>

<p align="center">
  <img
    width="128"
    height="128"
    alt="A2M Logo"
    src="https://github.com/user-attachments/assets/64a4aa9f-e130-4eb7-9f29-77fc449567f6"
  />
</p>

<div align="center">

[![Download (Windows)](https://img.shields.io/badge/Download-Windows%20(A2MSetup.exe)-2563eb.svg?style=flat&logo=windows&logoColor=white)](https://github.com/Justagwas/A2M/releases/latest/download/A2MSetup.exe)

</div>

<p align="center"><b>Desktop audio-to-MIDI conversion tailored for piano recordings</b></p>

<p align="center">Runs entirely offline on Windows, CPU-first by design, with optional CUDA or DirectML acceleration</p>


<div align="center">

[![Version](https://img.shields.io/github/v/tag/Justagwas/A2M.svg?label=Version)](
https://github.com/Justagwas/A2M/tags)
[![Last Commit](https://img.shields.io/github/last-commit/Justagwas/A2M/main.svg?style=flat&cacheSeconds=3600)](
https://github.com/Justagwas/A2M/commits/main)
[![Stars](https://img.shields.io/github/stars/Justagwas/A2M.svg?style=flat&cacheSeconds=3600)](
https://github.com/Justagwas/A2M/stargazers)
[![Open Issues](https://img.shields.io/github/issues/Justagwas/A2M.svg)](
https://github.com/Justagwas/A2M/issues)
[![License](https://img.shields.io/github/license/Justagwas/A2M.svg)](
https://github.com/Justagwas/A2M/blob/main/LICENSE)

</div>

## Overview

A2M (Audio to MIDI) is a Windows desktop app that converts local, piano-focused audio into `.mid` files.

The app uses ONNX Runtime, supports CPU by default, and can switch to optional GPU acceleration through runtime packs (CUDA or DirectML). It includes two transcription engines in settings: `Legacy v1.0.0` and `Modern v2.0.0` (Legacy default).

## Basic usage

1. Download and install from the [latest release](https://github.com/Justagwas/A2M/releases/latest/download/A2MSetup.exe).
2. If prompted, allow the model download.
3. Click **Choose audio** and select a local file.
4. Select CPU or GPU mode in settings.
5. Click **Convert to MIDI**.
6. Open the output folder from the footer (**Open downloads folder**) or from your configured save path.

## Features

- Local audio-to-MIDI transcription workflow.
- ONNX Runtime inference pipeline.
- CPU-first behavior with optional GPU runtime packs.
- GPU provider preference options: `Auto`, `CUDA`, `DirectML`.
- In-app runtime-pack installation for GPU dependencies.
- Legacy and Modern transcription engines.
- Modern tuning options for adaptive thresholds, input normalization/denoise, overlap stitching, and auto calibration.
- Output location controls, UI scale controls, and update checks.
- Stop/cancel handling for model download, runtime-pack download, cuDNN install, and conversion.

## Feature sections

### Runtime and Acceleration

- A2M runs on CPU by default.
- Optional GPU runtime packs are installed per-user under:
  - `%LOCALAPPDATA%\A2M\runtime_packs\cuda`
  - `%LOCALAPPDATA%\A2M\runtime_packs\dml`
- Runtime-pack endpoints are defined in [`A2M/a2m/core/app_config.json`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/app_config.json).

### Transcription Engines

- `Legacy v1.0.0` is the default engine.
- `Modern v2.0.0` exposes additional behavior controls and diagnostics.
- Modern controls are shown contextually in settings based on selected engine/calibration mode.

## Preview

- Project page: <https://www.justagwas.com/projects/a2m>
- Download page: <https://www.justagwas.com/projects/a2m/download>
- Releases: <https://github.com/Justagwas/a2m/releases>

<details>
<summary>For Developers</summary>

### Requirements

- Windows (primary runtime target).
- Python 3.11+.
- Dependencies in [`A2M/requirements.txt`](https://github.com/Justagwas/A2M/blob/main/A2M/requirements.txt).

### Running From Source

```powershell
cd A2M
py -m pip install -r requirements.txt
py A2M.py
```

### Configuration Files

- App metadata and release/update/runtime URLs: [`A2M/a2m/core/app_config.json`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/app_config.json)
- App constants and defaults: [`A2M/a2m/core/constants.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/constants.py)
- Runtime settings serialization and normalization: [`A2M/a2m/core/config_service.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/config_service.py)
- Runtime/path resolution helpers: [`A2M/a2m/core/paths.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/paths.py)
- Model download/storage logic: [`A2M/a2m/core/model_service.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/model_service.py)
- GPU runtime-pack management: [`A2M/a2m/core/runtime_pack_service.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/runtime_pack_service.py)

</details>

## Security and OS Warnings

- Windows SmartScreen may show warnings for newer or unsigned binaries.
- Download from official links only:
  - <https://github.com/Justagwas/a2m/releases>
  - <https://www.justagwas.com/projects/a2m>
  - <https://sourceforge.net/projects/a2m/>
- Security policy and private vulnerability reporting: [`.github/SECURITY.md`](https://github.com/Justagwas/A2M/blob/main/.github/SECURITY.md)

## Contributing

Contributions are welcome.

- Start with [`.github/CONTRIBUTING.md`](https://github.com/Justagwas/A2M/blob/main/.github/CONTRIBUTING.md)
- Follow [`.github/CODE_OF_CONDUCT.md`](https://github.com/Justagwas/A2M/blob/main/.github/CODE_OF_CONDUCT.md)
- Use [Issues](https://github.com/Justagwas/A2M/issues) for bugs, requests, and questions
- Wiki: <https://github.com/Justagwas/A2M/wiki>

## License

Licensed under the Apache License 2.0.

See [`LICENSE`](https://github.com/Justagwas/A2M/blob/main/LICENSE).

## Contact

- Email: [email@justagwas.com](mailto:email@justagwas.com)
- Website: <https://www.justagwas.com/projects/a2m>
