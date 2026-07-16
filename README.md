# A2M - Audio to MIDI

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

<p align="center">Transcribes locally on Windows, CPU-first by design, with optional CUDA or DirectML acceleration</p>

<div align="center">

[![Version](https://img.shields.io/github/v/tag/Justagwas/A2M.svg?label=Version)](https://github.com/Justagwas/A2M/tags)
[![License](https://img.shields.io/github/license/Justagwas/A2M.svg)](https://github.com/Justagwas/A2M/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Justagwas/A2M/main.svg?style=flat&cacheSeconds=3600)](https://github.com/Justagwas/A2M/commits/main)
[![Open Issues](https://img.shields.io/github/issues/Justagwas/A2M.svg)](https://github.com/Justagwas/A2M/issues)
[![Stars](https://img.shields.io/github/stars/Justagwas/A2M.svg?style=flat&cacheSeconds=3600)](https://github.com/Justagwas/A2M/stargazers)
![Installs (7d)](https://img.shields.io/badge/dynamic/json?style=flat&url=https%3A%2F%2Fdownload-stats-worker.justagwas.workers.dev%2Fdownloads%2Fa2m%3Frange%3Dweek&query=%24.data.weekly.all&label=Installs%20(7d))

</div>

## Overview

A2M is a Windows desktop application designed to achieve high transcription accuracy when converting piano recordings into editable MIDI files. The A2M Piano Engine estimates notes, timing, velocity, sustain, and soft pedal events locally, using CPU processing by default with optional CUDA or DirectML acceleration. Configurable CPU usage and GPU memory settings allow system load to be balanced against processing speed. Although runtime depends on recording duration and available hardware, transcription typically completes promptly on suitable systems. Clean solo piano recordings provide the best results.

## Basic usage

1. Download and install from the [latest release](https://github.com/Justagwas/A2M/releases/latest/download/A2MSetup.exe).
2. On first launch, A2M assesses the available hardware and selects appropriate default CPU and GPU resource settings.
3. If prompted, approve the initial transcription model download.
4. Click **Choose Audio** and select a local file.
5. If supported, optionally select GPU execution in **Settings**.
6. Click **Convert to MIDI**.
7. Open the result from the configured save location or by selecting **Open output folder** in the footer.

## Features

- Local transcription of polyphonic piano recordings.
- A file picker that recognizes MP3, WAV, FLAC, OGG, M4A, AAC, WMA, AIFF, and AIF files, subject to codec availability.
- A consistent A2M Piano Engine for note, timing, velocity, sustain pedal, and soft pedal estimation.
- ONNX Runtime inference on CPU, CUDA, or DirectML.
- CPU usage controls and GPU memory controls scaled to the available hardware.
- Optional expressive or uniform MIDI velocity and optional pedal export.
- Verified model and runtime packages with integrated dependency installation.
- Configurable output location, interface scaling, and update checks.
- Cancellation support during downloads, dependency installation, and transcription.

## Technical overview

### Audio and feature processing

The source recording is decoded with SoundFile, with audioread available as a fallback. Audio is resampled to 44,100 Hz and analyzed in centered frames of 4,096 samples with a 1,024 sample hop. Six window sizes provide complementary spectral views, which are projected into 229 mel bands and log normalized for model inference.

Recordings are processed in overlapping sections of 16 seconds that advance by 8 seconds. This overlap supplies context near section boundaries and allows section results to be reconstructed on the original recording timeline.

### Inference and event decoding

The scorer model evaluates possible start and end frame pairs for 88 piano pitches and two pedal tracks. A dynamic programming decoder selects coherent intervals, after which the attribute model estimates velocity and subframe timing adjustments. Events from successive sections are then joined, resolved, and clipped to the duration of the source recording.

### Runtime and acceleration

- A2M runs on CPU by default.
- CUDA and DirectML execute the same models and decoding process as CPU mode.
- Optional GPU runtime packs are installed for the current user under:
  - `%LOCALAPPDATA%\A2M\runtime_packs\cuda`
  - `%LOCALAPPDATA%\A2M\runtime_packs\dml`
- Runtime-pack endpoints are defined in [`A2M/a2m/core/config.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/config.py).

### MIDI construction

Decoded events are ordered and written as a 960 PPQ MIDI sequence. Notes retain their estimated pitch, timing, and selected velocity treatment. Sustain and soft pedal controller events can be included or excluded before export. A2M assigns a unique output filename so an existing transcription is not overwritten unintentionally.

The model package is obtained from `https://downloads.justagwas.com/a2m/PianoModel.a2m` when required and retained for subsequent use. Its package structure and internal components are validated before inference. CPU execution is included with the application dependencies, while optional GPU components remain separate.

For a visual and detailed technical account of this process, see [How A2M Works](https://github.com/Justagwas/A2M/wiki/How-A2M-Works), the [Audio Processing Pipeline](https://github.com/Justagwas/A2M/wiki/Audio-Processing-Pipeline), and [From Model Output to MIDI](https://github.com/Justagwas/A2M/wiki/From-Model-Output-to-MIDI).

## Project resources

- Project page: <https://www.justagwas.com/projects/a2m>
- Download page: <https://www.justagwas.com/projects/a2m/download>
- Documentation: <https://github.com/Justagwas/A2M/wiki>
- Releases: <https://github.com/Justagwas/a2m/releases>

<details>
<summary>For Developers</summary>

### Requirements

- Windows (primary runtime target).
- Python 3.11 or later.
- Dependencies in [`A2M/requirements.txt`](https://github.com/Justagwas/A2M/blob/main/A2M/requirements.txt).

### Running From Source

```powershell
cd A2M
py -m pip install -r requirements.txt
py A2M.py
```

### Configuration Files

- Application metadata, constants, and service endpoints: [`A2M/a2m/core/config.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/config.py)
- Runtime setting serialization and normalization: [`A2M/a2m/core/config_service.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/config_service.py)
- Runtime and path resolution: [`A2M/a2m/core/paths.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/paths.py)
- Model acquisition, storage, and validation: [`A2M/a2m/core/model_service.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/model_service.py)
- GPU runtime pack management: [`A2M/a2m/core/runtime_pack_service.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/runtime_pack_service.py)
- Audio analysis and event decoding: [`A2M/a2m/core/piano_engine.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/piano_engine.py)
- MIDI construction and output naming: [`A2M/a2m/core/conversion_service.py`](https://github.com/Justagwas/A2M/blob/main/A2M/a2m/core/conversion_service.py)

</details>

## Security and Windows warnings

- Windows SmartScreen may display a warning for a new or unsigned release.
- Obtain A2M only from the official distribution locations:
  - <https://github.com/Justagwas/a2m/releases>
  - <https://www.justagwas.com/projects/a2m>
  - <https://sourceforge.net/projects/a2m/>
- Refer to [`.github/SECURITY.md`](https://github.com/Justagwas/A2M/blob/main/.github/SECURITY.md) for the security policy and private vulnerability reporting procedure.

## Contributing

Contributions are welcome.

- Start with [`.github/CONTRIBUTING.md`](https://github.com/Justagwas/A2M/blob/main/.github/CONTRIBUTING.md)
- Follow [`.github/CODE_OF_CONDUCT.md`](https://github.com/Justagwas/A2M/blob/main/.github/CODE_OF_CONDUCT.md)
- Use [Issues](https://github.com/Justagwas/A2M/issues) for defect reports, feature proposals, and questions
- Wiki: <https://github.com/Justagwas/A2M/wiki>

## License

Licensed under the GNU General Public License v3.0 (GPL-3.0).

See [`LICENSE`](https://github.com/Justagwas/A2M/blob/main/LICENSE).

## Contact

- Email: [email@justagwas.com](mailto:email@justagwas.com)
- Website: <https://www.justagwas.com/projects/a2m>
