<p align="center">
  <img
    width="192"
    height="192"
    alt="A2M Logo"
    src="https://github.com/user-attachments/assets/00308900-94b8-48dc-a38a-12c058195bc1"
  />
</p>

<h1 align="center">A2M</h1>

<h3 align="center">Desktop audio-to-MIDI transcription for piano-focused recordings</h3>

<p align="center">
  Convert local audio files into MIDI files in a few clicks<br/>
  Runs locally on your machine with no cloud upload
</p>

<p align="center">
  <a href="https://github.com/Justagwas/A2M/releases/latest/download/A2MSetup.exe">
    <img
      src="https://img.shields.io/badge/Download%20for%20Windows-2563eb?style=for-the-badge&logo=windows&logoColor=white"
      alt="Download A2M for Windows"
    />
  </a>
</p>

<p align="center">
  <a href="https://justagwas.com/projects/a2m">Website</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/Justagwas/A2M/releases">Releases</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/Justagwas/A2M/issues">Issues</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/Justagwas/A2M/wiki">Documentation</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/Justagwas/A2M/blob/main/LICENSE">License</a>
</p>

## Overview

A2M is a Windows desktop app that transcribes local audio files into `.mid` files.

- Input: local audio file (`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.aiff`, `.aif`)
- Output: MIDI file in `Downloads/A2M`
- Model: `PianoModel.pth` (downloaded once, about 165 MB)

The app is built with the bundled `piano_transcription_inference` codebase:
https://github.com/qiuqiangkong/piano_transcription_inference

## Basic usage

1. Launch A2M.
2. Click **Choose Audio** and pick a local file.
3. If prompted, download the transcription model.
4. Click **Convert to MIDI**.
5. Open `Downloads/A2M` to access the exported `.mid` file.

## Features

- Local transcription workflow (no account and no cloud upload for audio files)
- First-run guided model download with progress + fallback install path
- Stop button for in-progress transcription
- In-app progress bar and status console
- Automatic/manual update checks against the project update manifest
- Dark/light theme toggle and UI size scaling
- Safe output naming (invalid characters removed and duplicate filenames auto-suffixed)

## Model download and storage

- Model URL is:
  `https://downloads.justagwas.com/a2m/PianoModel.pth`
- A2M first tries to store `PianoModel.pth` next to the app.
- If the app directory is not writable, it falls back to:
  `%USERPROFILE%\piano_transcription_inference_data`

## Runtime and conversion behavior

- PyTorch must be available at runtime for conversion to run.
- Current runtime behavior is CPU-only; GPU mode is shown in UI but intentionally disabled in code.
- MIDI post-processing removes very short/low-velocity notes and overlapping duplicates per pitch.

## Preview

- Project page with full preview gallery: [justagwas.com/projects/a2m](https://www.justagwas.com/projects/a2m)
- OpenPiano Installer Download link: [justagwas.com/projects/a2m/download](https://www.justagwas.com/projects/a2m/download)

<details><summary>For Developers</summary>

### Requirements

- Python 3.10+ (Windows recommended for full parity with release behavior)
- Dependencies from:
  https://github.com/Justagwas/A2M/blob/main/A2M/requirements.txt

### Running From Source
```powershell
cd A2M
py -m pip install -r requirements.txt
py A2M.py
```

</details>

## Security and OS Warnings

- A2M is open source and intended for local desktop use.
- Windows SmartScreen may display "Protected your PC" for newer/unsigned binaries.
- Only run installers downloaded from official A2M release links.

Note: audio conversion runs locally, but A2M uses network access for model download and update checks.

## Contributing

- Contribution guide:
  https://github.com/Justagwas/A2M/blob/main/.github/CONTRIBUTING.md
- Open issues:
  https://github.com/Justagwas/A2M/issues
- Issue templates:
  https://github.com/Justagwas/A2M/tree/main/.github/ISSUE_TEMPLATE
- Code of Conduct:
  https://github.com/Justagwas/A2M/blob/main/.github/CODE_OF_CONDUCT.md

## License

Licensed under Apache-2.0:
https://github.com/Justagwas/A2M/blob/main/LICENSE

## Contact

- Email: [email@justagwas.com](mailto:email@justagwas.com)
- Website: <https://www.justagwas.com/projects/a2m>