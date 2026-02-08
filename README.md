<p align="center">
  <img
    width="192"
    height="192"
    alt="A2M Logo"
    src="https://github.com/user-attachments/assets/00308900-94b8-48dc-a38a-12c058195bc1"
  />
</p>

<h1 align="center">A2M</h1>


<h3 align="center">Audio to MIDI</h3>

<p align="center">
  Convert local Audio files into MIDI files in just a few clicks.
</p>

<br/>

<p align="center">
  <a href="https://github.com/Justagwas/A2M/releases/latest/download/A2MSetup.exe">
    <img
      src="https://img.shields.io/badge/Download%20A2M%20for%20Windows-2563eb?style=for-the-badge&logo=windows&logoColor=white"
      alt="Download A2M for Windows"
    >
  </a>
</p>

<p align="center">
  <a href="https://Justagwas.com/projects/a2m">Website</a>
  &nbsp;&bull;&nbsp;
  <a href="https://github.com/Justagwas/A2M/releases">Releases</a>
  &nbsp;&bull;&nbsp;
  <a href="https://github.com/Justagwas/A2M/issues">Issues</a>
  &nbsp;&bull;&nbsp;
  <a href="https://github.com/Justagwas/A2M/blob/main/LICENSE">License</a>
</p>

---

## Overview

A2M is a desktop application that converts a local audio file into a MIDI file
using a trained transcription model.

- Choose a file
- Click Convert
- Get a MIDI file in your `Downloads/A2M` folder

It works best with clean, single-instrument input and is primarily tuned
for solo piano recordings.  

---

## Features

- Local-only processing (no uploads, no accounts)  
- CPU processing by default  
- Optional NVIDIA GPU acceleration  
- ML-based piano transcription  
- Standard MIDI output  
- Fully open source  

---

## Quick Start

1. Launch A2M
2. Choose an audio file
3. Click Convert

The generated MIDI file is saved to:

```
Downloads/A2M
```

---

## Model Download

A2M requires a transcription model. On first run, it will prompt you to download it (~165 MB). This is required for MIDI output.

## Model Source & Credit

This project uses the piano transcription model from:

- https://github.com/qiuqiangkong/piano_transcription_inference

Credits to the original authors and contributors of that project for the model and research work.

---

<details>
<summary><strong>For Developers</strong></summary>

### Configuration

Settings are stored in:

```
.a2m_config.json
```

Located in your user profile folder.

### Requirements

Core dependencies include:

- torch (CPU build recommended for distribution)
- librosa
- pretty_midi
- numpy
- pathvalidate

### Run From Source

```bash
pip install -r requirements.txt
python A2M.py
```

### Build (PyInstaller)

```bash
py -m PyInstaller -F -w -i "icon.ico" --clean A2M.py
```

The executable will be generated in the `dist/` directory.

</details>

---

## Security & Warnings

Operating systems may show warnings when downloading A2M because it is not yet widely recognized.

A2M is:

- Fully open source
- Local-only (no accounts, no telemetry)
- Operates only on files you choose

If downloaded from the official repository or release page, it can be independently verified and safely used.

---

## Contributing

Issues, suggestions, and pull requests are welcome.

If reporting bugs, please include clear reproduction steps.

---

## License

Apache-2.0

See `LICENSE` for details.

---

## Contact

[email@justagwas.com](mailto:email@justagwas.com)
