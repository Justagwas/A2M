<p align="center">
  <img
    width="256"
    height="256"
    alt="A2M Logo"
    src="https://github.com/user-attachments/assets/b602dc5c-d979-4389-a310-910944f9bbdb"
  />
</p>

<h1 align="center">A2M — Audio to MIDI</h1>

<p align="center">
  Convert local audio files into MIDI.<br/>
  No uploads, no accounts, no cloud processing.
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
  <a href="https://Justagwas.com/projects/a2m">Website</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/Justagwas/A2M/releases">Releases</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/Justagwas/A2M/issues">Issues</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/Justagwas/A2M/blob/main/LICENSE">License</a>
</p>

---

## Overview

A2M is a desktop application that converts a local audio file into a MIDI file
using a trained transcription model.

It is built to be straightforward and respectful of your data:
you choose a file, click convert, and receive a MIDI file you can immediately
use in a DAW or notation tool.

All processing happens locally on your machine.

---

## Who Is This For?

A2M is useful if you want to:

- Transcribe a piano recording into MIDI
- Extract notes from a practice take or demo
- Turn an idea into editable MIDI quickly
- Avoid uploading audio to online services

It works best with clean, single-instrument input and is primarily tuned
for solo piano recordings.

---

## How It Works

1. Launch A2M  
2. Select an audio file  
3. Click **Convert**

The generated MIDI file is saved to:

```
Downloads/A2M
```

---

## Features

- Local-only processing (no uploads, no accounts)
- CPU processing by default
- Optional NVIDIA GPU acceleration
- ML-based piano transcription
- Standard MIDI output
- Fully open source

---

## Model Download

A2M requires a transcription model to generate MIDI.

On first launch, the app will prompt you to download the model (~165 MB).
This download is required and only happens once.

---

## Model Source and Credit

A2M uses the piano transcription model from:

- https://github.com/qiuqiangkong/piano_transcription_inference

Full credit goes to the original authors and contributors for their research
and implementation. A2M provides a desktop interface around this work.

---

<details>
<summary><strong>For Developers</strong></summary>

### Configuration

User settings are stored in:

```
.a2m_config.json
````

This file is located in your user profile directory.

---

### Core Dependencies

- torch (CPU build recommended for distribution)
- librosa
- pretty_midi
- numpy
- pathvalidate

---

### Run From Source

```bash
pip install -r requirements.txt
python A2M.py
````

---

### Build (PyInstaller)

```bash
py -m PyInstaller -F -w -i "icon.ico" --clean A2M.py
```

The executable will be generated in the `dist/` directory.

</details>

---

## Security and OS Warnings

Operating systems may show warnings when downloading or running A2M because
it is not yet widely recognized.

A2M is:

* Fully open source
* Local-only (no telemetry, no background network activity)
* Limited to processing files you explicitly choose

If downloaded from the official GitHub releases page, the application can be
inspected, built from source, or safely used as-is.

---

## Contributing

Issues, feature suggestions, and pull requests are welcome.

If reporting a bug, please include clear steps to reproduce the issue.

---

## License

Apache License 2.0

See the `LICENSE` file for details.

---

## Contact

For questions or feedback:

[email@justagwas.com](mailto:email@justagwas.com)
