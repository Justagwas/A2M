# A2M documentation

This directory contains the original figures and examples used by the
A2M README, Wiki, and project page documentation.

## Figures

| Figure | Purpose |
|---|---|
| `pipeline-overview.svg` | Complete audio to MIDI path based on the current source |
| `feature-frontend.svg` | Framing, spectral power, mel projection, and normalization formulas |
| `interval-decoding.svg` | Conceptual interval scores, Viterbi decoding, and exact timing refinement |
| `segment-stitching.svg` | Sixteen second sections, eight second movement, and overlap handling |
| `demo-audio-midi-alignment.svg` | Real waveform, log-frequency spectrogram, score reference, and A2M output |

The diagrams are original A2M project assets. Conceptual panels are identified
as conceptual. Dimensions, formulas, and behavioral claims come from the
current application code and model manifest.

## Worked example

The [`examples`](examples/) directory contains a 30 second real piano excerpt,
a score reference mapped to the recording, and the MIDI produced by A2M 3.0.0.
The accompanying figure connects the sound, its frequency content, the written
notes, and the application result on one shared time axis.

The example is meant for listening and visual inspection. It is not a claim
about accuracy across every piano, room, microphone, or recording style.
