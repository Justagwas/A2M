# A2M worked conversion example

This example follows one real piano recording from audio to score comparison.
The same 30 second excerpt can be heard, viewed in a log-frequency spectrogram,
and compared with both the written notes and the MIDI produced by A2M.

## Files

| File | Contents |
|---|---|
| [`a2m-demo-input.mp3`](a2m-demo-input.mp3) | The 30 second piano recording given to A2M |
| [`a2m-demo-source.mid`](a2m-demo-source.mid) | The score reference mapped to the active recording section |
| [`a2m-demo-output.mid`](a2m-demo-output.mid) | The MIDI produced by A2M 3.0.0 |
| [Conversion figure](../assets/demo-audio-midi-alignment.svg) | Waveform, log-frequency spectrogram, reference notes, and A2M output |

## Recording

The excerpt is the opening section of Johann Sebastian Bach's *Goldberg
Variations, Variation 1*, performed by Kimiko Ishizaka for the
[Open Goldberg Variations](https://opengoldbergvariations.org/) project. The
[recording is released under CC0](https://commons.wikimedia.org/wiki/File:Goldberg_Variations_02_Variatio_1_a_1_Clav.ogg),
which permits copying, modification, and distribution without a permission or
attribution requirement.

The reference MIDI represents the same notes from the public domain score. Its
timing is mapped from the written section to the active audio region from 0.15
seconds through 28.35 seconds. It is a comparison reference, not a recreation
of the pianist's exact timing or dynamics.

## Conditions

| Property | Value |
|---|---|
| Input | MP3, 44.1 kHz, stereo, 192 kbps |
| Duration | 30.0 seconds |
| Reference notes | 282 |
| A2M notes | 283 |
| A2M execution | CPU, Balanced mode |
| Velocity | Expressive |
| Pedal event export | Enabled |
| A2M version | 3.0.0 |
| Piano Engine bundle | 1.0.1 |

## Result

| Measure | Result |
|---|---|
| Same pitch onsets within 250 ms | 282 of 282 |
| Same pitch onsets within 100 ms | 233 of 282 |
| Mean absolute onset difference | 66 ms |
| Missed reference notes | 0 |
| Additional A2M notes | 1 |

The current engine recovered every reference pitch event within the stated 250
ms comparison window. It produced one additional note. In the spectrogram,
thin red markers indicate the temporal and pitch locations of note events
detected by A2M. The complete A2M piano roll is shown against the score
reference below it.

## Comparison method

1. The onset times in the first notated section were linearly mapped to the
   corresponding region of the audio recording.
2. Each reference note was matched to the nearest unmatched A2M note of the same pitch,
   provided that their onset times differed by no more than 250 ms.
3. Each A2M note could be paired once. An unmatched reference note counted as
   missed. An unmatched A2M note counted as additional.
4. The mean absolute onset error was calculated across all 282 matched note pairs.

The wider 250 ms boundary allows for expressive performance timing and the
section level score mapping. The 100 ms count is included to show the tighter
timing result as well.

This is one clean solo piano excerpt. It demonstrates the documentation and
conversion process, but it does not measure general performance across other
instruments, acoustic spaces, recording equipment, or repertoire.
