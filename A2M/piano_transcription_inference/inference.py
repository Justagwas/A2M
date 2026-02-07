import os
import numpy as np

try:
    import torch
except Exception:
    torch = None

from .utilities import (RegressionPostProcessor, write_events_to_midi)
from . import config
from .model_paths import MODEL_MIN_BYTES, MODEL_DOWNLOAD_URL, resolve_checkpoint_path

if torch is not None:
    from .models import Regress_onset_offset_frame_velocity_CRNN, Note_pedal
    from .pytorch_utils import forward
else:
    Regress_onset_offset_frame_velocity_CRNN = None
    Note_pedal = None
    forward = None


class PianoTranscription(object):
    def __init__(self, model_type='Note_pedal', checkpoint_path=None, 
        segment_samples=16000*10, device=None, batch_size=1):
        """Class for transcribing piano solo recording.

        Args:
          model_type: str
          checkpoint_path: str
          segment_samples: int
          device: 'cuda' | 'cpu'
          batch_size: int
        """
        if torch is None:
            raise RuntimeError("PyTorch is required for PianoTranscription. Install it first.")
        if isinstance(device, str):
            device_str = device.strip().lower()
            if device_str == 'cpu':
                device = torch.device('cpu')
            elif device_str in ('gpu', 'cuda'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device is None:
            env_pref = os.environ.get('A2M_DEVICE') or os.environ.get('A2M_DEVICE_PREFERENCE')
            if env_pref:
                env_pref = env_pref.strip().lower()
                if env_pref == 'cpu':
                    device = torch.device('cpu')
                elif env_pref in ('gpu', 'cuda'):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        print('Checkpoint path: {}'.format(checkpoint_path))

        if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) < MODEL_MIN_BYTES:
            raise RuntimeError(
                "Model checkpoint not found or incomplete. "
                f"Download it via the A2M app ({MODEL_DOWNLOAD_URL}) and retry."
            )

        print('Using {} for inference.'.format(device))
        try:
            batch_size = int(batch_size)
        except Exception:
            batch_size = 1
        if batch_size < 1:
            batch_size = 1
        self.batch_size = batch_size

        self.segment_samples = segment_samples
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.onset_threshold = 0.3
        self.offset_threshod = 0.3
        self.frame_threshold = 0.1
        self.pedal_offset_threshold = 0.2

        # Build model
        Model = eval(model_type)
        self.model = Model(frames_per_second=self.frames_per_second, 
            classes_num=self.classes_num)

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # Parallel
        if 'cuda' in str(device):
            self.model.to(device)
            gpu_count = torch.cuda.device_count()
            print('GPU number: {}'.format(gpu_count))
            if gpu_count > 1:
                self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

    def transcribe(self, audio, midi_path, stop_event=None):
        """Transcribe an audio recording.

        Args:
          audio: (audio_samples,)
          midi_path: str, path to write out the transcribed MIDI.

        Returns:
          transcribed_dict, dict: {'output_dict':, ..., 'est_note_events': ...}

        """
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Transcription stopped by user.")
        audio = audio[None, :]  # (1, audio_samples)

        # Pad audio to be evenly divided by segment_samples
        audio_len = audio.shape[1]
        pad_len = int(np.ceil(audio_len / self.segment_samples))\
            * self.segment_samples - audio_len

        audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)

        # Enframe to segments
        segments = self.enframe(audio, self.segment_samples)
        """(N, segment_samples)"""
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Transcription stopped by user.")

        # Forward
        output_dict = forward(self.model, segments, batch_size=self.batch_size, stop_event=stop_event)
        """{'reg_onset_output': (N, segment_frames, classes_num), ...}"""
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Transcription stopped by user.")

        # Deframe to original length
        for key in output_dict.keys():
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Transcription stopped by user.")
            output_dict[key] = self.deframe(output_dict[key])[0 : audio_len]
        """output_dict: {
          'reg_onset_output': (N, segment_frames, classes_num), 
          'reg_offset_output': (N, segment_frames, classes_num), 
          'frame_output': (N, segment_frames, classes_num), 
          'velocity_output': (N, segment_frames, classes_num)}"""

        # Post processor
        post_processor = RegressionPostProcessor(self.frames_per_second, 
            classes_num=self.classes_num, onset_threshold=self.onset_threshold, 
            offset_threshold=self.offset_threshod, 
            frame_threshold=self.frame_threshold, 
            pedal_offset_threshold=self.pedal_offset_threshold)

        # Post process output_dict to MIDI events
        (est_note_events, est_pedal_events) = \
            post_processor.output_dict_to_midi_events(output_dict)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Transcription stopped by user.")

        # Write MIDI events to file
        if midi_path:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Transcription stopped by user.")
            write_events_to_midi(start_time=0, note_events=est_note_events, 
                pedal_events=est_pedal_events, midi_path=midi_path)
            print('Write out to {}'.format(midi_path))

        transcribed_dict = {
            'output_dict': output_dict, 
            'est_note_events': est_note_events,
            'est_pedal_events': est_pedal_events}

        return transcribed_dict

    def enframe(self, x, segment_samples):
        """Enframe long sequence to short segments.

        Args:
          x: (1, audio_samples)
          segment_samples: int

        Returns:
          batch: (N, segment_samples)
        """
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples])
            pointer += segment_samples // 2

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        """Deframe predicted segments to original sequence.

        Args:
          x: (N, segment_frames, classes_num)

        Returns:
          y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]

        else:
            x = x[:, 0 : -1, :]
            """Remove an extra frame in the end of each segment caused by the
            'center=True' argument when calculating spectrogram."""
            (N, segment_samples, classes_num) = x.shape
            assert segment_samples % 4 == 0

            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y
