from __future__ import annotations
from pathlib import Path
import numpy as np
from . import config
from .model_paths import MODEL_DOWNLOAD_URL, MODEL_MIN_BYTES, resolve_model_path
from .utilities import write_events_to_midi
from a2m.core import conversion_service

class PianoTranscription:

    def __init__(self, model_type: str='onnx', checkpoint_path: str | Path | None=None, segment_samples: int=16000 * 10, device: str | None='cpu', batch_size: int=1):
        normalized_type = str(model_type or 'onnx').strip().lower()
        if normalized_type not in {'onnx', 'note_pedal'}:
            raise ValueError("model_type must be 'onnx' or 'note_pedal'.")
        model_path = Path(resolve_model_path(checkpoint_path))
        if model_path.suffix.lower() != '.onnx':
            raise RuntimeError('A2M v2 accepts only ONNX models (.onnx).')
        if not model_path.exists() or model_path.stat().st_size < MODEL_MIN_BYTES:
            raise RuntimeError(f'ONNX model not found or incomplete. Download it via the A2M app ({MODEL_DOWNLOAD_URL}) and retry.')
        try:
            self.batch_size = max(1, int(batch_size))
        except Exception:
            self.batch_size = 1
        self.segment_samples = max(1, int(segment_samples))
        self.device = str(device or 'cpu').strip().lower()
        self.model_path = model_path
        self.frames_per_second = config.frames_per_second
        self.classes_num = config.classes_num
        self.onset_threshold = 0.3
        self.offset_threshold = 0.3
        self.frame_threshold = 0.1
        self.pedal_offset_threshold = 0.2
        self.sample_rate = config.sample_rate
        self.active_provider = 'CPU'

    def transcribe(self, audio: np.ndarray, midi_path: str | Path | None, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError('Transcription stopped by user.')
        audio_arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        est_note_events, est_pedal_events = conversion_service.transcribe_audio_array(audio_arr, stop_event=stop_event, model_path_override=self.model_path, device_override=self.device, batch_size_override=self.batch_size)
        self.active_provider = conversion_service.get_active_provider_label()
        if midi_path:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError('Transcription stopped by user.')
            write_events_to_midi(start_time=0, note_events=est_note_events, pedal_events=est_pedal_events, midi_path=str(midi_path))
        return {'output_dict': {}, 'est_note_events': est_note_events, 'est_pedal_events': est_pedal_events}
