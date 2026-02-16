import os
import numpy as np
import audioread
import librosa
from .piano_vad import note_detection_with_onset_offset_regress, pedal_detection_with_onset_offset_regress
from . import config

def write_events_to_midi(start_time, note_events, pedal_events, midi_path):
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1000000.0 // beats_per_second)
    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)
    track1 = MidiTrack()
    message_roll = []
    for note_event in note_events:
        message_roll.append({'time': note_event['onset_time'], 'midi_note': note_event['midi_note'], 'velocity': note_event['velocity']})
        message_roll.append({'time': note_event['offset_time'], 'midi_note': note_event['midi_note'], 'velocity': 0})
    if pedal_events:
        for pedal_event in pedal_events:
            message_roll.append({'time': pedal_event['onset_time'], 'control_change': 64, 'value': 127})
            message_roll.append({'time': pedal_event['offset_time'], 'control_change': 64, 'value': 0})
    message_roll.sort(key=lambda note_event: note_event['time'])
    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            if 'midi_note' in message.keys():
                track1.append(Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))
            elif 'control_change' in message.keys():
                track1.append(Message('control_change', channel=0, control=message['control_change'], value=message['value'], time=diff_ticks))
    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)
    midi_file.save(midi_path)

class RegressionPostProcessor(object):

    def __init__(self, frames_per_second, classes_num, onset_threshold, offset_threshold, frame_threshold, pedal_offset_threshold, max_note_frames=600):
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.max_note_frames = max_note_frames
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale

    @staticmethod
    def _threshold_for_class(threshold, class_index, default):
        if isinstance(threshold, (int, float, np.floating)):
            return float(threshold)
        try:
            arr = np.asarray(threshold, dtype=np.float32).reshape(-1)
        except Exception:
            return float(default)
        if arr.size <= 0:
            return float(default)
        idx = int(class_index)
        if idx < 0:
            idx = 0
        if idx >= arr.size:
            idx = arr.size - 1
        value = arr[idx]
        if not np.isfinite(value):
            return float(default)
        return float(value)

    def output_dict_to_midi_events(self, output_dict):
        est_on_off_note_vels, est_pedal_on_offs = self.output_dict_to_note_pedal_arrays(output_dict)
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)
        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)
        return (est_note_events, est_pedal_events)

    def output_dict_to_note_pedal_arrays(self, output_dict):
        onset_output, onset_shift_output = self.get_binarized_output_from_regression(reg_output=output_dict['reg_onset_output'], threshold=self.onset_threshold, neighbour=2)
        output_dict['onset_output'] = onset_output
        output_dict['onset_shift_output'] = onset_shift_output
        offset_output, offset_shift_output = self.get_binarized_output_from_regression(reg_output=output_dict['reg_offset_output'], threshold=self.offset_threshold, neighbour=4)
        output_dict['offset_output'] = offset_output
        output_dict['offset_shift_output'] = offset_shift_output
        if 'reg_pedal_offset_output' in output_dict.keys():
            pedal_offset_output, pedal_offset_shift_output = self.get_binarized_output_from_regression(reg_output=output_dict['reg_pedal_offset_output'], threshold=self.pedal_offset_threshold, neighbour=4)
            output_dict['pedal_offset_output'] = pedal_offset_output
            output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)
        if 'reg_pedal_onset_output' in output_dict.keys():
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)
        else:
            est_pedal_on_offs = None
        return (est_on_off_note_vels, est_pedal_on_offs)

    def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        frames_num, classes_num = reg_output.shape
        for k in range(classes_num):
            threshold_k = self._threshold_for_class(threshold, k, 0.3)
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold_k and self.is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift
        return (binary_output, shift_output)

    def is_monotonic_neighbour(self, x, n, neighbour):
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False
        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]
        for piano_note in range(classes_num):
            frame_threshold = self._threshold_for_class(self.frame_threshold, piano_note, 0.1)
            est_tuples_per_note = note_detection_with_onset_offset_regress(frame_output=output_dict['frame_output'][:, piano_note], onset_output=output_dict['onset_output'][:, piano_note], onset_shift_output=output_dict['onset_shift_output'][:, piano_note], offset_output=output_dict['offset_output'][:, piano_note], offset_shift_output=output_dict['offset_shift_output'][:, piano_note], velocity_output=output_dict['velocity_output'][:, piano_note], frame_threshold=frame_threshold, max_note_frames=self.max_note_frames)
            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)
        est_tuples = np.array(est_tuples)
        est_midi_notes = np.array(est_midi_notes)
        if len(est_tuples) == 0:
            return np.array([])
        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            velocities = est_tuples[:, 4]
            est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
            est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)
            return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        frames_num = output_dict['pedal_frame_output'].shape[0]
        est_tuples = pedal_detection_with_onset_offset_regress(frame_output=output_dict['pedal_frame_output'][:, 0], offset_output=output_dict['pedal_offset_output'][:, 0], offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0], frame_threshold=0.5)
        est_tuples = np.array(est_tuples)
        if len(est_tuples) == 0:
            return np.array([])
        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            velocity = int(round(float(est_on_off_note_vels[i][3]) * float(self.velocity_scale)))
            velocity = max(0, min(127, velocity))
            midi_events.append({'onset_time': est_on_off_note_vels[i][0], 'offset_time': est_on_off_note_vels[i][1], 'midi_note': int(est_on_off_note_vels[i][2]), 'velocity': velocity})
        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({'onset_time': pedal_on_offs[i, 0], 'offset_time': pedal_on_offs[i, 1]})
        return pedal_events

def load_audio(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type='kaiser_best', backends=None):
    y = []
    resolved_backends = backends
    if resolved_backends is None:
        ffdec_module = getattr(audioread, 'ffdec', None)
        ffmpeg_audio_file = getattr(ffdec_module, 'FFmpegAudioFile', None) if ffdec_module is not None else None
        if ffmpeg_audio_file is not None:
            resolved_backends = [ffmpeg_audio_file]
    buf_to_float = getattr(librosa.util, 'buf_to_float', None)
    if not callable(buf_to_float):
        buf_to_float = librosa.core.audio.util.buf_to_float
    to_mono = getattr(librosa, 'to_mono', None)
    if not callable(to_mono):
        to_mono = librosa.core.audio.to_mono
    resample = getattr(librosa, 'resample', None)
    if not callable(resample):
        resample = librosa.core.audio.resample
    open_kwargs = {}
    if resolved_backends is not None:
        open_kwargs['backends'] = resolved_backends
    with audioread.audio_open(os.path.realpath(path), **open_kwargs) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        s_start = int(np.round(sr_native * offset)) * n_channels
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + int(np.round(sr_native * duration)) * n_channels
        n = 0
        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)
            if n < s_start:
                continue
            if s_end < n_prev:
                break
            if s_end < n:
                frame = frame[:s_end - n_prev]
            if n_prev <= s_start <= n:
                frame = frame[s_start - n_prev:]
            y.append(frame)
    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = to_mono(y)
        if sr is not None:
            y = resample(y, orig_sr=sr_native, target_sr=sr, res_type=res_type)
        else:
            sr = sr_native
    y = np.ascontiguousarray(y, dtype=dtype)
    return (y, sr)
