def note_detection_with_onset_offset_regress(frame_output, onset_output, onset_shift_output, offset_output, offset_shift_output, velocity_output, frame_threshold, max_note_frames=600):
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None
    for i in range(onset_output.shape[0]):
        if onset_output[i] == 1:
            if bgn is not None:
                fin = max(i - 1, 0)
                output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, velocity_output[bgn]])
                frame_disappear, offset_occur = (None, None)
            bgn = i
        if bgn is not None and i > bgn:
            if frame_output[i] <= frame_threshold and (not frame_disappear):
                frame_disappear = i
            if offset_output[i] == 1 and (not offset_occur):
                offset_occur = i
            if frame_disappear:
                if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                    fin = offset_occur
                else:
                    fin = frame_disappear
                output_tuples.append([bgn, fin, onset_shift_output[bgn], offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = (None, None, None)
            max_len_reached = False
            if bgn is not None and max_note_frames is not None:
                try:
                    max_frames_limit = int(max_note_frames)
                except Exception:
                    max_frames_limit = 0
                if max_frames_limit > 0:
                    max_len_reached = i - bgn >= max_frames_limit
            if bgn is not None and (max_len_reached or i == onset_output.shape[0] - 1):
                fin = i
                output_tuples.append([bgn, fin, onset_shift_output[bgn], offset_shift_output[fin], velocity_output[bgn]])
                bgn, frame_disappear, offset_occur = (None, None, None)
    output_tuples.sort(key=lambda pair: pair[0])
    return output_tuples

def pedal_detection_with_onset_offset_regress(frame_output, offset_output, offset_shift_output, frame_threshold):
    output_tuples = []
    bgn = None
    frame_disappear = None
    offset_occur = None
    for i in range(1, frame_output.shape[0]):
        if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
            if bgn is None:
                bgn = i
        if bgn is not None and i > bgn:
            if frame_output[i] <= frame_threshold and (not frame_disappear):
                frame_disappear = i
            if offset_output[i] == 1 and (not offset_occur):
                offset_occur = i
            if offset_occur:
                fin = offset_occur
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = (None, None, None)
            if frame_disappear and i - frame_disappear >= 10:
                fin = frame_disappear
                output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                bgn, frame_disappear, offset_occur = (None, None, None)
    output_tuples.sort(key=lambda pair: pair[0])
    return output_tuples
