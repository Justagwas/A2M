import numpy as np
import torch


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]
 

def forward(model, x, batch_size, stop_event=None):
    """Forward data to model in mini-batch. 
    
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int

    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
    """
    
    output_dict = {}
    device = next(model.parameters()).device
    
    pointer = 0
    total_batches = int(np.ceil(len(x) / batch_size))
    batch_index = 0

    while pointer < len(x):
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Transcription stopped by user.")
        batch_index += 1
        print('Segment {} / {}'.format(batch_index, total_batches))

        batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size

        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Transcription stopped by user.")

        for key in batch_output_dict.keys():
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict
