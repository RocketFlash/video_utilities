import numpy as np

def get_frame_idxs_from_interval(
    start_idx, 
    end_idx, 
    n_frames=1
):
    frame_indexes_list = list(range(start_idx, end_idx))
    if n_frames >= len(frame_indexes_list):
        return frame_indexes_list
    else:
        if n_frames==1:
            return [int((end_idx + start_idx)/2)]
            
        indices = np.linspace(0, len(frame_indexes_list) - 1, n_frames, dtype=int)
        return [frame_indexes_list[i] for i in indices]