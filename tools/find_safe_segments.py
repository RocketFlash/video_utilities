import click
import time
import cv2
import json
import numpy as np
from pathlib import Path
import multiprocessing as mp
from tqdm.auto import tqdm
from video_utilities import (
    VideoFrameSplitter,
    VideoFrameSplitterConfig,
)


def find_safe_video_segments(
    nsfw_scores, 
    threshold=0.5, 
    min_segment_length=10
):
    """
    Find the first segment of the video that doesn't contain NSFW content.
    
    Args:
        nsfw_scores: 1D numpy array of NSFW scores for each second
        threshold: NSFW score threshold to consider content as NSFW
        min_segment_length: Minimum length (in seconds) required for a clean segment
    """
    
    clean_indices = np.where(nsfw_scores < threshold)[0]
    
    if len(clean_indices) == 0:
        return None
    
    segments = []
    segment_start = clean_indices[0]
    
    for i in range(1, len(clean_indices)):
        if clean_indices[i] != clean_indices[i-1] + 1:
            segments.append((segment_start, clean_indices[i-1] + 1))
            segment_start = clean_indices[i]
    
    if len(clean_indices) > 0:
        segments.append((segment_start, clean_indices[-1] + 1))
    
    safe_segments = []
    for start, end in segments:
        if end - start >= min_segment_length:
            safe_segments.append([int(start), int(end)])

    return safe_segments
    

def find_safe_segments_combined(
    model_scores_dict, 
    threshold=0.5, 
    min_segment_length=5, 
    combination_method='mean'
):
    """
    Find the first clean segment using scores from multiple models.
    
    Args:
        model_scores_dict: Dictionary where keys are model names and values are 1D numpy arrays
                          of NSFW scores
        threshold: NSFW score threshold to consider content as NSFW
        min_segment_length: Minimum length (in seconds) required for a clean segment
        combination_method: How to combine scores ('max', 'mean', 'median')
    
    Returns:
        tuple: (start_time, end_time) of the first clean segment, or None if no valid segment found
    """
    if not model_scores_dict:
        return None
    
    model_scores_dict_classification = model_scores_dict['classification']
    model_arrays_classification = list(model_scores_dict_classification.values())
    array_lengths = [len(arr) for arr in model_arrays_classification]

    stacked_scores_detection = None
    if 'detection' in model_scores_dict:
        model_scores_dict_detection = model_scores_dict['detection']
        model_arrays_detection = list(model_scores_dict_detection.values())
        array_lengths += [len(model_arrays_detection[0])]
        stacked_scores_detection = np.vstack(model_arrays_detection)[0]

    if len(set(array_lengths)) > 1:
        raise ValueError("All model score arrays must have the same length")
    
    stacked_scores_classification = np.vstack(model_arrays_classification)
    
    if combination_method == 'max':
        combined_scores = np.max(stacked_scores_classification, axis=0)
    elif combination_method == 'mean':
        combined_scores = np.mean(stacked_scores_classification, axis=0)
    elif combination_method == 'median':
        combined_scores = np.median(stacked_scores_classification, axis=0) 
    else:
        raise ValueError("Invalid combination method. Choose 'max', 'mean', or 'median'")
    
    if stacked_scores_detection is not None:
        combined_scores = np.maximum(combined_scores, stacked_scores_detection)

    safe_segments = find_safe_video_segments(
        combined_scores, 
        threshold, 
        min_segment_length
    )

    return safe_segments, combined_scores


def add_text_to_image(
    image, 
    text, 
    position=(10, 30), 
    font=cv2.FONT_HERSHEY_SIMPLEX,      
    font_scale=1.0, 
    color=(255, 255, 255), 
    thickness=2, 
    background=True, 
    bg_color=(0, 0, 0), 
    bg_alpha=0.5, 
    padding=5
):
    """
    Add text to an OpenCV image at the specified position (default: top-left corner).
    
    Args:
        image: NumPy array representing an OpenCV image (BGR format)
        text: Text string to write on the image
        position: Tuple (x, y) for the position of the text's bottom-left corner
        font: OpenCV font type
        font_scale: Size of the font
        color: Tuple (B, G, R) for text color
        thickness: Text thickness
        background: Whether to add a background rectangle behind the text
        bg_color: Tuple (B, G, R) for background color
        bg_alpha: Transparency of the background (0.0 to 1.0)
        padding: Padding around text for the background rectangle
        
    Returns:
        Modified image as a NumPy array
    """

    img_copy = image.copy()
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    if background:
        x, y = position
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + padding
        
        if bg_alpha < 1.0:
            overlay = img_copy.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            cv2.addWeighted(overlay, bg_alpha, img_copy, 1 - bg_alpha, 0, img_copy)
        else:
            cv2.rectangle(img_copy, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    cv2.putText(img_copy, text, position, font, font_scale, color, thickness)
    
    return img_copy


def worker_function(
    queue, 
    scores_dir,
    frame_max_size,
    frame_interval_sec,
    n_sec_max,
    nsfw_threshold,
    min_segment_length,
    nsfw_offset,
    snapshots_save_dir, 
    counter, 
    lock
):
    video_frame_splitter_config = VideoFrameSplitterConfig(
        start_idx=0,
        frame_interval_sec=frame_interval_sec,
        frame_max_size=frame_max_size,
        n_sec_max=n_sec_max
    )
    video_frame_splitter = VideoFrameSplitter(
        config=video_frame_splitter_config
    )

    model_names_classification = [
        # 'adamcodd',
        'falconsai',
        'lovetillion',
        'marqo'
    ]

    model_names_detection = [
        'nudenet640m',
    ]

    combination_method = 'mean'
    
    while True:
        try:
            video_path = queue.get(timeout=1)
            if video_path is None:  
                break

            video_name = video_path.stem  
            
            video_frame_scores_dict = {
                'classification' : {},
            }

            if len(model_names_detection)>0:
                video_frame_scores_dict['detection'] = {}

            for model_name_classification in model_names_classification:
                model_scores_dir = scores_dir / model_name_classification
                model_video_frame_scores_file_path = model_scores_dir / f'{video_name}.npy'
                model_video_frame_scores = np.load(model_video_frame_scores_file_path)
                video_frame_scores_dict['classification'][model_name_classification] = model_video_frame_scores

            for model_name_detection in model_names_detection:
                model_scores_dir = scores_dir / model_name_detection
                model_video_frame_scores_file_path = model_scores_dir / f'{video_name}.npy'
                model_video_frame_scores = np.load(model_video_frame_scores_file_path)
                video_frame_scores_dict['detection'][model_name_detection] = model_video_frame_scores

            sfw_segments, combined_scores = find_safe_segments_combined(
                model_scores_dict=video_frame_scores_dict, 
                threshold=nsfw_threshold, 
                min_segment_length=min_segment_length, 
                combination_method=combination_method
            )

            if sfw_segments:
                sfw_segment = sfw_segments[0]
                if sfw_segment[0] == 0:
                    video_results_save_dir = snapshots_save_dir / video_name
                    video_results_save_dir.mkdir(exist_ok=True)

                    frame_seconds_to_read = min(sfw_segment[1] + nsfw_offset, len(combined_scores))
                    selected_seconds = list(range(frame_seconds_to_read))
                    selected_second_scores = combined_scores[:len(selected_seconds)]
                    video_frames_data = video_frame_splitter(
                        video_path,
                        selected_seconds=selected_seconds,
                        verbose=False
                    )
                    frame_images = [
                        frame_data.image
                        for frame_data in video_frames_data.frames
                    ]

                    safe_segments_dict = {
                        'safe_segments' : sfw_segments
                    }

                    with open(video_results_save_dir / 'metadata.json', "w") as f:
                        json.dump(safe_segments_dict, f, indent=4)

                    frames_vis = []
                    for frame_idx in range(len(frame_images)):
                        frame_image = frame_images[frame_idx] 
                        if frame_idx < sfw_segment[1]:
                            vis_text = f'SFW (nsfw score: {selected_second_scores[frame_idx]:.2f})'
                            vis_bg_color = (0, 255, 0)
                        else:
                            vis_text = f'NSFW (nsfw score: {selected_second_scores[frame_idx]:.2f})'
                            vis_bg_color = (255, 0, 0)
                            
                        frame_vis = add_text_to_image(
                            frame_image, 
                            vis_text, 
                            bg_color=vis_bg_color, 
                            font_scale=0.5
                        )
                        frames_vis.append(frame_vis)
                        vis_image_save_path = video_results_save_dir / f'second_{frame_idx}.jpg'
                        frame_vis = cv2.cvtColor(frame_vis.copy(), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(vis_image_save_path, frame_vis)

            with lock:
                counter.value += 1
        except mp.queues.Empty:
            continue


@click.command()
@click.option(
    "--dataset_dir", 
    type=str, 
    required=True, 
)
@click.option(
    "--scores_dir", 
    type=str, 
    required=True, 
)
@click.option(
    "--snapshots_save_dir",
    default='./snapshots',
    type=str,
    required=True,
)
@click.option(
    "--frame_max_size", 
    type=int, 
    default=512
)
@click.option(
    "--frame_interval_sec", 
    type=int, 
    default=1
)
@click.option(
    "--n_sec_max", 
    type=int, 
    default=None
)
@click.option(
    "--nsfw_threshold", 
    type=float, 
    default=0.7
)
@click.option(
    "--min_segment_length", 
    type=int, 
    default=10
)
@click.option(
    "--nsfw_offset", 
    type=int, 
    default=5
)
@click.option(
    "--n_workers", 
    type=int, 
    default=32
)
def find_safe_segments(
    dataset_dir,
    scores_dir,
    snapshots_save_dir,
    frame_max_size,
    frame_interval_sec,
    n_sec_max,
    nsfw_threshold,
    min_segment_length,
    nsfw_offset,
    n_workers
):
    mp.set_start_method('spawn')
    
    dataset_dir = Path(dataset_dir)
    video_paths = list(dataset_dir.glob('*.mp4'))
    scores_dir = Path(scores_dir)
    
    snapshots_save_dir = Path(snapshots_save_dir)
    snapshots_save_dir.mkdir(exist_ok=True)

    n_total_workers = n_workers
    task_queue = mp.Queue()
    counter = mp.Value('i', 0) 
    lock = mp.Lock()

    print('Total N workers   : ', n_total_workers)

    workers = []

    for _ in range(n_workers):
        p = mp.Process(
            target=worker_function,
            args=(
                task_queue, 
                scores_dir,
                frame_max_size,
                frame_interval_sec,
                n_sec_max,
                nsfw_threshold,
                min_segment_length,
                nsfw_offset,
                snapshots_save_dir,
                counter, 
                lock
            )
        )
        p.start()
        workers.append(p)

    for video_path in tqdm(video_paths, desc="Queueing videos"):
        task_queue.put(video_path)

    for _ in range(n_workers):
        task_queue.put(None)

    total_tasks = len(video_paths)

    pbar = tqdm(
        total=total_tasks, 
        desc="Processing tasks"
    )
    while counter.value < total_tasks:
        with lock:
            pbar.n = counter.value  
        pbar.refresh()
        time.sleep(1)

    pbar.n = total_tasks  
    pbar.close()

    for p in workers:
        p.join()

    print("All videos processed.")


if __name__ == '__main__':
    find_safe_segments()