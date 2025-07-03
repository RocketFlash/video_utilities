import os
import cv2
import json
import click
import torch
import time
import numpy as np
from PIL import Image
import multiprocessing as mp
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from video_utilities import (
    VideoFrameSplitter,
    VideoFrameSplitterConfig,
)
import onnxruntime
import nudenet
from nudenet import NudeDetector


class NudeDetectorGPU(NudeDetector):
    def __init__(
        self, 
        model_path=None, 
        providers=None, 
        inference_resolution=320,
        device_id=0
    ):
        print('Model path: ', model_path)
        print('Inference resolution: ', inference_resolution)
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(nudenet.__path__[0], "320n.onnx")
            if model_path is None
            else model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            provider_options=[{'device_id': device_id}, {}]
        )
        model_inputs = self.onnx_session.get_inputs()

        self.input_width = inference_resolution
        self.input_height = inference_resolution
        self.input_name = model_inputs[0].name


class NSFWDetector:
    def __init__(
        self,
        model_path=None,
        device_id=0
    ):
        self.device_id = device_id
        self.model_path = model_path
        model, processor = self.get_model_and_processor(
            model_path=self.model_path
        )
        self.model = model
        self.processor = processor

    
    def get_model_and_processor(
        self,
        model_path,
    ):
        model = None
        processor = None

        if model_path is None:
            inference_resolution = 320
        else:
            inference_resolution = 640

        model = NudeDetectorGPU(
            model_path=model_path, 
            inference_resolution=inference_resolution,
            device_id=self.device_id
        )
            
        return model, processor


    def __call__(
        self,
        images,
        batch_size=32
    ):
        predictions = self.model.detect_batch(
            images, 
            batch_size=batch_size
        )
        return predictions


def worker_function(
    queue, 
    gpu_id, 
    frame_max_size,
    frame_interval_sec,
    n_sec_max,
    predictions_save_dir, 
    model_path,
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

    nsfw_detector = NSFWDetector(
        model_path=model_path,
        device_id=gpu_id
    )

    while True:
        try:
            video_path = queue.get(timeout=1)  
            if video_path is None:  
                break
            video_name = video_path.stem
            predictions_save_path = predictions_save_dir / f'{video_name}.json'

            if predictions_save_path.is_file():
                with lock:
                    counter.value += 1
                continue

            video_frames_data = video_frame_splitter(video_path, verbose=False)
            if video_frames_data is None:
                continue

            frame_images = [
                cv2.cvtColor(frame_data.image, cv2.COLOR_RGBA2BGR)
                for frame_data in video_frames_data.frames
            ]

            predictions = nsfw_detector(
                frame_images, 
                batch_size=32
            )

            with open(predictions_save_path, "w") as f:
                json.dump(predictions, f, indent=4)
                
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
    "--predictions_save_dir",
    default='./embeddings',
    type=str,
    required=True,
)
@click.option(
    "--model_path",
    default=None,
    type=str,
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
    "--n_gpus", 
    type=int, 
    default=4
)
@click.option(
    "--n_workers_per_gpu", 
    type=int, 
    default=1
)
def generate_nsfw_detector_predictions(
    dataset_dir,
    predictions_save_dir,
    model_path,
    frame_max_size,
    frame_interval_sec,
    n_sec_max,
    n_gpus,
    n_workers_per_gpu
):
    mp.set_start_method('spawn')
    
    dataset_dir = Path(dataset_dir)
    video_paths = list(dataset_dir.glob('*.mp4'))
    
    predictions_save_dir = Path(predictions_save_dir)
    predictions_save_dir.mkdir(exist_ok=True)

    n_total_workers = n_gpus * n_workers_per_gpu
    task_queue = mp.Queue()
    counter = mp.Value('i', 0) 
    lock = mp.Lock()

    print('N gpus: ', n_gpus)
    print('N workers per gpu : ', n_workers_per_gpu)
    print('Total N workers   : ', n_total_workers)

    workers = []
    for gpu_id in range(n_gpus):
        for _ in range(n_workers_per_gpu):
            p = mp.Process(
                target=worker_function,
                args=(
                    task_queue, 
                    gpu_id, 
                    frame_max_size,
                    frame_interval_sec,
                    n_sec_max,
                    predictions_save_dir,
                    model_path,
                    counter, 
                    lock
                )
            )
            p.start()
            workers.append(p)

    for video_path in tqdm(video_paths, desc="Queueing videos"):
        task_queue.put(video_path)

    for _ in range(n_total_workers):
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
    generate_nsfw_detector_predictions()