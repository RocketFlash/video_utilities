import click
import torch
import time
import json
import numpy as np
from PIL import Image
import pandas as pd
import multiprocessing as mp
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    Union,
    Optional,
    List,
    Dict
)
from video_utilities import (
    VideoDownloader,
    VideoSceneDetector,
    VideoSceneDetectorConfig,
)


class VideoScenesExtractor:
    def __init__(
        self,
        video_downloader=None,
        scene_detector=None,
        save_dir='./',
        overwrite_if_exist=False
    ):
        self.video_downloader = video_downloader
        self.scene_detector = scene_detector
        self.save_dir = Path(save_dir)
        self.overwrite_if_exist = overwrite_if_exist
        

    def download_video_and_detect_scenes(
        self, 
        video_id
    ):
        video_id_str = str(video_id)
        status_dict = self.video_downloader(video_id_str, verbose=False)
        is_success = status_dict[video_id_str]['success']
        video_path = status_dict[video_id_str]['video_path']

        scenes_list = None
        if is_success:
            scenes_list = self.scene_detector(
                video_path,
            )
            Path(video_path).unlink(missing_ok=True)
            
        return scenes_list


    def save_scene_data_to_json(
        self,
        scenes_list, 
        save_path='./scene_data.json'
    ):
        scene_data_dicts = [asdict(scene) for scene in scenes_list]
        
        with open(save_path, 'w') as json_file:
            json.dump(
                scene_data_dicts, 
                json_file, 
                indent=4
            )


    def __call__(self, video_id):
        save_path = self.save_dir / f'{video_id}.json'
        if not save_path.is_file() or self.overwrite_if_exist:
            try:
                scenes_list = self.download_video_and_detect_scenes(video_id)
                if scenes_list is not None:
                    self.save_scene_data_to_json(scenes_list, save_path=save_path)
            except:
                print(f'Something wrong with video: {video_id}')


def worker_function(
    queue, 
    url_template,
    secret,
    quality,
    video_tmp_save_dir,
    scene_save_dir, 
    counter, 
    lock,
    overwrite_if_exist
):
    video_downloader = VideoDownloader(
        url_template=url_template,
        secret=secret,
        quality=quality,
        save_dir=video_tmp_save_dir,
        if_quality_not_exist_strategy='higher'
    )

    scene_detector_config = VideoSceneDetectorConfig(
        show_progress=False,
        backend='opencv'
    )
    scene_detector = VideoSceneDetector(
        config=scene_detector_config
    )
    
    extractor = VideoScenesExtractor(
        video_downloader=video_downloader,
        scene_detector=scene_detector,
        save_dir=scene_save_dir,
        overwrite_if_exist=overwrite_if_exist
    )
    
    while True:
        try:
            video_id = queue.get(timeout=1)  
            if video_id is None:  
                break
            extractor(video_id)
            with lock:
                counter.value += 1
        except mp.queues.Empty:
            continue


@click.command()
@click.option(
    "--npy_dataset_dir", 
    type=str, 
    required=True, 
)
@click.option(
    "--video_tmp_save_dir", 
    type=str, 
    default='./'
)
@click.option(
    "--scene_save_dir",
    default='./scenes',
    type=str,
    required=True,
)
@click.option(
    "--url_template", 
    type=str, 
    required=True,
)
@click.option(
    "--secret", 
    type=str, 
    required=True,
)
@click.option(
    "--quality", 
    type=str, 
    default='max'
)
def generate_scenes_data(
    npy_dataset_dir,
    video_tmp_save_dir,
    scene_save_dir,
    url_template,
    secret,
    quality
):
    print('Prepare video ids...')
    npy_dataset_dir = Path(npy_dataset_dir)
    npy_file_paths = list(npy_dataset_dir.glob('*.npy'))
    video_ids = [npy_file.stem for npy_file in npy_file_paths]
    
    video_tmp_save_dir = Path(video_tmp_save_dir)
    video_tmp_save_dir.mkdir(exist_ok=True)

    scene_save_dir = Path(scene_save_dir)
    scene_save_dir.mkdir(exist_ok=True)

    num_workers = 16
    task_queue = mp.Queue()
    counter = mp.Value('i', 0) 
    lock = mp.Lock()

    workers = []
    overwrite_if_exist = False

    print(f'N workers: {num_workers}')
    print('Load workers...')
    for _ in range(num_workers):
        p = mp.Process(
            target=worker_function,
            args=(
                task_queue, 
                url_template,
                secret,
                quality,
                video_tmp_save_dir,
                scene_save_dir,
                counter, 
                lock,
                overwrite_if_exist
            )
        )
        p.start()
        workers.append(p)

    for video_id in tqdm(
        video_ids, 
        desc="Queueing videos"
    ):
        task_queue.put(video_id)

    total_tasks = len(video_ids)

    pbar = tqdm(
        total=total_tasks, 
        desc="Processing tasks"
    )
    while counter.value < total_tasks:
        with lock:
            pbar.n = counter.value  
        pbar.refresh()
        time.sleep(1)

    for _ in workers:
        task_queue.put(None)

    pbar.n = total_tasks  
    pbar.close()

    for p in workers:
        p.join()

    print("All videos processed.")


if __name__ == '__main__':
    generate_scenes_data()
    