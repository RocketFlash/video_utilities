import click
import torch
import time
import numpy as np
from PIL import Image
import pandas as pd
import multiprocessing as mp
from tqdm.auto import tqdm
from pathlib import Path
from video_utilities import (
    FeatureExtractor,
    FeatureExtractorConfig,
    VideoFrameSplitter,
    VideoFrameSplitterConfig,
    VideoDownloader
)
from pathlib import Path


class EmbeddingsGenerator:
    def __init__(
        self,
        feature_extractor,
        video_downloader, 
        video_frame_splitter,
        save_dir,
        batch_size: int = 64,
        overwrite_if_exist: bool = False,
    ):
        self.feature_extractor = feature_extractor
        self.video_downloader = video_downloader
        self.video_frame_splitter = video_frame_splitter
        self.batch_size = batch_size
        self.overwrite_if_exist = overwrite_if_exist
        self.save_dir = save_dir


    def process_frames(
        self,
        frames,
        verbose: bool = False
    ):
        frame_embeddings = []
        bar = range(0, len(frames), self.batch_size)

        if verbose:
            bar = tqdm(bar)

        for i in bar:
            batch = frames[i:i+self.batch_size]
            
            outputs = self.feature_extractor.generate_image_embeddings(
                batch
            )
            
            frame_embeddings.append(outputs.cpu().numpy())

        if verbose:
            bar.close()
        
        frame_embeddings = np.concatenate(frame_embeddings, axis=0)
        
        return frame_embeddings


    def download_video_and_extract_frames(
        self,
        video_id,
    ):
        video_id_str = str(video_id)
        status_dict = self.video_downloader(video_id_str, verbose=False)
        is_success = status_dict[video_id_str]['success']
        video_path = status_dict[video_id_str]['video_path']

        video_frames_data = None
        if is_success:
            video_frames_data = self.video_frame_splitter(
                video_path,
                verbose=False
            )
            Path(video_path).unlink(missing_ok=True)
            
        return video_frames_data


    def __call__(self, video_id):
        save_path = self.save_dir / f'{video_id}.npy'
        if not save_path.is_file() or self.overwrite_if_exist:
            try:
                video_frames_data = self.download_video_and_extract_frames(video_id)
                if video_frames_data is not None:
                    frame_images = [frame_data.image for frame_data in video_frames_data.frames]
                    frame_embeddings = self.process_frames(frame_images, verbose=False)
                    np.save(save_path, frame_embeddings)
            except:
                print(f'Something wrong with video: {video_id}')


def worker_function(
    queue, 
    gpu_id, 
    url_template, 
    secret,
    quality,
    frame_max_size,
    frame_interval_sec,
    video_tmp_save_dir, 
    embeddings_save_dir, 
    counter, 
    lock
):
    torch.cuda.set_device(gpu_id)

    video_downloader = VideoDownloader(
        url_template=url_template,
        secret=secret,
        quality=quality,
        save_dir=video_tmp_save_dir,
        if_quality_not_exist_strategy='higher'
    )

    video_frame_splitter_config = VideoFrameSplitterConfig(
        start_idx=0,
        frame_interval_sec=frame_interval_sec,
        frame_max_size=frame_max_size
    )
    video_frame_splitter = VideoFrameSplitter(
        config=video_frame_splitter_config
    )
    
    feature_extractor_config = FeatureExtractorConfig(device="cuda")
    feature_extractor = FeatureExtractor(
        config=feature_extractor_config,
    )

    embeddings_generator = EmbeddingsGenerator(
        feature_extractor=feature_extractor,
        video_downloader=video_downloader, 
        video_frame_splitter=video_frame_splitter,
        save_dir=embeddings_save_dir,
        batch_size=feature_extractor_config.batch_size,
        overwrite_if_exist=False,
    )
    
    while True:
        try:
            video_id = queue.get(timeout=1)  
            if video_id is None:  
                break
            embeddings_generator(video_id)
            with lock:
                counter.value += 1
        except mp.queues.Empty:
            continue


@click.command()
@click.option(
    "--dataset_file_path", 
    type=str, 
    required=True, 
)
@click.option(
    "--video_tmp_save_dir", 
    type=str, 
    default='./'
)
@click.option(
    "--embeddings_save_dir",
    default='./embeddings',
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
    "--n_gpus", 
    type=int, 
    default=1
)
@click.option(
    "--n_workers_per_gpu", 
    type=int, 
    default=1
)
def generate_video_fvs(
    dataset_file_path,
    video_tmp_save_dir,
    embeddings_save_dir,
    url_template,
    secret,
    quality,
    frame_max_size,
    frame_interval_sec,
    n_gpus,
    n_workers_per_gpu
):
    mp.set_start_method('spawn')
    
    dataset_df = pd.read_parquet(
        dataset_file_path, 
        engine='fastparquet'
    )
    video_ids = dataset_df.video_id.unique()

    video_tmp_save_dir = Path(video_tmp_save_dir)
    video_tmp_save_dir.mkdir(exist_ok=True)

    embeddings_save_dir = Path(embeddings_save_dir)
    embeddings_save_dir.mkdir(exist_ok=True)

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
                    url_template, 
                    secret,
                    quality,
                    frame_max_size,
                    frame_interval_sec,
                    video_tmp_save_dir, 
                    embeddings_save_dir,
                    counter, 
                    lock
                )
            )
            p.start()
            workers.append(p)

    for video_id in tqdm(video_ids, desc="Queueing videos"):
        task_queue.put(video_id)

    for _ in range(n_gpus):
        task_queue.put(None)

    total_tasks = len(video_ids)

    pbar = tqdm(total=total_tasks, desc="Processing tasks")
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
    generate_video_fvs()
    