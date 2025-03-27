import click
import torch
import time
import numpy as np
from PIL import Image
import pandas as pd
import multiprocessing as mp
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Union,
    Optional,
    List,
    Dict
)
from video_utilities import (
    VideoFrameSplitter,
    VideoFrameSplitterConfig,
    VideoDownloader
)
from pathlib import Path
from transformers import AutoModel, AutoProcessor


@dataclass
class VideoFeaturesExtractorConfig():
    model_name: str = "google/siglip2-so400m-patch14-384"
    attn_implementation: str = "flash_attention_2"
    dtype: torch.dtype = torch.float16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    

class VideoFeaturesExtractor:
    def __init__(
        self,
        config=None,
        video_downloader=None,
        video_frame_splitter=None,
        save_dir='./',
        overwrite_if_exist=False
    ):
        self.video_downloader = video_downloader
        self.video_frame_splitter = video_frame_splitter
        self.save_dir = Path(save_dir)
        self.overwrite_if_exist = overwrite_if_exist
        
        if config is None:
            config = self.get_default_config()
        self.config = config
        self.set_params_from_config(config)

        model, processor = self.get_model_and_processor(
            model_name=self.model_name
        )
        self.model = model
        self.processor = processor


    def get_default_config(self):
        return VideoFeaturesExtractorConfig()


    def set_params_from_config(
        self, 
        config: VideoFeaturesExtractorConfig
    ):
        for key, value in vars(config).items():
            setattr(self, key, value)


    def get_model_and_processor(
        self,
        model_name: str,
    ):
        model = AutoModel.from_pretrained(
            model_name, 
            device_map=self.device,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor


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
            
            inputs = self.processor(
                images=batch, 
                return_tensors="pt", 
                padding="max_length"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            
            frame_embeddings.append(outputs.cpu().numpy())

        if verbose:
            bar.close()
        
        frame_embeddings = np.concatenate(frame_embeddings, axis=0)
        
        return frame_embeddings


    def download_video_and_extract_frames(
        self, 
        video_id
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
    video_downloader, 
    video_frame_splitter, 
    embeddings_save_dir, 
    counter, 
    lock
):
    torch.cuda.set_device(gpu_id)
    
    config = VideoFeaturesExtractorConfig(device="cuda")
    extractor = VideoFeaturesExtractor(
        config=config,
        video_downloader=video_downloader,
        video_frame_splitter=video_frame_splitter,
        save_dir=embeddings_save_dir,
        overwrite_if_exist=False
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
def generate_video_fvs(
    dataset_file_path,
    video_tmp_save_dir,
    embeddings_save_dir,
    url_template,
    secret,
    quality
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

    video_downloader = VideoDownloader(
        url_template=url_template,
        secret=secret,
        quality=quality,
        save_dir=video_tmp_save_dir,
        if_quality_not_exist_strategy='higher'
    )

    video_frame_splitter_config = VideoFrameSplitterConfig(
        start_idx=0,
        frame_interval_sec=1,
        frame_max_size=512
    )
    video_frame_splitter = VideoFrameSplitter(
        config=video_frame_splitter_config
    )

    num_gpus = 4
    workers_per_gpu = 4
    total_workers = num_gpus * workers_per_gpu
    task_queue = mp.Queue()
    counter = mp.Value('i', 0) 
    lock = mp.Lock()

    workers = []
    for gpu_id in range(num_gpus):
        for _ in range(workers_per_gpu):
            p = mp.Process(
                target=worker_function,
                args=(
                    task_queue, 
                    gpu_id, 
                    video_downloader, 
                    video_frame_splitter, 
                    embeddings_save_dir,
                    counter, 
                    lock
                )
            )
            p.start()
            workers.append(p)

    for video_id in tqdm(video_ids, desc="Queueing videos"):
        task_queue.put(video_id)

    for _ in range(num_gpus):
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
    