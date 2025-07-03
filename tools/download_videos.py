import click
import time
import pandas as pd
import multiprocessing as mp
from tqdm.auto import tqdm
from pathlib import Path
from video_utilities import (
    VideoDownloader
)
from pathlib import Path


def worker_function(
    queue, 
    url_template, 
    secret,
    quality,
    save_dir, 
    counter, 
    lock
):
    video_downloader = VideoDownloader(
        url_template=url_template,
        secret=secret,
        quality=quality,
        save_dir=save_dir,
        if_quality_not_exist_strategy='higher'
    )

    while True:
        try:
            video_id = queue.get(timeout=1)  
            if video_id is None:  
                break
            try:
                video_id_str = str(video_id)
                video_downloader(video_id_str, verbose=False)
            except:
                print(f'Something wrong with video: {video_id}')

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
    "--save_dir", 
    type=str, 
    default='./'
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
    "--n_workers", 
    type=int, 
    default=32
)
@click.option(
    "--n_files_max", 
    type=int, 
    default=-1
)
def download_videos(
    dataset_file_path,
    save_dir,
    url_template,
    secret,
    quality,
    n_workers,
    n_files_max
):
    mp.set_start_method('spawn')

    dataset_file_path = Path(dataset_file_path)
    
    if dataset_file_path.suffix == '.parquet':
        dataset_df = pd.read_parquet(
            dataset_file_path, 
            engine='fastparquet'
        )
        video_ids = dataset_df.video_id.unique()
    elif dataset_file_path.suffix == '.txt':
        with open(dataset_file_path, 'r') as file:
            video_ids = [line.strip() for line in file]
        video_ids = list(set(video_ids))
    else:
        dataset_df = pd.read_csv(
            dataset_file_path, 
        )
        video_ids = dataset_df.video_id.unique()

    if n_files_max>0:
        n_files_max = min(n_files_max, len(video_ids))
        video_ids = video_ids[:n_files_max]

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

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
                url_template, 
                secret,
                quality,
                save_dir, 
                counter, 
                lock
            )
        )
        p.start()
        workers.append(p)

    for video_id in tqdm(video_ids, desc="Queueing videos"):
        task_queue.put(video_id)
    task_queue.put(None)

    total_tasks = len(video_ids)

    pbar = tqdm(total=total_tasks, desc="Downloading videos")
    while counter.value < total_tasks:
        with lock:
            pbar.n = counter.value  
        pbar.refresh()
        time.sleep(1)

    pbar.n = total_tasks  
    pbar.close()

    for p in workers:
        p.join()

    print("All videos downloaded.")


if __name__ == '__main__':
    download_videos()
    