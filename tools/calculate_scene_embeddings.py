import click
import numpy as np
import multiprocessing as mp
from tqdm.auto import tqdm
from pathlib import Path
from functools import partial
from utils import load_embeddings_from_npy_file


def calculate_scene_embeddings_from_scenes_and_embeddings(
    video_frame_embeddings_file_path,
    embeddings_dir,
    scenes_dir,
    save_dir
):
    file_id = video_frame_embeddings_file_path.stem
    embeddings_file_path = embeddings_dir / f'{file_id}.npy'
    scenes_file_path = scenes_dir / f'{file_id}.npy'
    scene_embeddings_save_path = save_dir / f'{file_id}.npy'

    if not embeddings_file_path.is_file():
        print('No embeddings file for ', file_id)
        return 
    
    if not scenes_file_path.is_file():
        print('No scenes file for ', file_id)
        return 
    
    scene_list = np.load(
        scenes_file_path, 
        mmap_mode='r'
    )
    
    try:
        embeddings, file_id, n_frames = load_embeddings_from_npy_file(
            video_frame_embeddings_file_path
        )
        
        video_scene_embeddings = []
        for scene_idx, scene_info in enumerate(scene_list):
            scene_start, scene_end = scene_info
            frame_embeddings = embeddings[scene_start:scene_end, :]
            scene_embeddings = np.mean(frame_embeddings, axis=0)
            video_scene_embeddings.append(scene_embeddings)
        
        video_scene_embeddings = np.array(video_scene_embeddings)
        
        np.save(scene_embeddings_save_path, video_scene_embeddings)
    except:
        print(f'Something wrong with {embeddings_file_path}')


@click.command()
@click.option(
    "--embeddings_dir", 
    type=str, 
    required=True, 
)
@click.option(
    "--scenes_dir", 
    type=str, 
    required=True, 
)
@click.option(
    "--save_dir",
    default='./embeddings',
    type=str,
    required=True,
)
@click.option(
    "--num_workers",
    default=16,
    type=int,
    required=True,
)
def calculate_scene_embeddings(
    embeddings_dir,
    scenes_dir,
    save_dir,
    num_workers
):
    embeddings_dir = Path(embeddings_dir)
    scenes_dir = Path(scenes_dir)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    video_frame_embeddings_file_paths = list(embeddings_dir.glob('*.npy'))

    process_func = partial(
        calculate_scene_embeddings_from_scenes_and_embeddings, 
        embeddings_dir=embeddings_dir,
        scenes_dir=scenes_dir,
        save_dir=save_dir
    )

    with mp.Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_func, video_frame_embeddings_file_paths),
            total=len(video_frame_embeddings_file_paths)
        ):
            pass 

    print("All npy files processed.")


if __name__ == '__main__':
    calculate_scene_embeddings()
    