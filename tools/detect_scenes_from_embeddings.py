import click
import numpy as np
import multiprocessing as mp
from tqdm.auto import tqdm
from pathlib import Path
from functools import partial
from scipy.spatial.distance import cosine as cosine_distance
from utils import load_embeddings_from_npy_file


def consecutive_cosine_distance(
    embeddings,
):
    distances = np.array([
        cosine_distance(embeddings[i-1], embeddings[i]) 
        for i in range(1, len(embeddings))
    ])
    
    return distances


def find_scenes(
    distances, 
    threshold=0.1
):
    scene_boundaries = [0]
    n_frames = len(distances)
        
    for i in range(1, len(distances)):
        if distances[i] > threshold:
            scene_boundaries.append(i + 1)
            
    if n_frames + 1 not in scene_boundaries:
        scene_boundaries.append(n_frames + 1)

    scene_list = []
    for i in range(len(scene_boundaries) - 1):
        start = scene_boundaries[i]
        end = scene_boundaries[i + 1]
        scene_list.append((start, end))
        
    return scene_list


def calculate_adaptive_threshold(distances):
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    return threshold


def detect_scenes_from_embeddings_file(
    video_embeddings_file_path,
    scenes_save_dir,
    min_threshold=0.06
):
    file_id = video_embeddings_file_path.stem

    scenes_save_path = scenes_save_dir / f'{file_id}.npy'

    if scenes_save_path.is_file():
        return 
    
    try:
        embeddings, file_id, n_frames = load_embeddings_from_npy_file(
            video_embeddings_file_path
        )
        
        distances_cosine = consecutive_cosine_distance(
            embeddings,
        )
        
        threshold = calculate_adaptive_threshold(
            distances=distances_cosine
        )

        threshold = max(threshold, min_threshold)
        
        scene_list = find_scenes(
            distances_cosine, 
            threshold=threshold
        )
        
        scene_list = np.array(scene_list)
        np.save(scenes_save_path, scene_list)
    except:
        print(f'Something wrong with {video_embeddings_file_path}')


@click.command()
@click.option(
    "--dataset_dir", 
    type=str, 
    required=True, 
)
@click.option(
    "--embeddings_save_dir",
    default='./embeddings',
    type=str,
    required=True,
)
@click.option(
    "--scenes_save_dir",
    default='./scenes',
    type=str,
    required=True,
)
@click.option(
    "--num_workers",
    default=16,
    type=int,
    required=True,
)
@click.option(
    "--min_threshold",
    default=0.06,
    type=float,
    required=True,
)
def detect_scenes_from_embeddings(
    dataset_dir,
    embeddings_save_dir,
    scenes_save_dir,
    num_workers,
    min_threshold
):
    dataset_dir = Path(dataset_dir)

    embeddings_save_dir = Path(embeddings_save_dir)
    embeddings_save_dir.mkdir(exist_ok=True)

    scenes_save_dir = Path(scenes_save_dir)
    scenes_save_dir.mkdir(exist_ok=True)

    video_fvs_file_paths = list(dataset_dir.glob('*.npy'))

    process_func = partial(
        detect_scenes_from_embeddings_file, 
        embeddings_save_dir=embeddings_save_dir,
        scenes_save_dir=scenes_save_dir,
        min_threshold=min_threshold
    )

    with mp.Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_func, video_fvs_file_paths),
            total=len(video_fvs_file_paths)
        ):
            pass 

    print("All npy files processed.")


if __name__ == '__main__':
    detect_scenes_from_embeddings()
    