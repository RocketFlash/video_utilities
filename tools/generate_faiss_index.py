import time
import faiss
import numpy as np
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp


def load_npy_file(
    file_path
):
    file_id = int(file_path.stem)
    embeddings = np.load(
        file_path, 
        mmap_mode='r'
    )
    norms = np.linalg.norm(
        embeddings, 
        axis=1, 
        keepdims=True
    )
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    n_frames = embeddings.shape[0]
    
    return embeddings, file_id, n_frames


def load_npy_files_parallel(
    file_paths,
    n_workers=32,
    verbose=False
):
    with mp.Pool(n_workers) as pool:
        if verbose:
            result_list = list(tqdm(
                pool.imap_unordered(load_npy_file, file_paths),
                total=len(file_paths),
                desc="Loading .npy files"
            ))
        else:
            result_list = pool.imap_unordered(load_npy_file, file_paths)
                
    embeddings_list = []
    file_names_list = []
    video_lens_list = []
    for file_result in result_list:
        embeddings_list.append(file_result[0])
        file_names_list.append(file_result[1])
        video_lens_list.append(file_result[2])

    embeddings = np.vstack(embeddings_list).astype(np.float32)
        
    return embeddings, file_names_list, video_lens_list


def sample_training_files(
    video_fvs_file_paths, 
    n_files, 
):
    video_fvs_file_paths = np.array(video_fvs_file_paths)

    video_data_paths = np.random.choice(
        video_fvs_file_paths, 
        n_files,
    )

    return video_data_paths


if __name__ == '__main__':
    # dataset_dir = Path('/mnt/nfs/ml/raf/video_fvs/')
    # save_dir = dataset_dir / 'faiss_data'

    dataset_dir = Path('/mnt/nfs/ml/raf/video_embeddings_emb_scenes_v2')
    save_dir = dataset_dir / 'faiss_data'
    save_dir.mkdir(exist_ok=True)

    random_seed = 28
    n_gpus = 4
    n_workers = 32
    embedding_size = 1152
    n_centroids = 10_000
    n_subquantizers = 8
    nbits = 8
    train_n_files = 10_000
    retrain_if_exist = False
    batch_size = 5000

    np.random.seed(random_seed)
    video_fvs_file_paths = list(dataset_dir.glob('*.npy'))
    n_videos = len(video_fvs_file_paths)

    trained_index_path = save_dir / 'index_trained.faiss'

    if (not trained_index_path.is_file()) or retrain_if_exist:
        coarse_quantizer = faiss.IndexFlatIP(embedding_size)
        index_cpu = faiss.IndexIVFPQ(
            coarse_quantizer, 
            embedding_size, 
            n_centroids, 
            n_subquantizers, 
            nbits
        )
        index_cpu.metric_type = faiss.METRIC_INNER_PRODUCT

        train_file_paths = sample_training_files(
            video_fvs_file_paths, 
            n_files=train_n_files, 
        )

        train_embeddings, _, _ = load_npy_files_parallel(
            file_paths=train_file_paths,
            n_workers=n_workers,
            verbose=True
        )
        print('Training data shape: ', train_embeddings.shape)

        print('Training index...')
        start_time = time.time()
        index_cpu.train(train_embeddings)
        end_time = time.time()
        print(f'The training lasted {end_time-start_time} seconds')

        faiss.write_index(index_cpu, str(trained_index_path))        
    else:
        print('Loading index from: ', str(trained_index_path))
        index_cpu = faiss.read_index(str(trained_index_path))
        index_cpu.metric_type = faiss.METRIC_INNER_PRODUCT
        assert index_cpu.is_trained, "Index is not trained!"

    print('Index is trained: ', index_cpu.is_trained)

    print('Move index to GPUs...')
    index_gpu = faiss.index_cpu_to_all_gpus(
        index_cpu
    )

    file_names_all = []
    video_lens_all = []

    print('Adding embeddings to the index...')
    for start in tqdm(
        range(0, n_videos, batch_size)
    ):
        end = min(start + batch_size, n_videos)
        batch_file_paths = video_fvs_file_paths[start:end]
        embeddings, file_names, video_lens = load_npy_files_parallel(
            file_paths=batch_file_paths,
            n_workers=n_workers,
            verbose=True
        )
        file_names_all += file_names
        video_lens_all += video_lens
        index_gpu.add(embeddings)

    file_names_all = np.array(file_names_all)
    video_lens_all = np.array(video_lens_all)

    print('file names shape: ', file_names_all.shape)
    print('video lens shape: ', video_lens_all.shape)

    # print(file_names_all[:10])
    # print(video_lens_all[:10])

    print('Saving index and ids...')
    np.save(save_dir / 'file_names.npy', file_names_all)
    np.save(save_dir / 'video_lens.npy', video_lens_all)

    index_cpu = faiss.index_gpu_to_cpu(index_gpu)
    faiss.write_index(index_cpu, str(save_dir / 'index.faiss'))