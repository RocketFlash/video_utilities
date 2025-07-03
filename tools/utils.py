import numpy as np


def load_embeddings_from_npy_file(
    file_path,
    normilize: bool = True
):
    file_id = int(file_path.stem)
    embeddings = np.load(
        file_path, 
        mmap_mode='r'
    )

    if normilize:
        norms = np.linalg.norm(
            embeddings, 
            axis=1, 
            keepdims=True
        )
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    n_frames = embeddings.shape[0]
    return embeddings, file_id, n_frames