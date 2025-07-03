import click
import gdown
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gdown
import multiprocessing as mp
from functools import partial


def download_file(
    file_id, 
    save_dir="."
):
    save_path = save_dir / f'{file_id}.jpg'
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, output=str(save_path), fuzzy=True)
        return (file_id, True)
    except:
        print(f"Error downloading {file_id}")
        return (file_id, False)


def download_files_parallel(
    image_links, 
    save_dir=".", 
    num_processes=4
):
    """Download files in parallel using multiprocessing with progress bar."""
    # Create a partial function with the output directory
    download_func = partial(download_file, save_dir=save_dir)
    
    # Create a pool
    pool = mp.Pool(processes=num_processes)
    
    # Use imap to get results as they complete and wrap with tqdm
    results = []
    with tqdm(total=len(image_links), desc="Downloading files") as pbar:
        for result in pool.imap_unordered(download_func, image_links):
            results.append(result)
            pbar.update()
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Process results
    successful = [r[0] for r in results if r[1]]
    failed = [(r[0], r[2] if len(r) > 2 else "Unknown error") for r in results if not r[1]]
    
    return {
        "total": len(image_links),
        "successful": len(successful),
        "failed": len(failed),
        "failed_ids": failed
    }

@click.command()
@click.option(
    "--dataset_csv", 
    type=str, 
    required=True, 
)
@click.option(
    "--urls_col_name", 
    type=str, 
    default='Link'
)
@click.option(
    "--n_workers", 
    type=int, 
    default=32
)
@click.option(
    "--save_dir",
    default='./',
    type=str,
    required=True,
)
def download(
    dataset_csv,
    urls_col_name,
    n_workers,
    save_dir,
):
    data_path = Path(dataset_csv)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)

    image_links = df[urls_col_name].tolist()

    image_links = [
        image_link.replace('&export=download', '').split('id=')[1] 
        for image_link in image_links
    ]

    result = download_files_parallel(
        image_links=image_links, 
        save_dir=save_dir, 
        num_processes=n_workers
    )
    
    print(f"Downloaded {result['successful']} of {result['total']} files")
    if result['failed']:
        print(f"Failed to download {len(result['failed_ids'])} files")
        print("Failed IDs:", result['failed_ids'])

if __name__ == '__main__':
    download()