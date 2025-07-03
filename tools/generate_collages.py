import os
import glob
from PIL import Image
import math
import multiprocessing
from functools import partial
import click
from tqdm import tqdm

def process_subfolder(subfolder, save_dir, num_columns):
    subfolder_name = os.path.basename(subfolder)
    
    # Get all second images in the subfolder
    image_paths = sorted(glob.glob(os.path.join(subfolder, "second_*.jpg")), 
                        key=lambda x: int(os.path.basename(x).replace("second_", "").replace(".jpg", "")))
    
    if not image_paths:
        return f"No images found in {subfolder}"
        
    try:
        # Load the first image to get dimensions
        sample_img = Image.open(image_paths[0])
        img_width, img_height = sample_img.size
        
        # Calculate grid dimensions
        num_images = len(image_paths)
        num_rows = math.ceil(num_images / num_columns)
        
        # Create blank canvas for the collage
        collage_width = num_columns * img_width
        collage_height = num_rows * img_height
        collage = Image.new('RGB', (collage_width, collage_height))
        
        # Place images in the grid
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            row = i // num_columns
            col = i % num_columns
            x = col * img_width
            y = row * img_height
            collage.paste(img, (x, y))
        
        # Save the collage
        save_path = os.path.join(save_dir, f"{subfolder_name}.jpg")
        collage.save(save_path)
        return f"Created collage for {subfolder_name}"
    except Exception as e:
        return f"Error processing {subfolder}: {e}"

def create_collage(root_dir, save_dir, num_columns):
    # Ensure save folder exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all subfolders in the root folder
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
    if not subfolders:
        print("No subfolders found in the root directory")
        return
    
    # Set up multiprocessing pool
    num_cpus = multiprocessing.cpu_count()
    process_func = partial(process_subfolder, save_dir=save_dir, num_columns=num_columns)
    
    # Process subfolders in parallel with progress bar
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = list(tqdm(
            pool.imap(process_func, subfolders),
            total=len(subfolders),
            desc="Creating collages",
            unit="folder"
        ))
    
    # Print results
    for result in results:
        if result:
            print(result)

@click.command()
@click.option('--root-dir', required=True, help='Path to the root directory containing subfolders with images')
@click.option('--save-dir', required=True, help='Path to save the generated collages')
@click.option('--num-columns', required=True, type=int, help='Number of columns in the collage grid')
def main(root_dir, save_dir, num_columns):
    """Create collages from second images in subfolders."""
    print(f"Creating collages from {root_dir}, saving to {save_dir}, with {num_columns} columns...")
    create_collage(root_dir, save_dir, num_columns)
    print("Done!")

if __name__ == '__main__':
    main()