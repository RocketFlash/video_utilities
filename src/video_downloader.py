import os
import requests
from typing import Union
from pathlib import Path
from tqdm.auto import tqdm


class VideoDownloader:
    r"""
    VideoDownloader downloads video using api

    Args:
        url_template (`str`):
            Get download links API template
        secret (`str`):
            Secret to API
        quality (`str`, *optional*, defaults to `max`):
            Quality of downloaded video ['max', 'min', '480p', '720p', '1080p']
        save_dir (`str` or `Path`, *optional*, defaults to `./`):
            Save directory path
        chunk_size (`int`, *optional*, defaults to `4096`):
            Download file chunk size
    """
    def __init__(
        self,
        url_template: str,
        secret: str,
        quality: str = 'max',
        save_dir: Union[str, os.PathLike] = './',
        chunk_size: int = 4096
    ):
        self.url_template = url_template
        self.secret = secret
        self.quality = quality
        self.save_dir = Path(save_dir)
        self.chunk_size = chunk_size


    def get_download_links(
        self,
        video_ids: Union[list, str],
    ):
        if isinstance(video_ids, str):
            video_ids = [video_ids]
    
        video_ids_str = ','.join(video_ids)
        url = self.url_template.format(video_ids_str, self.secret)
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Error: Received status code {response.status_code}")
            data = None
            
        return data


    def download_video(
        self,
        links_dict,
    ):
        id_name = links_dict['id']
    
        quality_to_url_dict = {
            link_dict['quality'] : link_dict['url'] 
            for link_dict in links_dict['sources']
        } 
    
        available_qualities = sorted([
            int(q.replace('p', '')) 
            for q in quality_to_url_dict.keys()
        ])
        
        if self.quality=='min':
            quality_str = f'{available_qualities[0]}p'
        elif self.quality=='max':
            quality_str = f'{available_qualities[-1]}p'
        else:
            quality_str = f'{self.quality}'
        
        if quality_str in quality_to_url_dict:
            video_url = quality_to_url_dict[quality_str]
            video_name = video_url.split('/')[-1]
            save_name = f'{id_name}_{video_name}'
            save_path = self.save_dir / save_name
    
            response = requests.get(video_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
    
            if response.status_code == 200:
                with open(save_path, 'wb') as f, tqdm(
                    desc="Downloading video",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=self.chunk_size,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                            
                print(f"Video downloaded successfully as {save_path}")
            else:
                print(f"Error: Received status code {response.status_code}")
            
        else:
            print(f"Video in {quality_str} quality does not exist")


    def __call__(
        self,
        video_ids: Union[list, str],
    ):
        links_data = self.get_download_links(
            video_ids=video_ids,
        )
        for video_data in links_data['result']:
            self.download_video(
                video_data,
            )
        