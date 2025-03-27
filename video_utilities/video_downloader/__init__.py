import os
import requests
from typing import Union
from pathlib import Path
from tqdm.auto import tqdm


class VideoDownloader:
    r"""
    VideoDownloader downloads video using specific api

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
        overwrite_if_exist (`bool`, *optional*, defaults to `False`):
            If True overwrites file if it already exists in the folder
        if_quality_not_exist_strategy (`str`, *optional*, defaults to `none`):
            strategy in case required quality is not available, could be 'none', 'lower', 'higher'
    """
    def __init__(
        self,
        url_template: str,
        secret: str,
        quality: str = 'max',
        save_dir: Union[str, os.PathLike] = './',
        chunk_size: int = 4096,
        overwrite_if_exist: bool = False,
        if_quality_not_exist_strategy='none'
    ):
        self.url_template = url_template
        self.secret = secret
        self.quality = quality
        self.save_dir = Path(save_dir)
        self.chunk_size = chunk_size
        self.overwrite_if_exist = overwrite_if_exist
        self.if_quality_not_exist_strategy = if_quality_not_exist_strategy


    def get_download_links(
        self,
        video_ids: Union[list, str],
    ):
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
        verbose=False
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
            required_quality = int(self.quality.replace('p', ''))

            if required_quality in available_qualities:
                quality_str = f'{self.quality}'
            else:
                if self.if_quality_not_exist_strategy=='lower':
                    lower_list = []
                    for quality in available_qualities:
                        if quality<required_quality:
                            lower_list.append(quality)
                    if len(lower_list):
                        quality_str = f'{max(lower_list)}p'
                    else:
                        quality_str = f'{self.quality}'

                    if verbose:
                        print(f'Can not download {self.quality}, downloading {quality_str} instead')
                elif self.if_quality_not_exist_strategy=='higher':
                    higher_list = []
                    for quality in available_qualities:
                        if quality>required_quality:
                            higher_list.append(quality)
                    if len(higher_list):
                        quality_str = f'{min(higher_list)}p'
                    else:
                        quality_str = f'{self.quality}'

                    if verbose:
                        print(f'Can not download {self.quality}, downloading {quality_str} instead')
                else:
                    quality_str = f'{self.quality}'
        
        success = False
        save_path = ''

        if quality_str in quality_to_url_dict:
            video_url = quality_to_url_dict[quality_str]
            video_name = video_url.split('/')[-1]
            save_name = f'{id_name}_{video_name}'
            save_path = self.save_dir / save_name

            if save_path.is_file() and not self.overwrite_if_exist:
                success = True
                if verbose:
                    print(f"Video already exists in {save_path}")
                return success, str(save_path)
    
            response = requests.get(video_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
    
            if response.status_code == 200:
                if verbose: 
                    bar = tqdm(
                        desc="Downloading video",
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=self.chunk_size,
                    )

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            if verbose:
                                bar.update(len(chunk))
                success = True
                if verbose:   
                    bar.close()      
                    print(f"Video downloaded successfully as {save_path}")
            else:
                if verbose:
                    print(f"Error: Received status code {response.status_code}")
            
        else:
            if verbose:
                print(f"Video in {quality_str} quality does not exist")

        return success, str(save_path)


    def __call__(
        self,
        video_ids: Union[list, str],
        verbose: bool = False
    ):
        if isinstance(video_ids, str):
            video_ids = [video_ids]
            
        links_data = self.get_download_links(
            video_ids=video_ids,
        )
        status_dict = {}

        for video_id, video_data in zip(video_ids, links_data['result']):
            success, video_path = self.download_video(
                video_data,
                verbose=verbose
            )
            status_dict[video_id] = dict(
                success=success,
                video_path=video_path
            )

        return status_dict

        