import numpy as np
from PIL import Image
from ..video_frame_splitter import VideoFrame
from ..video_captioner import VideoFrameOutputResult
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter


def plot_bar_charts(
    frames_results: List[Union[Dict, VideoFrameOutputResult]],
    per_scene: bool = False,
    frames: Optional[List[Union[np.ndarray, VideoFrame]]] = None,
    resize_to: int = 64,
    num_cols_frames: int = 3,
    num_cols_stats: int = 3,
    save_dir: Optional[Union[str, Path]] = None,
    video_name: str = "video_tagging_results",
    show: bool = True
):
    all_tags = {}
    scene_tags = {}

    for frame in frames_results:
        if isinstance(frame, VideoFrameOutputResult):
            outputs = frame.outputs
            scene_id = frame.scene_id
        else:
            outputs = frame
            scene_id = 0  # Default scene if not provided

        for category, tags in outputs.items():
            if category not in all_tags:
                all_tags[category] = []
            if tags is not None:
                all_tags[category].extend(tags)

            if per_scene:
                if scene_id not in scene_tags:
                    scene_tags[scene_id] = {}
                if category not in scene_tags[scene_id]:
                    scene_tags[scene_id][category] = []
                if tags is not None:
                    scene_tags[scene_id][category].extend(tags)

    if save_dir:
        save_dir = Path(save_dir) / video_name
        save_dir.mkdir(parents=True, exist_ok=True)

    if per_scene:
        for scene_id, scene_data in scene_tags.items():
            scene_frames = [f for f in frames if (isinstance(f, VideoFrame) and f.scene_id == scene_id) or 
                            (isinstance(f, np.ndarray) and frames_results[frames.index(f)].scene_id == scene_id)] if frames else None
            fig = _create_subplots(
                scene_data, 
                f"File: {video_name}, Scene {scene_id}", 
                scene_frames, 
                resize_to, 
                num_cols_frames, 
                num_cols_stats
            )
            
            if save_dir:
                fig.savefig(save_dir / f'scene_{scene_id}.png')
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    else:
        fig = _create_subplots(
            all_tags, 
            f"File: {video_name}, Full video", 
            frames, 
            resize_to, 
            num_cols_frames, 
            num_cols_stats
        )
        
        if save_dir:
            fig.savefig(save_dir / 'full_video.png')
        
        if show:
            plt.show()
        else:
            plt.close(fig)

def _create_subplots(
    tag_data: Dict[str, List[str]], 
    title_prefix: str, 
    frames: Optional[List[Union[np.ndarray, VideoFrame]]], 
    resize_to: int, 
    num_cols_frames: int, 
    num_cols_stats: int
):
    tag_counts = {category: Counter(tags) for category, tags in tag_data.items()}
    num_categories = len(tag_counts)
    
    num_rows_stats = (num_categories - 1) // num_cols_stats + 1
    fig_title_height = 0.05
    fig_title_font_size = 28
    category_font_size = 18
    xticks_font_size = 14
    if frames:
        num_frame_rows = (len(frames) - 1) // num_cols_frames + 1
        fig = plt.figure(figsize=(20, 5 * num_frame_rows + 5 * num_rows_stats + fig_title_height))  
        gs = fig.add_gridspec(3, 1, height_ratios=[fig_title_height, num_frame_rows, num_rows_stats])  

        fig.suptitle(f'{title_prefix}', fontsize=fig_title_font_size, y=0.98) 

        ax_frames = fig.add_subplot(gs[1])  
        _plot_frames(ax_frames, frames, resize_to, num_cols_frames)

        # Tag categories subplots
        gs_tags = gs[2].subgridspec(num_rows_stats, num_cols_stats, hspace=0.4, wspace=0.3) 
    else:
        fig = plt.figure(figsize=(20, 5 * num_rows_stats + fig_title_height)) 
        gs = fig.add_gridspec(2, 1, height_ratios=[fig_title_height, num_rows_stats])  
        
        fig.suptitle(f'{title_prefix}', fontsize=fig_title_font_size, y=0.98)  

        gs_tags = gs[1].subgridspec(num_rows_stats, num_cols_stats, hspace=0.4, wspace=0.3)  

    for idx, (category, counts) in enumerate(tag_counts.items()):
        row = idx // num_cols_stats
        col = idx % num_cols_stats
        ax = fig.add_subplot(gs_tags[row, col])

        tags = list(counts.keys())
        values = list(counts.values())

        ax.bar(tags, values)
        ax.set_title(f'{category.capitalize()}', fontsize=category_font_size)
        # ax.set_xlabel('Tags')
        if idx % num_cols_stats==0:
            ax.set_ylabel('Frequency', fontsize=xticks_font_size)
        ax.tick_params(axis='x', rotation=45, labelsize=xticks_font_size)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

# The _plot_frames function remains unchanged, but uses num_cols_frames instead of num_cols
def _plot_frames(ax, frames, resize_to, num_cols_frames):
    num_frames = len(frames)
    num_rows = (num_frames - 1) // num_cols_frames + 1

    for idx, frame in enumerate(frames):
        row = idx // num_cols_frames
        col = idx % num_cols_frames

        if isinstance(frame, VideoFrame):
            img = frame.image
        else:
            img = frame

        # Resize image
        h, w = img.shape[:2]
        aspect = w / h
        if w > h:
            new_w = resize_to
            new_h = int(resize_to / aspect)
        else:
            new_h = resize_to
            new_w = int(resize_to * aspect)
        img_resized = np.array(Image.fromarray(img).resize((new_w, new_h)))

        ax_sub = ax.inset_axes([col/num_cols_frames, 1-(row+1)/num_rows, 1/num_cols_frames, 1/num_rows])
        ax_sub.imshow(img_resized)
        ax_sub.axis('off')

    ax.axis('off')