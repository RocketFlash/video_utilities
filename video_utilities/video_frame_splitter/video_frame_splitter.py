import os
import cv2
import random
import numpy as np
from tqdm.auto import tqdm
from typing import (
    Union, 
    Optional,
    List, 
    Dict,
    Tuple
)

try:
    import decord
    DECORD_INSTALLED = True
except ImportError:
    DECORD_INSTALLED = False

from dataclasses import dataclass
from .config import VideoFrameSplitterConfig
from ..video_scene_detector import SceneData


@dataclass
class VideoFrame:
    image: np.ndarray
    idx: int
    timestamp: float
    scene_id: Optional[int]
    

@dataclass
class VideoFramesData:
    frames: List[VideoFrame]
    frame_w_orig: int
    frame_h_orig: int
    video_len_orig: float
    n_frames_orig: int
    fps: float
    frame_w: int
    frame_h: int
    start_idx: int
    start_sec: float
    n_frames_max: Optional[int]
    n_sec_max: Optional[float]
    frame_interval: Optional[int]
    frame_interval_sec: Optional[float]
    video_len: float
    is_scene_based_selection: bool
    selected_n_frames: int
    selected_frame_idxs: List[int]
    selected_frame_scene_ids: List[int]
    
    def __str__(self):
        info_str = f"Original video info:\n"
        info_str += f"    original video frame width     : {self.frame_w_orig}\n"
        info_str += f"    original video frame height    : {self.frame_h_orig}\n"
        info_str += f"    original video len in seconds  : {self.video_len_orig}\n"
        info_str += f"    original video len in n frames : {self.n_frames_orig}\n"
        info_str += f"    fps : {self.fps}\n"
        info_str += f"\n"
        info_str += f"Splitted video info:\n"
        info_str += f"    frame width    : {self.frame_w}\n"
        info_str += f"    frame height   : {self.frame_h}\n"
        info_str += f"    start index    : {self.start_idx}\n"
        info_str += f"    start second   : {self.start_sec}\n"
        info_str += f"    n frames max   : {self.n_frames_max}\n"
        info_str += f"    n seconds max  : {self.n_sec_max}\n"
        info_str += f"    frame interval : {self.frame_interval}\n"
        info_str += f"    frame interval in seconds: {self.frame_interval_sec}\n"
        info_str += f"    video len in seconds     : {self.video_len}\n"
        info_str += f"    scene based selection    : {self.is_scene_based_selection}\n"
        info_str += f"    selected n frames        : {self.selected_n_frames}\n"
        info_str += f"    selected frame indexes   : {self.selected_frame_idxs}\n"
        info_str += f"    selected frame scene ids : {self.selected_frame_scene_ids}\n"
        return info_str
    
    
    def get_scene_frames_dict(self):
        scene_frames_dict = {}
        for idx, scene_id in enumerate(self.selected_frame_scene_ids):
            if scene_id not in scene_frames_dict:
                scene_frames_dict[scene_id] = []
            scene_frames_dict[scene_id].append(self.frames[idx])
        return scene_frames_dict


class VideoFrameSplitter:
    def __init__(self, config: Optional[VideoFrameSplitterConfig] = None):
        self.config = config if config is not None else self.get_default_config()
        self.set_params_from_config(self.config)

    def get_default_config(self) -> VideoFrameSplitterConfig:
        return VideoFrameSplitterConfig()

    def set_params_from_config(self, config: VideoFrameSplitterConfig):
        for key, value in vars(config).items():
            setattr(self, key, value)

    def _determine_frame_indices(
        self,
        n_frames_total: int,
        fps: float,
        scene_list: Optional[List[SceneData]],
        selected_frame_idxs: Optional[List[int]],
        selected_seconds: Optional[List[float]],
        verbose: bool
    ) -> Tuple[List[int], List[int], bool, Optional[int], Optional[float]]:
        is_scene_based_selection = False
        selected_frame_scene_ids = []

        # Priority 1: User provides exact frame indices or seconds
        if selected_frame_idxs is not None:
            return selected_frame_idxs, [], False, None, None
        if selected_seconds is not None:
            return [int(round(sec * fps)) for sec in selected_seconds], [], False, None, None

        start_idx = int(self.start_sec * fps) if self.start_sec is not None else self.start_idx
        start_idx = min(start_idx, n_frames_total - 1)

        frame_interval = int(round(self.frame_interval_sec * fps)) if self.frame_interval_sec is not None else self.frame_interval
        frame_interval = max(1, frame_interval)
        frame_interval_sec = frame_interval / fps if fps > 0 else -1

        # Priority 2: Random frame selection
        if self.n_random_frames is not None:
            end_idx = n_frames_total
            if self.n_sec_max is not None:
                end_idx = start_idx + int(self.n_sec_max * fps)
            elif self.n_frames_max is not None:
                end_idx = start_idx + (self.n_frames_max * frame_interval)
            end_idx = min(end_idx, n_frames_total)
            
            frame_population = range(start_idx, end_idx, frame_interval)
            if self.n_random_frames >= len(frame_population):
                if verbose:
                    print(f"Warning: Requested {self.n_random_frames} random frames, but only {len(frame_population)} are available. Returning all.")
                indices = list(frame_population)
            else:
                indices = sorted(random.sample(list(frame_population), self.n_random_frames))
            return indices, [], False, frame_interval, frame_interval_sec

        # Priority 3: Scene-based selection
        if scene_list is not None:
            is_scene_based_selection = True
            indices, scene_ids = self.get_selected_frames_from_scenes(scene_list, start_idx, n_frames_total, fps)
            return indices, scene_ids, is_scene_based_selection, None, None

        # Priority 4: Default interval-based selection
        n_frames_available = (n_frames_total - start_idx)
        n_frames_max = self.n_frames_max
        if self.n_sec_max is not None:
            n_frames_max = int(self.n_sec_max * fps)

        n_frames_to_select = n_frames_available
        if n_frames_max is not None:
            n_frames_to_select = min(n_frames_max, n_frames_available)

        indices = list(range(start_idx, start_idx + n_frames_to_select, frame_interval))
        return indices, [], False, frame_interval, frame_interval_sec

    def __call__(
        self,
        video_path: Union[str, os.PathLike],
        scene_list: Optional[List[SceneData]] = None,
        selected_frame_idxs: Optional[List[int]] = None,
        selected_seconds: Optional[List[float]] = None,
        verbose: bool = True
    ) -> Optional[VideoFramesData]:
        
        if self.video_reader_type == 'decord' and DECORD_INSTALLED:
            try:
                video = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
                n_frames_total = len(video)
                fps = video.get_avg_fps()
            except decord.DECORDError as e:
                if verbose: print(f"Decord failed to open {video_path}: {e}")
                return None
        else:
            if self.video_reader_type == 'decord':
                if verbose: print("Warning: 'decord' not found or failed, using slow 'opencv' reader.")
            video = cv2.VideoCapture(str(video_path))
            if not video.isOpened():
                if verbose: print(f"OpenCV failed to open {video_path}")
                return None
            n_frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(video.get(cv2.CAP_PROP_FPS))

        if n_frames_total == 0:
            if verbose: print("Video has no frames.")
            return None
        
        (
            final_selected_idxs,
            selected_frame_scene_ids,
            is_scene_based_selection,
            final_frame_interval,
            final_frame_interval_sec
        ) = self._determine_frame_indices(n_frames_total, fps, scene_list, selected_frame_idxs, selected_seconds, verbose)

        if not final_selected_idxs:
            if verbose: print("No frames were selected based on the criteria.")
            return None

        frames: List[VideoFrame] = []
        
        if self.video_reader_type == 'decord' and DECORD_INSTALLED:
            try:
                image_batch = video.get_batch(final_selected_idxs).asnumpy()
                
                iterator = tqdm(enumerate(final_selected_idxs), total=len(final_selected_idxs), desc="Processing Frames") if verbose else enumerate(final_selected_idxs)
                
                for i, frame_idx in iterator:
                    scene_id = selected_frame_scene_ids[i] if is_scene_based_selection else None
                    frames.append(VideoFrame(
                        image=image_batch[i],
                        idx=frame_idx,
                        timestamp=round(frame_idx / fps, 3),
                        scene_id=scene_id
                    ))
            except Exception as e:
                if verbose: print(f"Error during decord batch processing: {e}")
                return None # or handle fallback
        
        else:
            iterator = tqdm(sorted(final_selected_idxs), desc="Processing Frames (OpenCV)", total=len(final_selected_idxs)) if verbose else sorted(final_selected_idxs)
            for frame_idx in iterator:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, image = video.read()
                if not success:
                    if verbose: print(f"Warning: Failed to read frame {frame_idx}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                scene_id = selected_frame_scene_ids[final_selected_idxs.index(frame_idx)] if is_scene_based_selection else None
                frames.append(VideoFrame(
                    image=image,
                    idx=frame_idx,
                    timestamp=round(frame_idx / fps, 3),
                    scene_id=scene_id
                ))
            video.release()

        if not frames:
            return None
        
        frame_h_orig, frame_w_orig = frames[0].image.shape[:2]
        resize_scale = 1.0
        if self.frame_max_size and max(frame_h_orig, frame_w_orig) > self.frame_max_size:
            resize_scale = self.frame_max_size / max(frame_h_orig, frame_w_orig)
            for frame in frames:
                frame.image = cv2.resize(frame.image, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)

        final_frame_h, final_frame_w = frames[0].image.shape[:2]
        video_len = round((final_selected_idxs[-1] - final_selected_idxs[0]) / fps, 3) if fps > 0 else -1
        
        return VideoFramesData(
            frames=frames,
            frame_w_orig=frame_w_orig,
            frame_h_orig=frame_h_orig,
            video_len_orig=n_frames_total / fps if fps > 0 else -1,
            n_frames_orig=n_frames_total,
            fps=fps,
            frame_w=final_frame_w,
            frame_h=final_frame_h,
            start_idx=self.start_idx,
            start_sec=self.start_sec if self.start_sec is not None else self.start_idx / fps,
            n_frames_max=self.n_frames_max,
            n_sec_max=self.n_sec_max,
            frame_interval=final_frame_interval,
            frame_interval_sec=final_frame_interval_sec,
            video_len=video_len,
            is_scene_based_selection=is_scene_based_selection,
            selected_n_frames=len(frames),
            selected_frame_idxs=final_selected_idxs,
            selected_frame_scene_ids=selected_frame_scene_ids,
        )

    
    def get_frame_idxs_from_interval(self, start_idx: int, end_idx: int, n_frames: int = 1) -> List[int]:
        frame_indexes_list = list(range(start_idx, end_idx))
        if n_frames >= len(frame_indexes_list): return frame_indexes_list
        if n_frames == 1: return [int((end_idx + start_idx) / 2)]
        indices = np.linspace(0, len(frame_indexes_list) - 1, n_frames, dtype=int)
        return [frame_indexes_list[i] for i in indices]

    def calculate_frames_to_select(self, start_idx: int, end_idx: int, fps: float) -> int:
        scene_length_seconds = (end_idx - start_idx + 1) / fps
        if scene_length_seconds <= self.scene_length_threshold:
            proportion = scene_length_seconds / self.scene_length_threshold
            n_frames = int(self.min_n_frames_per_scene + (self.max_n_frames_per_scene - self.min_n_frames_per_scene) * proportion)
        else:
            n_frames = self.max_n_frames_per_scene
        return max(self.min_n_frames_per_scene, min(n_frames, end_idx - start_idx + 1))

    def get_selected_frames_from_scenes(self, scene_list: List[SceneData], start_idx: int, n_frames_total: int, fps: float) -> Tuple[List[int], List[int]]:
        if not scene_list:
            scene_list = [SceneData(scene_id=0, start_frame=0, end_frame=n_frames_total)]
        
        selected_frame_idxs = []
        selected_frame_scene_ids = []
        for scene in scene_list:
            start_scene_idx, end_scene_idx = scene.start_frame, scene.end_frame
            if start_idx >= end_scene_idx: continue
            if start_idx > start_scene_idx: start_scene_idx = start_idx
            
            n_frames_scene = self.calculate_frames_to_select(start_scene_idx, end_scene_idx, fps)
            scene_selected_idxs = self.get_frame_idxs_from_interval(start_scene_idx, end_scene_idx, n_frames_scene)
            
            selected_frame_idxs.extend(scene_selected_idxs)
            selected_frame_scene_ids.extend([scene.scene_id] * len(scene_selected_idxs))
        return selected_frame_idxs, selected_frame_scene_ids