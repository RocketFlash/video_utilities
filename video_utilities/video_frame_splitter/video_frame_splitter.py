import os
import cv2
import random
import logging
from pathlib import Path
from enum import Enum
from typing import Union, Optional, List, Dict, Tuple, Iterator
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from .config import (
    VideoFrameSplitterConfig,
    VideoReaderType
)
from ..video_scene_detector import SceneData

try:
    import decord
    DECORD_INSTALLED = True
except ImportError:
    DECORD_INSTALLED = False

logger = logging.getLogger(__name__)

class FrameSelectionStrategy(Enum):
    """Enumeration of frame selection strategies."""
    INTERVAL = "interval"
    RANDOM = "random" 
    SCENE_BASED = "scene_based"
    MANUAL = "manual"

@dataclass
class VideoFrame:
    """Data class representing a single video frame with metadata."""
    image: np.ndarray
    idx: int
    timestamp: float
    scene_id: Optional[int] = None
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the frame image (H, W, C)."""
        return self.image.shape
    
    @property
    def size(self) -> Tuple[int, int]:
        """Size of the frame image (W, H)."""
        h, w = self.image.shape[:2]
        return w, h
    
@dataclass
class VideoFramesData:
    """Container for video frames and associated metadata."""
    frames: List[VideoFrame]
    # Original video properties
    frame_w_orig: int
    frame_h_orig: int
    video_len_orig: float
    n_frames_orig: int
    fps: float
    # Processing parameters
    frame_w: int
    frame_h: int
    start_idx: int
    start_sec: float
    selection_strategy: FrameSelectionStrategy
    # Selection parameters
    n_frames_max: Optional[int] = None
    n_sec_max: Optional[float] = None
    frame_interval: Optional[int] = None
    frame_interval_sec: Optional[float] = None
    # Results
    video_len: float = 0.0
    selected_n_frames: int = field(init=False)
    selected_frame_idxs: List[int] = field(init=False)
    selected_frame_scene_ids: List[int] = field(init=False)
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.selected_n_frames = len(self.frames)
        self.selected_frame_idxs = [f.idx for f in self.frames]
        self.selected_frame_scene_ids = [f.scene_id or -1 for f in self.frames]
        
        if self.frames:
            self.video_len = round(
                (self.frames[-1].timestamp - self.frames[0].timestamp), 3
            )
    
    @property
    def total_frames(self) -> int:
        """Total number of selected frames."""
        return len(self.frames)
    
    @property
    def frame_shape(self) -> Tuple[int, int]:
        """Shape of processed frames (W, H)."""
        return self.frame_w, self.frame_h
    
    def get_scene_frames_dict(self) -> Dict[int, List[VideoFrame]]:
        """Group frames by scene ID."""
        scene_frames = {}
        for frame in self.frames:
            scene_id = frame.scene_id or -1
            if scene_id not in scene_frames:
                scene_frames[scene_id] = []
            scene_frames[scene_id].append(frame)
        return scene_frames
    
    def get_frames_in_time_range(self, start_sec: float, end_sec: float) -> List[VideoFrame]:
        """Get frames within a specific time range."""
        return [
            frame for frame in self.frames 
            if start_sec <= frame.timestamp <= end_sec
        ]
    
    def __str__(self) -> str:
        return (
            f"VideoFramesData: {self.selected_n_frames} frames "
            f"({self.video_len:.2f}s) from {self.n_frames_orig} total frames "
            f"({self.video_len_orig:.2f}s), strategy: {self.selection_strategy.value}"
        )

class VideoFrameSplitter:
    """
    Extract frames from video files with various selection strategies.
    
    Supports interval-based, random, scene-based, and manual frame selection
    with configurable preprocessing options.
    """
    
    def __init__(self, config: Optional[VideoFrameSplitterConfig] = None):
        """Initialize the video frame splitter."""
        self.config = config or VideoFrameSplitterConfig()
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """Validate required dependencies are available."""
        if (self.config.video_reader_type == VideoReaderType.DECORD and 
            not DECORD_INSTALLED):
            logger.warning("Decord not available, falling back to OpenCV")
            self.config.video_reader_type = VideoReaderType.OPENCV
    
    def _open_video(self, video_path: Union[str, Path]) -> Tuple[object, int, float, Tuple[int, int]]:
        """Open video file and return reader, frame count, fps, and dimensions."""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        reader_type = self.config.video_reader_type
        
        # Try decord first for AUTO or DECORD
        if reader_type in [VideoReaderType.AUTO, VideoReaderType.DECORD] and DECORD_INSTALLED:
            try:
                video = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
                n_frames = len(video)
                fps = video.get_avg_fps()
                # Get frame dimensions from first frame
                sample_frame = video[0].asnumpy()
                frame_h, frame_w = sample_frame.shape[:2]
                return video, n_frames, fps, (frame_w, frame_h)
            except Exception as e:
                if reader_type == VideoReaderType.DECORD:
                    raise RuntimeError(f"Failed to open video with decord: {e}")
                logger.warning(f"Decord failed, falling back to OpenCV: {e}")
        
        # Use OpenCV
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video with OpenCV: {video_path}")
        
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return video, n_frames, fps, (frame_w, frame_h)
    
    def _determine_selection_strategy(
        self,
        scene_list: Optional[List[SceneData]],
        selected_frame_idxs: Optional[List[int]],
        selected_seconds: Optional[List[float]]
    ) -> FrameSelectionStrategy:
        """Determine which frame selection strategy to use."""
        if selected_frame_idxs is not None or selected_seconds is not None:
            return FrameSelectionStrategy.MANUAL
        elif self.config.n_random_frames is not None:
            return FrameSelectionStrategy.RANDOM
        elif scene_list is not None:
            return FrameSelectionStrategy.SCENE_BASED
        else:
            return FrameSelectionStrategy.INTERVAL
    
    def _get_manual_frame_indices(
        self,
        selected_frame_idxs: Optional[List[int]],
        selected_seconds: Optional[List[float]],
        fps: float
    ) -> List[int]:
        """Get frame indices from manual selection."""
        if selected_frame_idxs is not None:
            return list(selected_frame_idxs)
        elif selected_seconds is not None:
            return [max(0, int(round(sec * fps))) for sec in selected_seconds]
        return []
    
    def _get_random_frame_indices(
        self, 
        n_frames_total: int, 
        fps: float
    ) -> Tuple[List[int], Optional[int], Optional[float]]:
        """Get random frame indices within specified constraints."""
        start_idx = self._get_effective_start_idx(fps)
        frame_interval, frame_interval_sec = self._get_effective_frame_interval(fps)
        
        # Determine end index
        end_idx = n_frames_total
        if self.config.n_sec_max is not None:
            end_idx = min(end_idx, start_idx + int(self.config.n_sec_max * fps))
        elif self.config.n_frames_max is not None:
            end_idx = min(end_idx, start_idx + (self.config.n_frames_max * frame_interval))
        
        # Create population of available frames
        frame_population = list(range(start_idx, end_idx, frame_interval))
        
        if self.config.n_random_frames >= len(frame_population):
            logger.warning(
                f"Requested {self.config.n_random_frames} random frames, "
                f"but only {len(frame_population)} available. Using all."
            )
            return frame_population, frame_interval, frame_interval_sec
        
        indices = sorted(random.sample(frame_population, self.config.n_random_frames))
        return indices, frame_interval, frame_interval_sec
    
    def _get_effective_start_idx(self, fps: float) -> int:
        """Get the effective starting frame index."""
        if self.config.start_sec is not None:
            return max(0, int(self.config.start_sec * fps))
        return max(0, self.config.start_idx)
    
    def _get_effective_frame_interval(self, fps: float) -> Tuple[int, float]:
        """Get the effective frame interval."""
        if self.config.frame_interval_sec is not None:
            interval = max(1, int(round(self.config.frame_interval_sec * fps)))
            interval_sec = interval / fps
        else:
            interval = max(1, self.config.frame_interval)
            interval_sec = interval / fps
        return interval, interval_sec
    
    def _get_interval_frame_indices(
        self, 
        n_frames_total: int, 
        fps: float
    ) -> Tuple[List[int], int, float]:
        """Get frame indices using interval-based selection."""
        start_idx = self._get_effective_start_idx(fps)
        frame_interval, frame_interval_sec = self._get_effective_frame_interval(fps)
        
        # Calculate maximum frames to select
        n_frames_available = n_frames_total - start_idx
        n_frames_max = n_frames_available
        
        if self.config.n_sec_max is not None:
            n_frames_max = min(n_frames_max, int(self.config.n_sec_max * fps))
        if self.config.n_frames_max is not None:
            n_frames_max = min(n_frames_max, self.config.n_frames_max)
        
        end_idx = min(n_frames_total, start_idx + n_frames_max)
        indices = list(range(start_idx, end_idx, frame_interval))
        
        return indices, frame_interval, frame_interval_sec
    
    def _calculate_scene_frame_count(self, scene: SceneData, fps: float) -> int:
        """Calculate number of frames to select from a scene."""
        # Use l_sec if available, otherwise calculate from frame count
        if scene.l_sec is not None:
            scene_duration = scene.l_sec
        else:
            n_frames = scene.n_frames or (scene.end_frame - scene.start_frame + 1)
            scene_duration = n_frames / fps
        
        if scene_duration <= self.config.scene_length_threshold:
            # Linear interpolation for short scenes
            proportion = scene_duration / self.config.scene_length_threshold
            n_frames = int(
                self.config.min_n_frames_per_scene + 
                (self.config.max_n_frames_per_scene - self.config.min_n_frames_per_scene) * proportion
            )
        else:
            n_frames = self.config.max_n_frames_per_scene
        
        # Ensure we don't exceed scene length
        max_possible = scene.n_frames or (scene.end_frame - scene.start_frame + 1)
        return max(self.config.min_n_frames_per_scene, min(n_frames, max_possible))
    
    def _get_scene_frame_indices(
        self, 
        scene_list: List[SceneData], 
        n_frames_total: int, 
        fps: float
    ) -> Tuple[List[int], List[int]]:
        """Get frame indices using scene-based selection."""
        if not scene_list:
            # Create single scene covering entire video
            scene_list = [SceneData(
                scene_id=0, 
                start_frame=0, 
                end_frame=n_frames_total - 1
            )]
        
        start_idx = self._get_effective_start_idx(fps)
        selected_indices = []
        scene_ids = []
        
        for scene in scene_list:
            # Skip scenes that end before our start index
            if scene.end_frame < start_idx:
                continue
            
            # Adjust scene boundaries if needed
            scene_start = max(scene.start_frame, start_idx)
            scene_end = scene.end_frame
            
            if scene_start >= scene_end:
                continue
            
            # Calculate frames to select from this scene
            adjusted_scene = SceneData(
                scene_id=scene.scene_id,
                start_frame=scene_start,
                end_frame=scene_end
            ).calculate_timing(fps)
            
            n_frames_scene = self._calculate_scene_frame_count(adjusted_scene, fps)
            scene_indices = self._distribute_frames_in_range(
                scene_start, scene_end, n_frames_scene
            )
            
            selected_indices.extend(scene_indices)
            scene_ids.extend([scene.scene_id] * len(scene_indices))
        
        return selected_indices, scene_ids
    
    def _distribute_frames_in_range(
        self, 
        start_idx: int, 
        end_idx: int, 
        n_frames: int
    ) -> List[int]:
        """Distribute n_frames evenly across the given range."""
        if n_frames <= 0:
            return []
        if n_frames == 1:
            return [int((start_idx + end_idx) / 2)]
        if n_frames >= (end_idx - start_idx):
            return list(range(start_idx, end_idx + 1))
        
        # Use numpy for even distribution
        indices = np.linspace(start_idx, end_idx, n_frames, dtype=int)
        return indices.tolist()
    
    def _read_frames_decord(
        self, 
        video: object, 
        frame_indices: List[int]
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Read frames using decord in batches."""
        batch_size = self.config.batch_size
        
        for i in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[i:i + batch_size]
            try:
                batch_frames = video.get_batch(batch_indices).asnumpy()
                for j, frame_idx in enumerate(batch_indices):
                    yield frame_idx, batch_frames[j]
            except Exception as e:
                logger.error(f"Error reading batch starting at index {i}: {e}")
                # Fallback to individual frame reading
                for frame_idx in batch_indices:
                    try:
                        frame = video[frame_idx].asnumpy()
                        yield frame_idx, frame
                    except Exception as frame_error:
                        logger.error(f"Failed to read frame {frame_idx}: {frame_error}")
    
    def _read_frames_opencv(
        self, 
        video: cv2.VideoCapture, 
        frame_indices: List[int]
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Read frames using OpenCV."""
        for frame_idx in sorted(frame_indices):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame
            else:
                logger.warning(f"Failed to read frame {frame_idx}")
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame according to configuration."""
        if self.config.frame_max_size is None:
            return frame
        
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.config.frame_max_size:
            return frame
        
        if self.config.maintain_aspect_ratio:
            scale = self.config.frame_max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            new_w = new_h = self.config.frame_max_size
        
        return cv2.resize(
            frame, 
            (new_w, new_h), 
            interpolation=self.config.interpolation_method
        )
    
    def extract_frames(
        self,
        video_path: Union[str, Path],
        scene_list: Optional[List[SceneData]] = None,
        selected_frame_idxs: Optional[List[int]] = None,
        selected_seconds: Optional[List[float]] = None,
    ) -> Optional[VideoFramesData]:
        """
        Extract frames from video according to configuration and parameters.
        
        Args:
            video_path: Path to the video file
            scene_list: Optional list of scene data for scene-based selection
            selected_frame_idxs: Optional manual frame indices
            selected_seconds: Optional manual timestamps in seconds
            
        Returns:
            VideoFramesData object containing extracted frames and metadata,
            or None if extraction failed
        """
        try:
            # Open video and get basic info
            video, n_frames_total, fps, (frame_w_orig, frame_h_orig) = self._open_video(video_path)
            
            if n_frames_total == 0:
                logger.error("Video contains no frames")
                return None
            
            # Determine selection strategy and get frame indices
            strategy = self._determine_selection_strategy(
                scene_list, selected_frame_idxs, selected_seconds
            )
            
            frame_indices = []
            scene_ids = []
            frame_interval = None
            frame_interval_sec = None
            
            if strategy == FrameSelectionStrategy.MANUAL:
                frame_indices = self._get_manual_frame_indices(
                    selected_frame_idxs, selected_seconds, fps
                )
            elif strategy == FrameSelectionStrategy.RANDOM:
                frame_indices, frame_interval, frame_interval_sec = self._get_random_frame_indices(
                    n_frames_total, fps
                )
            elif strategy == FrameSelectionStrategy.SCENE_BASED:
                frame_indices, scene_ids = self._get_scene_frame_indices(
                    scene_list, n_frames_total, fps
                )
            else:  # INTERVAL
                frame_indices, frame_interval, frame_interval_sec = self._get_interval_frame_indices(
                    n_frames_total, fps
                )
            
            if not frame_indices:
                logger.warning("No frames selected based on criteria")
                return None
            
            # Ensure indices are within bounds
            frame_indices = [idx for idx in frame_indices if 0 <= idx < n_frames_total]
            
            if not frame_indices:
                logger.warning("All selected frame indices are out of bounds")
                return None
            
            # Read frames
            frames = []
            
            if isinstance(video, cv2.VideoCapture):
                frame_reader = self._read_frames_opencv(video, frame_indices)
                reader_name = "OpenCV"
            else:  # decord
                frame_reader = self._read_frames_decord(video, frame_indices)
                reader_name = "Decord"
            
            # Process frames with progress bar
            if self.config.show_progress:
                frame_reader = tqdm(
                    frame_reader, 
                    total=len(frame_indices), 
                    desc=f"Processing frames ({reader_name})"
                )
            
            for frame_idx, frame_image in frame_reader:
                # Resize if needed
                processed_frame = self._resize_frame(frame_image)
                
                # Determine scene ID for this frame
                if strategy == FrameSelectionStrategy.SCENE_BASED and scene_ids:
                    scene_id = scene_ids[len(frames)] if len(frames) < len(scene_ids) else None
                else:
                    scene_id = None
                
                frames.append(VideoFrame(
                    image=processed_frame,
                    idx=frame_idx,
                    timestamp=round(frame_idx / fps, 3),
                    scene_id=scene_id
                ))
            
            # Clean up
            if isinstance(video, cv2.VideoCapture):
                video.release()
            
            if not frames:
                logger.error("No frames were successfully read")
                return None
            
            # Get final frame dimensions
            final_frame_h, final_frame_w = frames[0].image.shape[:2]
            
            return VideoFramesData(
                frames=frames,
                frame_w_orig=frame_w_orig,
                frame_h_orig=frame_h_orig,
                video_len_orig=n_frames_total / fps if fps > 0 else 0,
                n_frames_orig=n_frames_total,
                fps=fps,
                frame_w=final_frame_w,
                frame_h=final_frame_h,
                start_idx=self._get_effective_start_idx(fps),
                start_sec=self._get_effective_start_idx(fps) / fps if fps > 0 else 0,
                selection_strategy=strategy,
                n_frames_max=self.config.n_frames_max,
                n_sec_max=self.config.n_sec_max,
                frame_interval=frame_interval,
                frame_interval_sec=frame_interval_sec,
            )
            
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            return None

    def __call__(self, *args, **kwargs) -> Optional[VideoFramesData]:
        """Backward compatibility method."""
        return self.extract_frames(*args, **kwargs)