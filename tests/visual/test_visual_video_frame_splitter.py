"""
Visual tests for VideoFrameSplitter - generates images for manual inspection.
"""

import pytest
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches

from video_utilities import (
    VideoFrameSplitter, 
    VideoFrameSplitterConfig, 
    SceneData,
    VideoReaderType,
    FrameSelectionStrategy
)

class TestVisualVideoFrameSplitter:
    """Visual tests that save frame extraction results for inspection."""
    
    def test_basic_frame_extraction_visual(self, sample_video_path, test_results_dir):
        """Visualize basic frame extraction."""
        output_dir = test_results_dir / "frame_splitter" / "basic_extraction"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = VideoFrameSplitterConfig(
            n_frames_max=8,
            frame_max_size=256,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(sample_video_path)
        
        if result and result.frames:
            self._save_frame_grid(
                result.frames, 
                output_dir / "basic_extraction_grid.png",
                title=f"Basic Extraction - {len(result.frames)} frames"
            )
            
            # Save individual frames
            for i, frame in enumerate(result.frames):
                frame_path = output_dir / f"frame_{i:03d}_idx{frame.idx}_t{frame.timestamp:.2f}s.jpg"
                self._save_frame(frame.image, frame_path)
    
    def test_selection_strategies_visual(self, sample_video_path, test_results_dir):
        """Visualize different frame selection strategies."""
        output_dir = test_results_dir / "frame_splitter" / "selection_strategies"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        strategies = [
            ("interval", {"frame_interval": 10, "n_frames_max": 6}),
            ("time_based", {"frame_interval_sec": 2.0, "n_sec_max": 10.0}),
            ("random", {"n_random_frames": 6, "start_sec": 1.0, "n_sec_max": 12.0}),
        ]
        
        results = {}
        for strategy_name, params in strategies:
            config = VideoFrameSplitterConfig(show_progress=False, **params)
            splitter = VideoFrameSplitter(config)
            result = splitter.extract_frames(sample_video_path)
            
            if result and result.frames:
                results[strategy_name] = result
                
                # Save grid for each strategy
                self._save_frame_grid(
                    result.frames,
                    output_dir / f"{strategy_name}_grid.png",
                    title=f"{strategy_name.title()} Strategy - {len(result.frames)} frames"
                )
        
        # Create comparison grid
        self._create_strategy_comparison(results, output_dir / "strategy_comparison.png")
    
    def test_scene_based_visual(self, sample_video_path, test_results_dir):
        """Visualize scene-based frame selection."""
        output_dir = test_results_dir / "frame_splitter" / "scene_based"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        cap = cv2.VideoCapture(str(sample_video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Create scenes
        scenes = self._create_test_scenes(frame_count, fps, n_scenes=4)
        
        config = VideoFrameSplitterConfig(
            min_n_frames_per_scene=2,
            max_n_frames_per_scene=4,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(sample_video_path, scene_list=scenes)
        
        if result and result.frames:
            # Visualize scene distribution
            self._visualize_scene_distribution(result, scenes, output_dir / "scene_distribution.png")
            
            # Save frames grouped by scene
            scene_frames = result.get_scene_frames_dict()
            for scene_id, frames in scene_frames.items():
                if scene_id >= 0 and frames:
                    scene_dir = output_dir / f"scene_{scene_id}"
                    scene_dir.mkdir(exist_ok=True)
                    
                    self._save_frame_grid(
                        frames,
                        scene_dir / f"scene_{scene_id}_frames.png",
                        title=f"Scene {scene_id} - {len(frames)} frames"
                    )
    
    def test_frame_processing_visual(self, sample_video_path, test_results_dir):
        """Visualize frame processing with different sizes."""
        output_dir = test_results_dir / "frame_splitter" / "frame_processing"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sizes = [None, 128, 256, 512]
        results = {}
        
        for max_size in sizes:
            config = VideoFrameSplitterConfig(
                frame_max_size=max_size,
                n_frames_max=4,
                show_progress=False
            )
            splitter = VideoFrameSplitter(config)
            result = splitter.extract_frames(sample_video_path)
            
            if result and result.frames:
                size_name = f"size_{max_size}" if max_size else "original"
                results[size_name] = result
                
                # Save sample frame to show size difference
                sample_frame = result.frames[0]
                frame_path = output_dir / f"{size_name}_sample.jpg"
                self._save_frame(sample_frame.image, frame_path)
        
        # Create size comparison
        self._create_size_comparison(results, output_dir / "size_comparison.png")
    
    def test_frame_timeline_visual(self, sample_video_path, test_results_dir):
        """Create timeline visualization of extracted frames."""
        output_dir = test_results_dir / "frame_splitter" / "timeline"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames with different strategies
        strategies = {
            "interval": VideoFrameSplitterConfig(frame_interval=15, n_frames_max=10),
            "random": VideoFrameSplitterConfig(n_random_frames=8, n_sec_max=15.0),
            "time_based": VideoFrameSplitterConfig(frame_interval_sec=1.5, n_sec_max=12.0)
        }
        
        timeline_data = {}
        for name, config in strategies.items():
            config.show_progress = False
            splitter = VideoFrameSplitter(config)
            result = splitter.extract_frames(sample_video_path)
            
            if result and result.frames:
                timestamps = [frame.timestamp for frame in result.frames]
                frame_indices = [frame.idx for frame in result.frames]
                timeline_data[name] = {
                    'timestamps': timestamps,
                    'indices': frame_indices,
                    'frames': result.frames
                }
        
        self._create_timeline_plot(timeline_data, output_dir / "timeline_comparison.png")
    
    def _save_frame(self, frame_image: np.ndarray, output_path: Path):
        """Save a single frame."""
        if len(frame_image.shape) == 3 and frame_image.shape[2] == 3:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_image
        
        cv2.imwrite(str(output_path), frame_bgr)
    
    def _save_frame_grid(self, frames: list, output_path: Path, title: str = "Frames", cols: int = 4):
        """Save frames in a grid layout."""
        n_frames = len(frames)
        rows = (n_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.suptitle(title, fontsize=16)
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(rows * cols):
            ax = axes[i] if rows > 1 or cols > 1 else axes
            
            if i < n_frames:
                frame = frames[i]
                ax.imshow(frame.image)
                ax.set_title(f'Idx: {frame.idx}\nT: {frame.timestamp:.2f}s', fontsize=10)
            else:
                ax.set_visible(False)
            
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_strategy_comparison(self, results: dict, output_path: Path):
        """Create comparison of different selection strategies."""
        n_strategies = len(results)
        fig, axes = plt.subplots(n_strategies, 1, figsize=(15, n_strategies * 3))
        
        if n_strategies == 1:
            axes = [axes]
        
        for i, (strategy_name, result) in enumerate(results.items()):
            ax = axes[i]
            
            # Plot timeline
            timestamps = [frame.timestamp for frame in result.frames]
            indices = [frame.idx for frame in result.frames]
            
            ax.scatter(timestamps, [i] * len(timestamps), alpha=0.7, s=50)
            ax.set_ylabel(strategy_name.title())
            ax.set_ylim(-0.5, 0.5)
            ax.grid(True, alpha=0.3)
            
            # Add frame info
            for j, (ts, idx) in enumerate(zip(timestamps, indices)):
                ax.annotate(f'{idx}', (ts, 0), xytext=(0, 10), 
                           textcoords='offset points', ha='center', fontsize=8)
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle('Frame Selection Strategy Comparison')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_size_comparison(self, results: dict, output_path: Path):
        """Compare frame sizes visually."""
        n_sizes = len(results)
        fig, axes = plt.subplots(1, n_sizes, figsize=(n_sizes * 4, 4))
        
        if n_sizes == 1:
            axes = [axes]
        
        for i, (size_name, result) in enumerate(results.items()):
            ax = axes[i]
            
            if result.frames:
                frame = result.frames[0]
                ax.imshow(frame.image)
                ax.set_title(f'{size_name}\n{frame.shape[1]}x{frame.shape[0]}')
            
            ax.axis('off')
        
        plt.suptitle('Frame Size Comparison')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_scene_distribution(self, result, scenes: list, output_path: Path):
        """Visualize how frames are distributed across scenes."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Scene timeline
        for scene in scenes:
            ax1.barh(scene.scene_id, scene.l_sec, left=scene.start_sec, 
                    alpha=0.3, label=f'Scene {scene.scene_id}')
        
        # Plot extracted frames
        scene_frames = result.get_scene_frames_dict()
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenes)))
        
        for scene_id, frames in scene_frames.items():
            if scene_id >= 0 and frames:
                timestamps = [frame.timestamp for frame in frames]
                y_pos = [scene_id] * len(timestamps)
                ax1.scatter(timestamps, y_pos, c=[colors[scene_id]], s=50, zorder=10)
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Scene ID')
        ax1.set_title('Scene Timeline with Extracted Frames')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Frames per scene
        scene_counts = {}
        for scene_id, frames in scene_frames.items():
            if scene_id >= 0:
                scene_counts[scene_id] = len(frames)
        
        if scene_counts:
            ax2.bar(scene_counts.keys(), scene_counts.values())
            ax2.set_xlabel('Scene ID')
            ax2.set_ylabel('Number of Frames')
            ax2.set_title('Frames Extracted per Scene')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_timeline_plot(self, timeline_data: dict, output_path: Path):
        """Create timeline plot showing frame extraction patterns."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(timeline_data)))
        
        for i, (strategy_name, data) in enumerate(timeline_data.items()):
            timestamps = data['timestamps']
            y_pos = [i] * len(timestamps)
            
            ax.scatter(timestamps, y_pos, c=[colors[i]], label=strategy_name, 
                      s=60, alpha=0.7)
            
            # Add frame indices as text
            for ts, idx in zip(timestamps, data['indices']):
                ax.annotate(str(idx), (ts, i), xytext=(0, 10), 
                           textcoords='offset points', ha='center', fontsize=8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Selection Strategy')
        ax.set_yticks(range(len(timeline_data)))
        ax.set_yticklabels(timeline_data.keys())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Frame Selection Timeline Comparison')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_test_scenes(self, frame_count: int, fps: float, n_scenes: int = 4):
        """Create test scenes for visualization."""
        frames_per_scene = frame_count // n_scenes
        scenes = []
        
        for i in range(n_scenes):
            start_frame = i * frames_per_scene
            end_frame = min((i + 1) * frames_per_scene - 1, frame_count - 1)
            
            scene = SceneData(scene_id=i, start_frame=start_frame, end_frame=end_frame)
            scene.calculate_timing(fps)
            scenes.append(scene)
        
        return scenes