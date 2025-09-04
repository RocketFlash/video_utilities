import os
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your classes (adjust import path as needed)
from .video_frame_splitter import (
    VideoFrameSplitter, 
    VideoFrameSplitterConfig, 
    SceneData, 
    VideoReaderType, 
    FrameSelectionStrategy
)


class VideoFrameSplitterTester:
    """Comprehensive tester for VideoFrameSplitter functionality."""
    
    def __init__(self, video_path: str):
        """Initialize tester with video path."""
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Testing with video: {self.video_path}")
        print(f"Video file size: {self.video_path.stat().st_size / (1024*1024):.1f} MB")
    
    def run_quick_test(self):
        """Run a quick test with basic functionality."""
        print("Running quick test...")
        
        try:
            config = VideoFrameSplitterConfig(
                n_frames_max=5,
                frame_max_size=256
            )
            
            splitter = VideoFrameSplitter(config)
            result = splitter.extract_frames(self.video_path)
            
            if result:
                print(f"✅ Quick test passed!")
                print(f"   Extracted {result.total_frames} frames")
                print(f"   Video: {result.video_len_orig:.1f}s, {result.fps:.1f} FPS")
                print(f"   Frame size: {result.frame_shape}")
                return True
            else:
                print("❌ Quick test failed!")
                return False
        except Exception as e:
            print(f"❌ Quick test failed with error: {e}")
            return False
    
    def test_basic_functionality(self):
        """Test basic frame extraction with default settings."""
        print("\n" + "="*60)
        print("TEST 1: Basic Functionality")
        print("="*60)
        
        config = VideoFrameSplitterConfig(
            frame_max_size=224,
            n_frames_max=10,
            show_progress=True
        )
        splitter = VideoFrameSplitter(config)
        
        start_time = time.time()
        result = splitter.extract_frames(self.video_path)
        elapsed = time.time() - start_time
        
        if result:
            print(f"✅ Success! Extracted {result.total_frames} frames in {elapsed:.2f}s")
            print(f"   Original video: {result.video_len_orig:.1f}s, {result.n_frames_orig} frames")
            print(f"   Frame size: {result.frame_shape}")
            print(f"   FPS: {result.fps:.2f}")
            print(f"   Selection strategy: {result.selection_strategy}")
            
            # Show first frame info
            if result.frames:
                first_frame = result.frames[0]
                print(f"   First frame: idx={first_frame.idx}, timestamp={first_frame.timestamp}s")
                print(f"   Frame shape: {first_frame.shape}")
        else:
            print("❌ Failed to extract frames")
        
        return result
    
    def test_different_readers(self):
        """Test different video readers."""
        print("\n" + "="*60)
        print("TEST 2: Different Video Readers")
        print("="*60)
        
        readers = [VideoReaderType.OPENCV, VideoReaderType.AUTO]
        if hasattr(VideoReaderType, 'DECORD'):
            readers.insert(0, VideoReaderType.DECORD)
        
        results = {}
        for reader_type in readers:
            print(f"\nTesting {reader_type.value} reader...")
            
            config = VideoFrameSplitterConfig(
                video_reader_type=reader_type,
                n_frames_max=5,
                show_progress=False
            )
            splitter = VideoFrameSplitter(config)
            
            start_time = time.time()
            result = splitter.extract_frames(self.video_path)
            elapsed = time.time() - start_time
            
            if result:
                print(f"   ✅ Success: {result.total_frames} frames in {elapsed:.2f}s")
                results[reader_type.value] = {
                    'result': result,
                    'time': elapsed,
                    'frames': result.total_frames
                }
            else:
                print(f"   ❌ Failed with {reader_type.value}")
        
        # Compare performance
        if len(results) > 1:
            print(f"\nPerformance comparison:")
            for reader, data in results.items():
                print(f"   {reader}: {data['time']:.3f}s for {data['frames']} frames")
        
        return results
    
    def test_selection_strategies(self):
        """Test different frame selection strategies."""
        print("\n" + "="*60)
        print("TEST 3: Selection Strategies")
        print("="*60)
        
        results = {}
        
        # 1. Interval-based selection
        print("\n3.1 Interval-based selection:")
        config = VideoFrameSplitterConfig(
            frame_interval=10,  # Every 10th frame
            n_frames_max=20,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(self.video_path)
        if result:
            print(f"   ✅ Interval: {result.total_frames} frames, interval={result.frame_interval}")
            results['interval'] = result
        
        # 2. Time-based interval
        print("\n3.2 Time-based interval selection:")
        config = VideoFrameSplitterConfig(
            frame_interval_sec=2.0,  # Every 2 seconds
            n_sec_max=20.0,          # First 20 seconds
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(self.video_path)
        if result:
            print(f"   ✅ Time-based: {result.total_frames} frames, interval={result.frame_interval_sec:.2f}s")
            results['time_interval'] = result
        
        # 3. Random selection
        print("\n3.3 Random selection:")
        config = VideoFrameSplitterConfig(
            n_random_frames=8,
            start_sec=10.0,
            n_sec_max=30.0,
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(self.video_path)
        if result:
            print(f"   ✅ Random: {result.total_frames} frames")
            frame_times = [f.timestamp for f in result.frames]
            print(f"   Timestamps: {frame_times}")
            results['random'] = result
        
        # 4. Manual selection
        print("\n3.4 Manual frame selection:")
        config = VideoFrameSplitterConfig(show_progress=False)
        splitter = VideoFrameSplitter(config)
        
        # Select specific frame indices
        manual_indices = [0, 100, 200, 300, 500]
        result = splitter.extract_frames(
            self.video_path,
            selected_frame_idxs=manual_indices
        )
        if result:
            print(f"   ✅ Manual indices: {result.total_frames} frames")
            print(f"   Selected indices: {result.selected_frame_idxs}")
            results['manual_indices'] = result
        
        # Select by timestamps
        manual_seconds = [5.0, 10.5, 15.2, 20.8]
        result = splitter.extract_frames(
            self.video_path,
            selected_seconds=manual_seconds
        )
        if result:
            print(f"   ✅ Manual timestamps: {result.total_frames} frames")
            actual_times = [f.timestamp for f in result.frames]
            print(f"   Requested: {manual_seconds}")
            print(f"   Actual: {actual_times}")
            results['manual_seconds'] = result
        
        return results
    
    def create_mock_scenes(self, fps: float = 30.0, total_frames: int = 3000):
        """Create mock scenes with proper timing calculation."""
        # Make scenes that fit within the actual video
        quarter = total_frames // 4
        scenes = [
            SceneData(scene_id=0, start_frame=0, end_frame=quarter-1),
            SceneData(scene_id=1, start_frame=quarter, end_frame=2*quarter-1),
            SceneData(scene_id=2, start_frame=2*quarter, end_frame=3*quarter-1),
            SceneData(scene_id=3, start_frame=3*quarter, end_frame=total_frames-1),
        ]
        # Calculate timing for each scene
        for scene in scenes:
            scene.calculate_timing(fps)
        return scenes
    
    def test_scene_based_selection(self):
        """Test scene-based frame selection."""
        print("\n" + "="*60)
        print("TEST 4: Scene-based Selection")
        print("="*60)
        
        # First get video info to create appropriate scenes
        config = VideoFrameSplitterConfig(n_frames_max=1, show_progress=False)
        splitter = VideoFrameSplitter(config)
        temp_result = splitter.extract_frames(self.video_path)
        
        if not temp_result:
            print("❌ Could not get video info for scene testing")
            return {}
        
        fps = temp_result.fps
        total_frames = temp_result.n_frames_orig
        
        # Create scenes that fit within the actual video
        scenes = self.create_mock_scenes(fps=30.0, total_frames=3000)
        
        # Calculate timing for each scene
        for scene in scenes:
            scene.calculate_timing(fps)
        
        print(f"Created {len(scenes)} mock scenes for video with {total_frames} frames:")
        for scene in scenes:
            print(f"   {scene}")
        
        # Test with different scene selection parameters
        configs = [
            ("Default scene params", VideoFrameSplitterConfig(show_progress=False)),
            ("More frames per scene", VideoFrameSplitterConfig(
                min_n_frames_per_scene=5,
                max_n_frames_per_scene=15,
                show_progress=False
            )),
            ("Fewer frames per scene", VideoFrameSplitterConfig(
                min_n_frames_per_scene=2,
                max_n_frames_per_scene=8,
                scene_length_threshold=30.0,
                show_progress=False
            ))
        ]
        
        results = {}
        for config_name, config in configs:
            print(f"\n4.{len(results)+1} {config_name}:")
            splitter = VideoFrameSplitter(config)
            result = splitter.extract_frames(self.video_path, scene_list=scenes)
            
            if result:
                print(f"   ✅ Extracted {result.total_frames} frames from {len(scenes)} scenes")
                
                # Group frames by scene
                scene_frames = result.get_scene_frames_dict()
                for scene_id, frames in scene_frames.items():
                    if scene_id >= 0:  # Valid scene ID
                        timestamps = [f.timestamp for f in frames]
                        print(f"   Scene {scene_id}: {len(frames)} frames, "
                              f"times: {timestamps[0]:.1f}s - {timestamps[-1]:.1f}s")
                
                results[config_name] = result
            else:
                print(f"   ❌ Failed")
        
        return results
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n" + "="*60)
        print("TEST 5: Edge Cases and Error Handling")
        print("="*60)
        
        splitter = VideoFrameSplitter(VideoFrameSplitterConfig(show_progress=False))
        
        # Test 1: Invalid video path
        print("\n5.1 Invalid video path:")
        result = splitter.extract_frames("nonexistent_video.mp4")
        print(f"   Result: {'❌ Correctly failed' if result is None else '⚠️ Unexpected success'}")
        
        # Test 2: Empty frame selection
        print("\n5.2 Empty frame selection:")
        result = splitter.extract_frames(self.video_path, selected_frame_idxs=[])
        print(f"   Result: {'❌ Correctly failed' if result is None else '⚠️ Unexpected success'}")
        
        # Test 3: Out of bounds frame indices
        print("\n5.3 Out of bounds frame indices:")
        result = splitter.extract_frames(self.video_path, selected_frame_idxs=[999999])
        print(f"   Result: {'❌ Correctly failed' if result is None else f'✅ Handled gracefully: {result.total_frames} frames'}")
        
        # Test 4: Invalid configuration
        print("\n5.4 Invalid configuration:")
        try:
            invalid_config = VideoFrameSplitterConfig(
                start_idx=-1,  # Should be non-negative
                frame_interval=0  # Should be at least 1
            )
            print("   ⚠️ Configuration validation failed to catch invalid values")
        except ValueError as e:
            print(f"   ✅ Correctly caught invalid config: {e}")
        
        # Test 5: Very large frame requests
        print("\n5.5 Large frame request:")
        config = VideoFrameSplitterConfig(
            n_frames_max=999999,  # Request more frames than exist
            show_progress=False
        )
        splitter = VideoFrameSplitter(config)
        result = splitter.extract_frames(self.video_path)
        if result:
            print(f"   ✅ Handled gracefully: got {result.total_frames} frames instead of 999999")
        else:
            print("   ❌ Failed unexpectedly")
    
    def test_frame_processing(self):
        """Test frame processing options."""
        print("\n" + "="*60)
        print("TEST 6: Frame Processing Options")
        print("="*60)
        
        # Test different frame sizes
        sizes = [None, 128, 256, 512, 1024]
        results = {}
        
        for max_size in sizes:
            print(f"\n6.{len(results)+1} Max frame size: {max_size}")
            
            config = VideoFrameSplitterConfig(
                frame_max_size=max_size,
                n_frames_max=3,
                show_progress=False
            )
            splitter = VideoFrameSplitter(config)
            result = splitter.extract_frames(self.video_path)
            
            if result:
                frame_shape = result.frames[0].shape
                print(f"   ✅ Frame shape: {frame_shape}")
                print(f"   Original: {result.frame_w_orig}x{result.frame_h_orig}")
                print(f"   Processed: {result.frame_w}x{result.frame_h}")
                results[f"size_{max_size}"] = result
        
        return results
    
    def visualize_results(self, results_dict: dict, max_frames_per_result: int = 4):
        """Visualize frame extraction results."""
        print("\n" + "="*60)
        print("VISUALIZATION: Sample Frames")
        print("="*60)
        
        for result_name, result in results_dict.items():
            if result is None or not result.frames:
                continue
            
            print(f"\nVisualizing: {result_name}")
            
            # Select frames to display
            n_frames = min(len(result.frames), max_frames_per_result)
            frame_indices = np.linspace(0, len(result.frames)-1, n_frames, dtype=int)
            
            fig, axes = plt.subplots(1, n_frames, figsize=(4*n_frames, 4))
            if n_frames == 1:
                axes = [axes]
            
            for i, frame_idx in enumerate(frame_indices):
                frame = result.frames[frame_idx]
                axes[i].imshow(frame.image)
                axes[i].set_title(f'Frame {frame.idx}\n{frame.timestamp:.1f}s')
                axes[i].axis('off')
            
            plt.suptitle(f'{result_name} - {result.total_frames} frames total')
            plt.tight_layout()
            plt.show()
    
    def benchmark_performance(self, n_runs: int = 3):
        """Benchmark performance with different configurations."""
        print("\n" + "="*60)
        print("BENCHMARK: Performance Testing")
        print("="*60)
        
        benchmark_configs = [
            ("Small frames, few", VideoFrameSplitterConfig(
                frame_max_size=128, n_frames_max=10, show_progress=False
            )),
            ("Medium frames, medium", VideoFrameSplitterConfig(
                frame_max_size=256, n_frames_max=20, show_progress=False
            )),
            ("Large frames, many", VideoFrameSplitterConfig(
                frame_max_size=512, n_frames_max=50, show_progress=False
            )),
        ]
        
        results = {}
        for config_name, config in benchmark_configs:
            print(f"\nBenchmarking: {config_name}")
            times = []
            
            for run in range(n_runs):
                splitter = VideoFrameSplitter(config)
                start_time = time.time()
                result = splitter.extract_frames(self.video_path)
                elapsed = time.time() - start_time
                
                if result:
                    times.append(elapsed)
                    print(f"   Run {run+1}: {elapsed:.3f}s ({result.total_frames} frames)")
                else:
                    print(f"   Run {run+1}: Failed")
            
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                print(f"   Average: {avg_time:.3f}±{std_time:.3f}s")
                results[config_name] = {
                    'times': times,
                    'avg': avg_time,
                    'std': std_time
                }
        
        return results
    
    def run_all_tests(self, visualize: bool = False, benchmark: bool = False):
        """Run all tests."""
        print("🚀 Starting comprehensive VideoFrameSplitter testing...")
        
        all_results = {}
        
        # Run basic tests
        all_results['basic'] = self.test_basic_functionality()
        all_results['readers'] = self.test_different_readers()
        all_results['strategies'] = self.test_selection_strategies()
        all_results['scenes'] = self.test_scene_based_selection()
        all_results['processing'] = self.test_frame_processing()
        
        # Run edge cases
        self.test_edge_cases()
        
        # Optional visualization
        if visualize:
            # Visualize some interesting results
            viz_results = {
                'Basic': all_results['basic'],
                'Random': all_results['strategies'].get('random'),
                'Scene-based': next(iter(all_results['scenes'].values()), None)
            }
            self.visualize_results(viz_results)
        
        # Optional benchmarking
        if benchmark:
            all_results['benchmark'] = self.benchmark_performance()
        
        print("\n" + "="*60)
        print("🎉 All tests completed!")
        print("="*60)
        
        return all_results


if __name__ == "__main__":
    VIDEO_PATH = "path/to/your/video.mp4"
    
    # Create tester instance
    tester = VideoFrameSplitterTester(VIDEO_PATH)
    
    # Quick test
    if tester.run_quick_test():
        # Full test suite
        results = tester.run_all_tests(
            visualize=False,
            benchmark=True
        )