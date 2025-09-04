import cv2
import json
import numpy as np
from IPython.display import display
from PIL import Image
import ipywidgets as widgets
from ..video_frame_splitter import VideoFrame
from ..video_captioner import VideoFrameOutputResult
from typing import Union, Optional, List, Dict, Any
import math
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
from .utils import create_video_from_frames


class FrameVisualizer:
    def __init__(
        self, 
        frame_max_size: Optional[int] = None,
        widget_width: Optional[int] = None,
        widget_height: Optional[int] = None,
        vlm_predictor=None
    ):
        """
        Initialize the visualizer with configuration.
        
        Args:
            frame_max_size: Maximum size for image processing (affects image quality)
            widget_width: Width of the widget in the notebook interface (affects display size)
            widget_height: Height of the widget in the notebook interface (affects display size)
            vlm_predictor: Optional predictor function
        """
        self.frame_max_size = frame_max_size
        self.widget_width = widget_width or 600   # Default widget width
        self.widget_height = widget_height or 400  # Default widget height
        self.vlm_predictor = vlm_predictor
        
        # These will be set per visualization call
        self.frames_list = None
        self.frame_results_list = None
        self.is_video_frame_format = None
        self.frame_height = None
        self.frame_width = None
        self.resize_scale = None
        self.display_height = None
        self.display_width = None

    def set_widget_size(self, width: int, height: int):
        """
        Set the widget display size in the notebook.
        
        Args:
            width: Widget width in pixels
            height: Widget height in pixels
        """
        self.widget_width = width
        self.widget_height = height

    def _create_browser_widgets(self):
        """Create widgets for frame browser interface."""
        widgets_dict = {
            'frame_slider': widgets.IntSlider(
                value=0,
                min=0,
                max=max(0, len(self.frames_list) - 1),
                step=1,
                continuous_update=False,
                description='Frame'
            ),
            'image_widget': widgets.Image(
                format='jpeg',
                width=self.widget_width,   # Widget display size
                height=self.widget_height  # Widget display size
            )
        }
        
        # Add VideoFrame info widgets
        if self.is_video_frame_format:
            widgets_dict.update({
                'timestamp_label': widgets.Label(value="Timestamp: --"),
                'frame_idx_label': widgets.Label(value="Frame idx: --"),
                'scene_id_label': widgets.Label(value="Scene id: --")
            })
        
        # Add prediction widgets
        if self.vlm_predictor is not None or self.frame_results_list is not None:
            widgets_dict['caption_widget'] = widgets.Textarea(
                value='',
                description='Prediction:',
                layout=widgets.Layout(width='400px', height=f'{self.widget_height}px')
            )
            
            if self.vlm_predictor is not None:
                widgets_dict['caption_button'] = widgets.Button(description="Get Prediction")
        
        return widgets_dict
    
    def browse(
        self, 
        frames: Union[np.ndarray, Image.Image, VideoFrame, List[Any]],
        frame_results: Optional[Union[List[Any], Any]] = None,
        widget_size: Optional[tuple] = None
    ) -> None:
        """
        Display interactive frame browser with slider controls.
        
        Args:
            frames: Single frame or list of frames to visualize
            frame_results: Optional pre-computed results/captions for frames
            widget_size: Optional (width, height) tuple for widget display size
        """
        # Temporarily override widget size if provided
        if widget_size is not None:
            original_width, original_height = self.widget_width, self.widget_height
            self.widget_width, self.widget_height = widget_size
        
        try:
            self._prepare_frames(frames, frame_results)
            
            widgets_dict = self._create_browser_widgets()
            self._setup_browser_interactions(widgets_dict)
            self._display_browser_interface(widgets_dict)
            
            # Initialize display
            self._update_frame_display(widgets_dict, 0)
        finally:
            # Restore original settings if they were overridden
            if widget_size is not None:
                self.widget_width, self.widget_height = original_width, original_height
  
    def play(
        self, 
        frames: Union[np.ndarray, Image.Image, VideoFrame, List[Any]],
        fps: int = 25,
        image_format: str = 'JPEG',
        stats: Optional[Dict[str, np.ndarray]] = None,
        stats_fig_size: tuple = (8, 2),
        show_frame_info: bool = False,
        loop_playback: bool = True,
        auto_play: bool = False,
        frame_results: Optional[Union[List[Any], Any]] = None,
        widget_size: Optional[tuple] = None
    ) -> None:
        """
        Display video-like playback interface with play controls.
        
        Args:
            frames: Single frame or list of frames to visualize
            fps: Frames per second for playback
            image_format: Image encoding format ('JPEG' or 'PNG')
            stats: Optional statistics to plot alongside video
            stats_fig_size: Size of statistics plots
            show_frame_info: Show frame metadata if available
            loop_playback: Enable looping playback
            auto_play: Start playing immediately
            frame_results: Optional pre-computed results/captions for frames
            widget_size: Optional (width, height) tuple for widget display size
        """
        # Temporarily override widget size if provided
        if widget_size is not None:
            original_width, original_height = self.widget_width, self.widget_height
            self.widget_width, self.widget_height = widget_size
        
        try:
            self._prepare_frames(frames, frame_results)
            
            # Pre-process frames for efficient playback
            processed_frames = self._process_frames_for_video(image_format)
            
            # Create widgets
            widgets_dict = self._create_video_widgets(processed_frames, fps, show_frame_info, loop_playback)
            
            # Create statistics plots if provided
            plot_widgets, joint_plots = self._create_stats_plots(stats, stats_fig_size)
            
            # Set up interactions
            self._setup_video_interactions(widgets_dict, processed_frames, joint_plots)
            
            # Display interface
            self._display_video_interface(widgets_dict, plot_widgets)
            
            # Auto-play if requested
            if auto_play:
                widgets_dict['play'].value = 1
            
            # Initialize display
            self._update_video_display(widgets_dict, joint_plots, 0)
        finally:
            # Restore original settings if they were overridden
            if widget_size is not None:
                self.widget_width, self.widget_height = original_width, original_height
    
    def browse_scenes(self,
                    scene_frames_dict: Dict[int, List[Any]],
                    grid_canvas_w: int = 512,
                    grid_num_cols: int = 3,
                    scene_captions: Optional[List[str]] = None,
                    widget_size: Optional[tuple] = None) -> None:
        """
        Display scene-based visualization with grid layouts.
        
        Args:
            scene_frames_dict: Dictionary mapping scene IDs to frame lists
            grid_canvas_w: Width of the grid canvas for image processing (affects quality)
            grid_num_cols: Number of columns in grid
            scene_captions: Optional captions for each scene
            widget_size: Optional (width, height) tuple for widget display size
        """
        if not scene_frames_dict:
            print("No scenes to visualize")
            return
        
        # Temporarily override widget size if provided
        if widget_size is not None:
            original_width, original_height = self.widget_width, self.widget_height
            self.widget_width, self.widget_height = widget_size
        
        try:
            # Prepare frames from first scene for format detection
            first_scene_frames = list(scene_frames_dict.values())[0]
            self._prepare_frames(first_scene_frames)
            
            widgets_dict = self._create_scene_widgets(scene_frames_dict, grid_canvas_w, scene_captions)
            self._setup_scene_interactions(widgets_dict, scene_frames_dict, grid_canvas_w, grid_num_cols, scene_captions)
            self._display_scene_interface(widgets_dict)
            
            # Initialize display
            self._update_scene_display(widgets_dict, scene_frames_dict, grid_canvas_w, grid_num_cols, scene_captions, 0)
        finally:
            # Restore original settings if they were overridden
            if widget_size is not None:
                self.widget_width, self.widget_height = original_width, original_height

    def set_vlm_predictor(self, vlm_predictor):
        """Update the VLM predictor."""
        self.vlm_predictor = vlm_predictor
    
    def set_frame_max_size(self, frame_max_size: Optional[int]):
        """Update the maximum frame size for display."""
        self.frame_max_size = frame_max_size
    
    def _prepare_frames(self, frames, frame_results=None):
        """Prepare frames and results for visualization."""
        self.frames_list = self._normalize_frames_input(frames)
        self.frame_results_list = self._normalize_results_input(frame_results, len(self.frames_list))
        
        if len(self.frames_list) == 0:
            raise ValueError("No frames provided")
        
        # Analyze frame format and dimensions
        self._analyze_frames()
        
        # Calculate resize parameters
        self.resize_scale = self._calculate_resize_scale()
        if self.resize_scale != 1:
            self.display_height = int(self.frame_height * self.resize_scale)
            self.display_width = int(self.frame_width * self.resize_scale)
        else:
            self.display_height = self.frame_height
            self.display_width = self.frame_width
    
    def _normalize_frames_input(self, frames):
        """Convert single frame or list of frames to consistent list format."""
        return frames if isinstance(frames, list) else [frames]
    
    def _normalize_results_input(self, frame_results, n_frames):
        """Convert frame_results to consistent list format."""
        if frame_results is None:
            return None
        if isinstance(frame_results, list):
            return frame_results
        return [frame_results] + [None] * (n_frames - 1)
    
    def _analyze_frames(self):
        """Analyze frame format and extract dimensions."""
        frame = self.frames_list[0]
        
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 3:
                self.frame_height, self.frame_width, _ = frame.shape
            else:
                self.frame_height, self.frame_width = frame.shape
            self.is_video_frame_format = False
        elif isinstance(frame, Image.Image):
            self.frame_width, self.frame_height = frame.size
            self.is_video_frame_format = False
        elif hasattr(frame, 'image'):  # VideoFrame
            if isinstance(frame.image, np.ndarray):
                if len(frame.image.shape) == 3:
                    self.frame_height, self.frame_width, _ = frame.image.shape
                else:
                    self.frame_height, self.frame_width = frame.image.shape
            else:  # PIL Image
                self.frame_width, self.frame_height = frame.image.size
            self.is_video_frame_format = True
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")
    
    def _calculate_resize_scale(self):
        """Calculate resize scale based on max size constraint."""
        if self.frame_max_size is None:
            return 1
        max_dim = max(self.frame_height, self.frame_width)
        return min(1, self.frame_max_size / max_dim)
     
    def _setup_browser_interactions(self, widgets_dict):
        """Set up event handlers for browser interface."""
        def on_slider_change(change):
            self._update_frame_display(widgets_dict, change['new'])
        
        def on_prediction_click(b):
            if self.vlm_predictor is not None:
                frame_idx = widgets_dict['frame_slider'].value
                frame = self.frames_list[frame_idx]
                
                # Get actual image data
                image_data = frame.image if self.is_video_frame_format else frame
                if isinstance(image_data, Image.Image):
                    image_data = np.array(image_data)
                
                outputs = self.vlm_predictor(image_data)
                widgets_dict['caption_widget'].value = str(outputs)
        
        widgets_dict['frame_slider'].observe(on_slider_change, names='value')
        if 'caption_button' in widgets_dict:
            widgets_dict['caption_button'].on_click(on_prediction_click)
    
    def _display_browser_interface(self, widgets_dict):
        """Display the browser interface layout."""
        output_widgets = [widgets_dict['image_widget']]
        if 'caption_widget' in widgets_dict:
            output_widgets.append(widgets_dict['caption_widget'])
        
        interface_widgets = [widgets_dict['frame_slider']]
        if 'caption_button' in widgets_dict:
            interface_widgets.append(widgets_dict['caption_button'])
        
        info_widgets = []
        if self.is_video_frame_format:
            info_widgets = [
                widgets_dict['timestamp_label'],
                widgets_dict['frame_idx_label'],
                widgets_dict['scene_id_label']
            ]
        
        display(widgets.VBox([
            widgets.HBox(output_widgets),
            widgets.VBox(info_widgets),
            widgets.HBox(interface_widgets)
        ]))
    
    def _update_frame_display(self, widgets_dict, frame_idx):
        """Update frame display for browser interface."""
        frame = self.frames_list[frame_idx]
        
        # Update VideoFrame info
        if self.is_video_frame_format:
            widgets_dict['timestamp_label'].value = f"Timestamp: {frame.timestamp:.2f}"
            widgets_dict['frame_idx_label'].value = f"Frame idx: {frame.idx}"
            scene_id_str = f"Scene id: {frame.scene_id}" if frame.scene_id is not None else "Scene id: --"
            widgets_dict['scene_id_label'].value = scene_id_str
            display_frame = frame.image
        else:
            display_frame = frame
        
        # Convert and resize for display
        display_image = self._prepare_display_image(display_frame)
        widgets_dict['image_widget'].value = display_image._repr_jpeg_()
        
        # Update caption
        if self.frame_results_list and 'caption_widget' in widgets_dict:
            self._update_caption_widget(widgets_dict['caption_widget'], frame_idx)
    
    def _process_frames_for_video(self, image_format):
        """Pre-process and encode frames for video playback."""
        processed_frames = []
        
        for frame in self.frames_list:
            # Extract image data
            image_data = frame.image if self.is_video_frame_format else frame
            
            # Convert to PIL and resize
            display_image = self._prepare_display_image(image_data)
            
            # Encode to bytes
            with BytesIO() as buf:
                if display_image.mode == 'L' and image_format.upper() == 'JPEG':
                    display_image = display_image.convert('RGB')
                display_image.save(buf, format=image_format.upper())
                processed_frames.append(buf.getvalue())
        
        return processed_frames
    
    def _create_video_widgets(self, processed_frames, fps, show_frame_info, loop_playback):
        """Create widgets for video playback interface."""
        max_frame_idx = max(0, len(processed_frames) - 1)
        
        widgets_dict = {
            'play': widgets.Play(
                value=0,
                min=0,
                max=max_frame_idx,
                step=1,
                interval=max(1, int(1000 / fps)),
                description="Play",
                disabled=False,
                repeat=loop_playback
            ),
            'slider': widgets.IntSlider(
                value=0,
                min=0,
                max=max_frame_idx,
                description="Frame"
            ),
            'image_widget': widgets.Image(
                value=processed_frames[0] if processed_frames else b'',
                width=self.widget_width,
                height=self.widget_height
            )
        }
        
        if show_frame_info:
            widgets_dict['frame_info'] = widgets.HTML(value="Frame: 0")
        
        return widgets_dict
    
    def _setup_video_interactions(self, widgets_dict, processed_frames, joint_plots):
        """Set up video playback interactions."""
        def sync_slider(change):
            widgets_dict['slider'].value = widgets_dict['play'].value
        
        def update_display(change):
            frame_idx = change['new']
            if 0 <= frame_idx < len(processed_frames):
                widgets_dict['image_widget'].value = processed_frames[frame_idx]
            self._update_video_display(widgets_dict, joint_plots, frame_idx)
        
        widgets_dict['play'].observe(sync_slider, names='value')
        widgets_dict['slider'].observe(update_display, names='value')

    def _update_video_display(self, widgets_dict, joint_plots, frame_idx):
        """Update video display and associated plots."""
        # Update frame info
        if 'frame_info' in widgets_dict:
            frame = self.frames_list[frame_idx]
            if hasattr(frame, 'timestamp') and hasattr(frame, 'idx'):
                info_text = f"Frame: {frame.idx}, Time: {frame.timestamp:.2f}s"
            else:
                info_text = f"Frame: {frame_idx}/{len(self.frames_list)-1}"
            widgets_dict['frame_info'].value = info_text
        
        # Update plots
        for fig, vline in joint_plots:
            try:
                vline.set_xdata([frame_idx, frame_idx])
                fig.canvas.draw_idle()
            except Exception:
                pass
    
    def _create_scene_widgets(self, scene_frames_dict, grid_canvas_w, scene_captions):
        """Create widgets for scene visualization."""
        n_scenes = len(scene_frames_dict)
        scene_keys = list(scene_frames_dict.keys())
        
        # Use widget_width for display size instead of grid_canvas_w
        display_width = min(self.widget_width, grid_canvas_w)  # Use smaller of the two
        
        widgets_dict = {
            'scene_slider': widgets.IntSlider(
                value=0,
                min=0,
                max=n_scenes - 1,
                step=1,
                continuous_update=False,
                description='Scene'
            ),
            'image_widget': widgets.Image(
                format='jpeg', 
                width=display_width,      # Widget display size
                height=self.widget_height # Widget display size
            ),
            'scene_keys': scene_keys,
            'grid_canvas_w': grid_canvas_w  # Keep original canvas size for image processing
        }
        
        # Add info widgets if using VideoFrame format
        example_frame = list(scene_frames_dict.values())[0][0]
        if hasattr(example_frame, 'image'):
            widgets_dict.update({
                'timestamp_label': widgets.Label(value="Timestamp range: --"),
                'frame_idx_label': widgets.Label(value="Frame idx range: --"),
                'scene_id_label': widgets.Label(value="Scene id: --")
            })
        
        if scene_captions:
            widgets_dict['scene_caption_label'] = widgets.Label(value="Scene caption: --")
        
        # Add VLM prediction if available
        if self.vlm_predictor:
            widgets_dict['caption_widget'] = widgets.Textarea(
                value='',
                description='Prediction:',
                layout=widgets.Layout(width='400px', height=f'{self.widget_height}px')
            )
            widgets_dict['caption_button'] = widgets.Button(description="Get Prediction")
        
        return widgets_dict
    
    def _setup_scene_interactions(self, widgets_dict, scene_frames_dict, grid_canvas_w, grid_num_cols, scene_captions):
        """Set up scene visualization interactions."""
        def on_scene_change(change):
            self._update_scene_display(widgets_dict, scene_frames_dict, grid_canvas_w, grid_num_cols, scene_captions, change['new'])
        
        widgets_dict['scene_slider'].observe(on_scene_change, names='value')
        
        if 'caption_button' in widgets_dict:
            def on_prediction_click(b):
                scene_idx = widgets_dict['scene_slider'].value
                scene_key = widgets_dict['scene_keys'][scene_idx]
                scene_frames = scene_frames_dict[scene_key]
                
                # Extract images
                scene_images = []
                for frame in scene_frames:
                    if hasattr(frame, 'image'):
                        scene_images.append(frame.image)
                    else:
                        scene_images.append(frame)
                
                outputs = self.vlm_predictor(scene_images)
                widgets_dict['caption_widget'].value = str(outputs)
            
            widgets_dict['caption_button'].on_click(on_prediction_click)
    
    def _display_scene_interface(self, widgets_dict):
        """Display scene visualization interface."""
        controls = [widgets_dict['scene_slider']]
        if 'caption_button' in widgets_dict:
            controls.append(widgets_dict['caption_button'])
        
        info_widgets = []
        for key in ['scene_caption_label', 'timestamp_label', 'frame_idx_label', 'scene_id_label']:
            if key in widgets_dict:
                info_widgets.append(widgets_dict[key])
        
        main_widgets = [widgets_dict['image_widget']]
        if info_widgets:
            main_widgets.append(widgets.VBox(info_widgets))
        main_widgets.append(widgets.HBox(controls))
        
        if 'caption_widget' in widgets_dict:
            display(widgets.HBox([
                widgets.VBox(main_widgets),
                widgets_dict['caption_widget']
            ]))
        else:
            display(widgets.VBox(main_widgets))
    
    def _update_scene_display(self, widgets_dict, scene_frames_dict, grid_canvas_w, grid_num_cols, scene_captions, scene_idx):
        """Update scene display."""
        scene_key = widgets_dict['scene_keys'][scene_idx]
        scene_frames = scene_frames_dict[scene_key]
        
        # Update captions
        if scene_captions and 'scene_caption_label' in widgets_dict:
            widgets_dict['scene_caption_label'].value = f"Scene caption: {scene_captions[scene_idx]}"
        
        # Extract images and create grid
        scene_images = []
        for frame in scene_frames:
            if hasattr(frame, 'image'):
                scene_images.append(np.array(frame.image) if isinstance(frame.image, Image.Image) else frame.image)
            else:
                scene_images.append(np.array(frame) if isinstance(frame, Image.Image) else frame)
        
        if scene_images:
            # Use the original grid_canvas_w for image processing (quality)
            grid_canvas = self._create_image_grid(scene_images, grid_canvas_w, grid_num_cols)
            grid_image = Image.fromarray(grid_canvas)
            
            # The widget will automatically scale this to fit the widget display size
            widgets_dict['image_widget'].value = grid_image._repr_jpeg_()
            
            # Update frame info for VideoFrame format
            if hasattr(scene_frames[0], 'image') and 'timestamp_label' in widgets_dict:
                start_frame, end_frame = scene_frames[0], scene_frames[-1]
                widgets_dict['timestamp_label'].value = f"Timestamp range: [{start_frame.timestamp:.2f} - {end_frame.timestamp:.2f}]"
                widgets_dict['frame_idx_label'].value = f"Frame idx range: [{start_frame.idx} - {end_frame.idx}]"
                widgets_dict['scene_id_label'].value = f"Scene id: {start_frame.scene_id}"

    def _prepare_display_image(self, frame):
        """Convert frame to PIL Image and resize if needed."""
        # Convert to PIL
        if isinstance(frame, np.ndarray):
            pil_img = Image.fromarray(frame, mode='L' if len(frame.shape) == 2 else None)
        elif isinstance(frame, Image.Image):
            pil_img = frame.copy()
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")
        
        # Resize if needed
        if self.resize_scale != 1:
            new_size = (int(pil_img.width * self.resize_scale), int(pil_img.height * self.resize_scale))
            pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        return pil_img
    
    def _update_caption_widget(self, caption_widget, frame_idx):
        """Update caption widget with frame results."""
        if (frame_idx < len(self.frame_results_list) and 
            self.frame_results_list[frame_idx] is not None):
            result = self.frame_results_list[frame_idx]
            
            if isinstance(result, dict):
                caption_widget.value = json.dumps(result, indent=4)
            elif isinstance(result, str):
                caption_widget.value = result
            elif hasattr(result, 'outputs') and result.outputs is not None:
                caption_widget.value = json.dumps(result.outputs, indent=4)
            else:
                caption_widget.value = str(result)
        else:
            caption_widget.value = ""
    
    def _create_stats_plots(self, stats, stats_fig_size):
        """Create statistics plots for video interface."""
        if stats is None:
            return widgets.VBox([]), []
        
        plt.close('all')
        plt.ioff()
        
        joint_plots = []
        plot_outputs = []
        
        try:
            for stat_name, stat_values in stats.items():
                if not isinstance(stat_values, np.ndarray):
                    stat_values = np.array(stat_values)
                
                fig, ax = plt.subplots(figsize=stats_fig_size)
                fig.canvas.header_visible = False
                ax.set_title(stat_name)
                
                # Handle different data shapes
                if len(stat_values.shape) == 2 and stat_values.shape[1] == 3:
                    colors = ['red', 'green', 'blue']
                    labels = ['x', 'y', 'z']
                    for i in range(3):
                        ax.plot(stat_values[:, i], color=colors[i], label=labels[i])
                    ax.legend()
                elif len(stat_values.shape) == 2 and stat_values.shape[1] > 1:
                    for i in range(min(stat_values.shape[1], 5)):
                        ax.plot(stat_values[:, i], label=f'dim_{i}')
                    if stat_values.shape[1] <= 5:
                        ax.legend()
                else:
                    ax.plot(stat_values.flatten())
                
                vline = ax.axvline(x=0, color='black', linewidth=2, alpha=0.7)
                ax.set_xlim(0, max(1, len(stat_values) - 1))
                plt.tight_layout()
                
                joint_plots.append((fig, vline))
                
                out = widgets.Output()
                with out:
                    display(fig.canvas)
                plot_outputs.append(out)
        
        except Exception as e:
            print(f"Error creating stats plots: {e}")
            joint_plots = []
            plot_outputs = []
        
        plt.ion()
        return widgets.VBox(plot_outputs) if plot_outputs else widgets.VBox([]), joint_plots
    
    def _display_video_interface(self, widgets_dict, plot_widgets):
        """Display video playback interface."""
        controls = [widgets_dict['play'], widgets_dict['slider']]
        if 'frame_info' in widgets_dict:
            controls.append(widgets_dict['frame_info'])
        
        controls_box = widgets.HBox(controls)
        video_area = widgets.VBox([controls_box, widgets_dict['image_widget']])
        
        if len(plot_widgets.children) > 0:
            main_area = widgets.HBox([video_area, plot_widgets])
        else:
            main_area = video_area
        
        display(main_area)
    
    def _create_image_grid(self, images, canvas_w, num_cols):
        """Create image grid for scene visualization."""
        num_rows = math.ceil(len(images) / num_cols)
        h, w = images[0].shape[:2]
        aspect_ratio = h / w
        
        cell_w = canvas_w // num_cols
        cell_h = int(cell_w * aspect_ratio)
        canvas_h = cell_h * num_rows
        
        if len(images[0].shape) == 3:
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        
        for idx, img in enumerate(images):
            row = idx // num_cols
            col = idx % num_cols
            
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            
            resized_img = cv2.resize(img, (cell_w, cell_h))
            canvas[y1:y2, x1:x2] = resized_img
        
        return canvas
    
    def export_video(self, 
                     frames: Union[np.ndarray, Image.Image, VideoFrame, List[Any]],
                     output_path: str, 
                     fps: int = 25,
                     codec: str = 'mp4v') -> None:
        """
        Export frames as MP4 video.
        
        Args:
            frames: Frames to export
            output_path: Path to save video
            fps: Frames per second
            codec: Video codec
        """
        self._prepare_frames(frames)
        create_video_from_frames(self.frames_list, output_path, fps, codec)


def visualize_frames(
    frames, 
    frame_max_size=None, 
    widget_width=600,
    widget_height=400,
    vlm_predictor=None, 
    frame_results=None
):
    """
    Convenience function for frame browsing.
    
    Args:
        frames: Frames to visualize
        frame_max_size: Maximum size for image processing
        widget_width: Width of widget in notebook
        widget_height: Height of widget in notebook
        vlm_predictor: Optional predictor
        frame_results: Optional results
    """
    visualizer = FrameVisualizer(frame_max_size, widget_width, widget_height, vlm_predictor)
    visualizer.browse(frames, frame_results)

def visualize_frames_video(
    frames, 
    frame_max_size=512, 
    widget_width=600,
    widget_height=400,
    fps=25, 
    image_format='JPEG', 
    stats=None, 
    stats_fig_size=(8, 2)
):
    """
    Convenience function for video playback.
    
    Args:
        frames: Frames to visualize
        frame_max_size: Maximum size for image processing
        widget_width: Width of widget in notebook
        widget_height: Height of widget in notebook
        fps: Frames per second
        image_format: Image format
        stats: Optional statistics
        stats_fig_size: Statistics plot size
    """
    visualizer = FrameVisualizer(frame_max_size, widget_width, widget_height)
    visualizer.play(frames, fps=fps, image_format=image_format, stats=stats, stats_fig_size=stats_fig_size)

def visualize_scenes(
    scene_frames_dict, 
    grid_canvas_w=512, 
    grid_num_cols=3, 
    widget_width=600,
    widget_height=400,
    vlm_predictor=None, 
    scene_captions=None
):
    """
    Convenience function for scene visualization.
    
    Args:
        scene_frames_dict: Dictionary mapping scene IDs to frame lists
        grid_canvas_w: Width of grid canvas for image processing (affects quality)
        grid_num_cols: Number of columns in grid
        widget_width: Width of widget in notebook (affects display size)
        widget_height: Height of widget in notebook (affects display size)
        vlm_predictor: Optional predictor
        scene_captions: Optional scene captions
    """
    visualizer = FrameVisualizer(widget_width=widget_width, widget_height=widget_height, vlm_predictor=vlm_predictor)
    visualizer.browse_scenes(scene_frames_dict, grid_canvas_w, grid_num_cols, scene_captions)