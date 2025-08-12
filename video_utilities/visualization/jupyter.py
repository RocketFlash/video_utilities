import cv2
import json
import numpy as np
from IPython.display import display
from PIL import Image
import ipywidgets as widgets
from ..video_frame_splitter import VideoFrame
from ..video_captioner import VideoFrameOutputResult
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)
import math
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO


def visualize_frames(
    frames: Union[
        np.ndarray, 
        Image.Image, 
        VideoFrame,
        List[np.ndarray], 
        List[Image.Image], 
        List[VideoFrame]
    ],
    frame_max_size: Optional[int] = None,
    vlm_predictor=None,
    frame_results: Optional[Union[List[VideoFrameOutputResult], List[dict], List[str], VideoFrameOutputResult, dict, str]] = None
):
    # Normalize input to list format
    frames_list = _normalize_frames_input(frames)
    frame_results_list = _normalize_results_input(frame_results, len(frames_list))
    
    n_frames = len(frames_list)
    if n_frames == 0:
        print("No frames to visualize")
        return
    
    # Determine frame format and get dimensions
    frame_info = _get_frame_info(frames_list[0])
    is_video_frame_format = frame_info['is_video_frame']
    frame_height, frame_width = frame_info['height'], frame_info['width']
    
    # Calculate resize scale
    resize_scale = _calculate_resize_scale(frame_height, frame_width, frame_max_size)
    if resize_scale != 1:
        frame_height = int(frame_height * resize_scale)
        frame_width = int(frame_width * resize_scale)
    
    # Create widgets
    widgets_dict = _create_widgets(
        n_frames, frame_height, frame_width, 
        is_video_frame_format, vlm_predictor, frame_results_list
    )
    
    # Display interface
    _display_interface(widgets_dict, is_video_frame_format)
    
    # Set up frame viewing logic
    _setup_frame_viewer(
        widgets_dict, frames_list, frame_results_list, 
        is_video_frame_format, resize_scale, vlm_predictor
    )

def _normalize_frames_input(frames):
    """Convert single frame or list of frames to consistent list format."""
    if isinstance(frames, list):
        return frames
    else:
        return [frames]

def _normalize_results_input(frame_results, n_frames):
    """Convert frame_results to consistent list format."""
    if frame_results is None:
        return None
    
    if isinstance(frame_results, list):
        return frame_results
    else:
        # Single result - replicate for all frames or just use for first frame
        return [frame_results] + [None] * (n_frames - 1)

def _get_frame_info(frame):
    """Extract frame information (dimensions and type)."""
    if isinstance(frame, np.ndarray):
        if len(frame.shape) == 3:
            height, width, _ = frame.shape
        else:
            height, width = frame.shape
        return {'height': height, 'width': width, 'is_video_frame': False}
    elif isinstance(frame, Image.Image):
        width, height = frame.size
        return {'height': height, 'width': width, 'is_video_frame': False}
    elif hasattr(frame, 'image'):  # VideoFrame
        if isinstance(frame.image, np.ndarray):
            if len(frame.image.shape) == 3:
                height, width, _ = frame.image.shape
            else:
                height, width = frame.image.shape
        else:  # PIL Image
            width, height = frame.image.size
        return {'height': height, 'width': width, 'is_video_frame': True}
    else:
        raise ValueError(f"Unsupported frame type: {type(frame)}")

def _calculate_resize_scale(frame_height, frame_width, frame_max_size):
    """Calculate resize scale based on max size constraint."""
    if frame_max_size is None:
        return 1
    
    max_dim = max(frame_height, frame_width)
    if max_dim > frame_max_size:
        return frame_max_size / max_dim
    return 1

def _create_widgets(n_frames, frame_height, frame_width, is_video_frame_format, vlm_predictor, frame_results):
    """Create all necessary widgets."""
    widgets_dict = {
        'frame_slider': widgets.IntSlider(
            value=0,
            min=0,
            max=max(0, n_frames-1),
            step=1,
            continuous_update=False,
            description='Frame'
        ),
        'image_widget': widgets.Image(
            format='jpeg',
            width=frame_width,
            height=frame_height
        )
    }
    
    # Add frame info widgets for VideoFrame format
    if is_video_frame_format:
        widgets_dict.update({
            'timestamp_label': widgets.Label(value="Timestamp: unknown"),
            'frame_idx_label': widgets.Label(value="Frame idx: unknown"),
            'scene_id_label': widgets.Label(value="Scene id: unknown")
        })
    
    # Add prediction widgets if needed
    if vlm_predictor is not None or frame_results is not None:
        widgets_dict['caption_widget'] = widgets.Textarea(
            value='',
            description='Prediction:',
            layout=widgets.Layout(
                width='400px',
                height=f'{frame_height}px'
            )
        )
        
        if vlm_predictor is not None:
            widgets_dict['caption_button'] = widgets.Button(
                description="Get Prediction"
            )
    
    return widgets_dict

def _display_interface(widgets_dict, is_video_frame_format):
    """Display the widget interface."""
    wgts_output = [widgets_dict['image_widget']]
    wgts_interface = [widgets_dict['frame_slider']]
    
    if 'caption_widget' in widgets_dict:
        wgts_output.append(widgets_dict['caption_widget'])
    
    if 'caption_button' in widgets_dict:
        wgts_interface.append(widgets_dict['caption_button'])
    
    wgts_frame_info = []
    if is_video_frame_format:
        wgts_frame_info = [
            widgets_dict['timestamp_label'],
            widgets_dict['frame_idx_label'], 
            widgets_dict['scene_id_label']
        ]
    
    display(
        widgets.VBox([
            widgets.HBox(wgts_output),
            widgets.VBox(wgts_frame_info),
            widgets.HBox(wgts_interface),
        ])
    )

def _setup_frame_viewer(widgets_dict, frames_list, frame_results_list, is_video_frame_format, resize_scale, vlm_predictor):
    """Set up the frame viewing logic and event handlers."""
    
    def view_frame(change):
        frame_idx = widgets_dict['frame_slider'].value
        frame = frames_list[frame_idx]
        
        # Handle VideoFrame format
        if is_video_frame_format:
            _update_frame_info_labels(widgets_dict, frame)
            display_frame = frame.image
        else:
            display_frame = frame
        
        # Convert and resize frame for display
        display_image = _prepare_display_image(display_frame, resize_scale)
        widgets_dict['image_widget'].value = display_image._repr_jpeg_()
        
        # Update caption if frame_results provided
        if frame_results_list is not None and 'caption_widget' in widgets_dict:
            _update_caption_widget(widgets_dict['caption_widget'], frame_results_list, frame_idx)
    
    def on_prediction_button_click(b):
        if vlm_predictor is not None:
            frame_idx = widgets_dict['frame_slider'].value
            frame = frames_list[frame_idx]
            
            # Get the actual image data
            if is_video_frame_format:
                input_frame = frame.image
            else:
                input_frame = frame
            
            # Convert PIL to numpy if needed for predictor
            if isinstance(input_frame, Image.Image):
                input_frame = np.array(input_frame)
            
            outputs = vlm_predictor(input_frame)
            widgets_dict['caption_widget'].value = str(outputs)
    
    # Set up event handlers
    widgets_dict['frame_slider'].observe(view_frame, names='value')
    
    if 'caption_button' in widgets_dict:
        widgets_dict['caption_button'].on_click(on_prediction_button_click)
    
    # Initial view
    view_frame(None)

def _update_frame_info_labels(widgets_dict, frame):
    """Update frame information labels for VideoFrame."""
    widgets_dict['timestamp_label'].value = f"Timestamp: {frame.timestamp:.2f}   "
    widgets_dict['frame_idx_label'].value = f"Frame idx: {frame.idx}   "
    scene_id_str = f"Scene id: {frame.scene_id}   " if frame.scene_id is not None else ''
    widgets_dict['scene_id_label'].value = scene_id_str

def _prepare_display_image(frame, resize_scale):
    """Convert frame to PIL Image and resize if needed."""
    # Convert to PIL Image
    if isinstance(frame, np.ndarray):
        # Handle grayscale
        if len(frame.shape) == 2:
            pil_img = Image.fromarray(frame, mode='L')
        else:
            pil_img = Image.fromarray(frame)
    elif isinstance(frame, Image.Image):
        pil_img = frame.copy()
    else:
        raise ValueError(f"Unsupported frame type for display: {type(frame)}")
    
    # Resize if needed
    if resize_scale != 1:
        new_width = int(pil_img.width * resize_scale)
        new_height = int(pil_img.height * resize_scale)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return pil_img

def _update_caption_widget(caption_widget, frame_results_list, frame_idx):
    """Update caption widget with frame results."""
    if frame_idx < len(frame_results_list) and frame_results_list[frame_idx] is not None:
        result = frame_results_list[frame_idx]
        
        if isinstance(result, dict):
            outputs_str = json.dumps(result, indent=4)
        elif isinstance(result, str):
            outputs_str = result
        elif hasattr(result, 'outputs') and result.outputs is not None:
            outputs_str = json.dumps(result.outputs, indent=4)
        else:
            outputs_str = str(result)
        
        caption_widget.value = outputs_str
    else:
        caption_widget.value = ""


def create_image_grid(
    images, 
    canvas_w, 
    num_cols
):
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
        
        y1 = row * cell_h
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w
        resized_img = cv2.resize(img, (cell_w, cell_h))
        canvas[y1:y2, x1:x2] = resized_img
    
    return canvas


def visualize_scenes(
    scene_frames_dict: Union[
        Dict[int, List[np.ndarray]], Dict[int, List[VideoFrame]]
    ],
    grid_canvas_w: int = 512,
    grid_num_cols: int = 3,
    vlm_predictor=None,
    scene_captions=None
):
    n_scenes = len(scene_frames_dict)
    is_video_frame_format = False
    example_frame = list(scene_frames_dict.values())[0][0]

    if not isinstance(example_frame, np.ndarray):
        is_video_frame_format = True

    timestamp_label = widgets.Label(
        value=f"Timestamp range: unknown"
    )
    frame_idx_label = widgets.Label(
        value=f"Frame idx range: unknown"
    )
    scene_id_label = widgets.Label(
        value=f"Scene id: unknown"
    )
    scene_caption_label = widgets.Label(
        value=f"Scene caption: unknown"
    )

    scene_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_scenes-1,
        step=1,
        continuous_update=False,
        description='Scene'
    )

    image_widget = widgets.Image(
        format='jpeg',
        width=grid_canvas_w,
        # height=frame_height
    )

    if scene_captions is not None:
        scene_caption_label.value = scene_captions[scene_slider.value]

    wgts_interface = [scene_slider]

    wgts_frame_info = []
    if is_video_frame_format:
        wgts_frame_info = [timestamp_label, frame_idx_label, scene_id_label]

    caption_widget = None
    if vlm_predictor is not None:
        caption_widget = widgets.Textarea(
            value='',
            description='Prediction:',
            layout=widgets.Layout(
                width='400px',
                height='512px',
            )
        )

        if vlm_predictor is not None:
            def on_button_click(b):       
                scene_images = scene_frames_dict[scene_slider.value]
                if is_video_frame_format:
                    scene_images = [frame.image for frame in scene_images]
                outputs = vlm_predictor(scene_images)
                caption_widget.value = outputs

            caption_button = widgets.Button(
                description="Get Prediction"
            )
            caption_button.on_click(on_button_click)     
            wgts_interface.append(caption_button)

    all_widgets = widgets.VBox([
        image_widget,
        scene_caption_label,
        widgets.VBox(wgts_frame_info),
        widgets.HBox(wgts_interface),
    ])
    if caption_widget is not None:
        all_widgets = widgets.HBox([
            all_widgets,
            caption_widget
        ])

    display(all_widgets)

    def view_frame(change):
        scene_images = scene_frames_dict[scene_slider.value]
        if scene_captions is not None:
            scene_caption_label.value = scene_captions[scene_slider.value]

        if is_video_frame_format:
            scene_start_timestamp = scene_images[0].timestamp
            scene_end_timestamp = scene_images[-1].timestamp
            scene_start_idx = scene_images[0].idx
            scene_end_idx = scene_images[-1].idx
            scene_id = scene_images[0].scene_id
            timestamp_label.value = f"Timestamp range: [ {scene_start_timestamp:.2f} - {scene_end_timestamp:.2f} ]"
            frame_idx_label.value = f"Frame idx range: [ {scene_start_idx} - {scene_end_idx} ]"
            scene_id_label.value  = f"Scene id: {scene_id}   "
            scene_images = [frame.image for frame in scene_images]
            
        canvas = create_image_grid(
            scene_images, 
            canvas_w=grid_canvas_w, 
            num_cols=grid_num_cols
        )
        img = Image.fromarray(canvas)
        image_widget.value = img._repr_jpeg_()

    scene_slider.observe(view_frame, names='value')
    view_frame(None)


def visualize_frames_video(
    frames: Union[
        np.ndarray, 
        Image.Image, 
        VideoFrame,
        List[np.ndarray], 
        List[Image.Image], 
        List[VideoFrame]
    ],
    frame_max_size: int = 512, 
    fps: int = 25,
    image_format: str = 'JPEG',  # JPEG or PNG
    stats: Optional[Dict[str, np.ndarray]] = None,
    stats_fig_size: tuple = (8, 2)
):
    """
    Create a video-like visualization with play controls for frames.
    
    Args:
        frames: Single frame or list of frames (numpy arrays, PIL Images, or VideoFrames)
        frame_max_size: Maximum size for the longest dimension
        fps: Frames per second for playback
        image_format: Image format for encoding ('JPEG' or 'PNG')
        stats: Dictionary of statistics to plot alongside video
        stats_fig_size: Figure size for statistics plots
    """
    # Normalize input to list format
    frames_list = _normalize_frames_input(frames)
    
    if len(frames_list) == 0:
        print("No frames to visualize")
        return
    
    # Process frames for display
    processed_frames = _process_frames_for_video(frames_list, frame_max_size, image_format)
    
    # Create widgets
    widgets_dict = _create_video_widgets(processed_frames, fps)
    
    # Create statistics plots if provided
    plot_widgets, joint_plots = _create_stats_plots(stats, stats_fig_size, len(frames_list))
    
    # Set up widget interactions
    _setup_video_interactions(widgets_dict, processed_frames, joint_plots)
    
    # Display interface
    _display_video_interface(widgets_dict, plot_widgets)

def _process_frames_for_video(frames_list, frame_max_size, image_format):
    """Process frames for video display - resize and encode."""
    processed_frames = []
    
    for frame in frames_list:
        # Extract actual image data
        if hasattr(frame, 'image'):  # VideoFrame
            image_data = frame.image
        else:
            image_data = frame
        
        # Convert PIL to numpy if needed for processing
        if isinstance(image_data, Image.Image):
            # Get dimensions from PIL image
            w, h = image_data.size
            numpy_frame = np.array(image_data)
        else:
            # Numpy array
            numpy_frame = image_data.copy()
            if len(numpy_frame.shape) == 3:
                h, w = numpy_frame.shape[:2]
            else:
                h, w = numpy_frame.shape
        
        # Resize if needed
        longest_dim = max(h, w)
        if longest_dim > frame_max_size:
            scale = frame_max_size / longest_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if isinstance(image_data, Image.Image):
                pil_img = image_data.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                resized_frame = cv2.resize(numpy_frame, (new_w, new_h))
                pil_img = Image.fromarray(resized_frame)
        else:
            if isinstance(image_data, Image.Image):
                pil_img = image_data.copy()
            else:
                pil_img = Image.fromarray(numpy_frame)
        
        # Encode to bytes
        with BytesIO() as buf:
            # Handle grayscale images
            if pil_img.mode == 'L' and image_format.upper() == 'JPEG':
                # Convert grayscale to RGB for JPEG
                pil_img = pil_img.convert('RGB')
            
            pil_img.save(buf, format=image_format.upper())
            processed_frames.append(buf.getvalue())
    
    return processed_frames

def _create_video_widgets(processed_frames, fps):
    """Create video playback widgets."""
    max_frame_idx = max(0, len(processed_frames) - 1)
    
    return {
        'play': widgets.Play(
            value=0,
            min=0,
            max=max_frame_idx,
            step=1,
            interval=max(1, int(1000 / fps)),
            description="Play",
            disabled=False
        ),
        'slider': widgets.IntSlider(
            value=0,
            min=0,
            max=max_frame_idx,
            description="Frame"
        ),
        'image_widget': widgets.Image(
            value=processed_frames[0] if processed_frames else b''
        )
    }

def _create_stats_plots(stats, stats_fig_size, n_frames):
    """Create statistics plots if stats are provided."""
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
                # 3D data (x, y, z)
                colors = ['red', 'green', 'blue']
                labels = ['x', 'y', 'z']
                for i in range(3):
                    ax.plot(stat_values[:, i], color=colors[i], label=labels[i])
                ax.legend()
            elif len(stat_values.shape) == 2 and stat_values.shape[1] > 1:
                # Multi-dimensional data
                for i in range(min(stat_values.shape[1], 5)):  # Limit to 5 lines
                    ax.plot(stat_values[:, i], label=f'dim_{i}')
                if stat_values.shape[1] <= 5:
                    ax.legend()
            else:
                # 1D data
                ax.plot(stat_values.flatten())
            
            # Add vertical line indicator
            vline = ax.axvline(x=0, color='black', linewidth=2, alpha=0.7)
            ax.set_xlim(0, max(1, len(stat_values) - 1))
            plt.tight_layout()
            
            joint_plots.append((fig, vline))
            
            # Create output widget for this plot
            out = widgets.Output()
            with out:
                display(fig.canvas)
            plot_outputs.append(out)
    
    except Exception as e:
        print(f"Error creating stats plots: {e}")
        joint_plots = []
        plot_outputs = []
    
    plt.ion()
    
    plots_box = widgets.VBox(plot_outputs) if plot_outputs else widgets.VBox([])
    return plots_box, joint_plots

def _setup_video_interactions(widgets_dict, processed_frames, joint_plots):
    """Set up widget interactions and event handlers."""
    
    def sync_widgets(change):
        """Sync slider with play widget."""
        widgets_dict['slider'].value = widgets_dict['play'].value
    
    def update_display(change):
        """Update image and plots when frame changes."""
        frame_idx = change['new']
        
        # Ensure frame index is within bounds
        if 0 <= frame_idx < len(processed_frames):
            widgets_dict['image_widget'].value = processed_frames[frame_idx]
        
        # Update plot vertical lines
        for fig, vline in joint_plots:
            try:
                vline.set_xdata([frame_idx, frame_idx])
                fig.canvas.draw_idle()
            except Exception as e:
                print(f"Error updating plot: {e}")
    
    # Set up event handlers
    widgets_dict['play'].observe(sync_widgets, names='value')
    widgets_dict['slider'].observe(update_display, names='value')

def _display_video_interface(widgets_dict, plot_widgets):
    """Display the complete video interface."""
    controls = widgets.HBox([
        widgets_dict['play'], 
        widgets_dict['slider']
    ])
    
    video_area = widgets.VBox([
        controls, 
        widgets_dict['image_widget']
    ])
    
    if len(plot_widgets.children) > 0:
        main_area = widgets.HBox([video_area, plot_widgets])
    else:
        main_area = video_area
    
    display(main_area)

# Enhanced version with additional features
def visualize_frames_video_advanced(
    frames: Union[
        np.ndarray, 
        Image.Image, 
        VideoFrame,
        List[np.ndarray], 
        List[Image.Image], 
        List[VideoFrame]
    ],
    frame_max_size: int = 512,
    fps: int = 25,
    image_format: str = 'JPEG',
    stats: Optional[Dict[str, np.ndarray]] = None,
    stats_fig_size: tuple = (8, 2),
    show_frame_info: bool = False,
    loop_playback: bool = True,
    auto_play: bool = False
):
    """
    Enhanced video visualization with additional features.
    
    Additional features:
        show_frame_info: Show frame information (timestamp, index) if available
        loop_playback: Enable loop playback
        auto_play: Start playing automatically
    """
    # Use the main function and add enhancements
    frames_list = _normalize_frames_input(frames)
    
    if len(frames_list) == 0:
        print("No frames to visualize")
        return
    
    # Process frames
    processed_frames = _process_frames_for_video(frames_list, frame_max_size, image_format)
    
    # Create enhanced widgets
    widgets_dict = _create_video_widgets(processed_frames, fps)
    
    # Add frame info if requested
    if show_frame_info:
        widgets_dict['frame_info'] = widgets.HTML(value="Frame: 0")
    
    # Create stats plots
    plot_widgets, joint_plots = _create_stats_plots(stats, stats_fig_size, len(frames_list))
    
    # Enhanced interactions
    def enhanced_update_display(change):
        frame_idx = change['new']
        
        if 0 <= frame_idx < len(processed_frames):
            widgets_dict['image_widget'].value = processed_frames[frame_idx]
        
        # Update frame info
        if 'frame_info' in widgets_dict:
            frame = frames_list[frame_idx]
            if hasattr(frame, 'timestamp') and hasattr(frame, 'idx'):
                info_text = f"Frame: {frame.idx}, Time: {frame.timestamp:.2f}s"
            else:
                info_text = f"Frame: {frame_idx}/{len(frames_list)-1}"
            widgets_dict['frame_info'].value = info_text
        
        # Update plots
        for fig, vline in joint_plots:
            try:
                vline.set_xdata([frame_idx, frame_idx])
                fig.canvas.draw_idle()
            except:
                pass
    
    # Set up interactions
    widgets_dict['play'].observe(lambda change: setattr(widgets_dict['slider'], 'value', widgets_dict['play'].value), names='value')
    widgets_dict['slider'].observe(enhanced_update_display, names='value')
    
    # Auto-play if requested
    if auto_play:
        widgets_dict['play'].value = 1  # Start playing
    
    # Display interface
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
    
    # Initialize display
    enhanced_update_display({'new': 0})