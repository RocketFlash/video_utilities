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
    frames: Union[List[np.ndarray], List[VideoFrame]],
    frame_max_size: Optional[int] = None,
    vlm_predictor=None,
    frames_results: Optional[List[VideoFrameOutputResult]] = None
):
    n_frames = len(frames)
    is_video_frame_format = False
    if isinstance(frames[0], np.ndarray):
        frame_height, frame_width, _ = frames[0].shape
    else:
        is_video_frame_format = True
        frame_height, frame_width, _ = frames[0].image.shape

    resize_scale = 1

    if frame_max_size is not None:
        max_dim = max(frame_height, frame_width)
        if max_dim > frame_max_size:
            resize_scale = frame_max_size / max_dim

    if resize_scale != 1:
        frame_height = int(frame_height * resize_scale)
        frame_width  = int(frame_width * resize_scale)

    timestamp_label = widgets.Label(
        value=f"Timestamp: unknown"
    )
    frame_idx_label = widgets.Label(
        value=f"Frame idx: unknown"
    )
    scene_id_label = widgets.Label(
        value=f"Scene id: unknown"
    )
    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_frames-1,
        step=1,
        continuous_update=False,
        description='Frame'
    )

    image_widget = widgets.Image(
        format='jpeg',
        width=frame_width,
        height=frame_height
    )

    wgts_output = [image_widget]
    wgts_interface = [frame_slider]

    wgts_frame_info = []
    if is_video_frame_format:
        wgts_frame_info = [timestamp_label, frame_idx_label, scene_id_label]

    if vlm_predictor is not None or frames_results is not None:
        caption_widget = widgets.Textarea(
            value='',
            description='Prediction:',
            layout=widgets.Layout(
                width='400px',
                height=f'{frame_height}px'
            )
        )
        wgts_output.append(caption_widget)

        if vlm_predictor is not None:
            def on_button_click(b):       
                frame = frames[frame_slider.value]
                if is_video_frame_format:
                    frame = frame.image

                outputs = vlm_predictor(frame)
                caption_widget.value = outputs

            caption_button = widgets.Button(
                description="Get Prediction"
            )
            caption_button.on_click(on_button_click)     
            wgts_interface.append(caption_button)

    display(
        widgets.VBox([
            widgets.HBox(wgts_output),
            widgets.VBox(wgts_frame_info),
            widgets.HBox(wgts_interface),
        ])
    )

    def view_frame(change):
        frame = frames[frame_slider.value]

        if is_video_frame_format:
            timestamp_label.value = f"Timestamp: {frame.timestamp:.2f}   "
            frame_idx_label.value = f"Frame idx: {frame.idx}   "
            scene_id_str = f"Scene id: {frame.scene_id}   " if frame.scene_id is not None else ''
            scene_id_label.value  = scene_id_str
            frame = frame.image
            
        if resize_scale != 1:
            frame = cv2.resize(
                frame,
                (0, 0),
                fx=resize_scale,
                fy=resize_scale
            )
        img = Image.fromarray(frame)
        image_widget.value = img._repr_jpeg_()

        if frames_results is not None:
            outputs = frames_results[frame_slider.value].outputs
            if outputs is not None:
                outputs_str = json.dumps(outputs, indent=4)
            else:
                outputs_str = 'Something wrong with inference'
            caption_widget.value = outputs_str

    frame_slider.observe(view_frame, names='value')
    view_frame(None)


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
    frames, 
    frame_max_size=512, 
    fps=25,
    image_format='JPEG', # JPEG or PNG
    stats=None,
    stats_fig_size=(8, 2)
):
    processed_frames = []
    
    for frame in frames:
        h, w = frame.shape[:2]
        longest_dim = max(h, w)
        
        if longest_dim > frame_max_size:
            scale = frame_max_size / longest_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame.copy(), (new_w, new_h))
        
        pil_img = Image.fromarray(frame)
        with BytesIO() as buf:
            pil_img.save(buf, format=image_format)
            processed_frames.append(buf.getvalue())
    
    play = widgets.Play(
        value=0,
        min=0,
        max=len(processed_frames)-1,
        step=1,
        interval=max(1, int(1000 / fps)),
        description="Play",
        disabled=False
    )
    
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(processed_frames)-1,
        description="Frame"
    )
    
    image_widget = widgets.Image(value=processed_frames[0])
    
    if stats is not None:
        plt.close('all')
        plt.ioff()
        joint_plots = []
        plot_outputs = []
        
        for stat_name, stat_values in stats.items():
            fig, ax = plt.subplots(figsize=stats_fig_size)
            fig.canvas.header_visible = False
            ax.set_title(stat_name)
            
            if stat_values.ndim == 2 and stat_values.shape[1] == 3:
                for i in range(3):
                    ax.plot(stat_values[:, i], label=['x', 'y', 'z'][i])
                ax.legend()
            else:
                ax.plot(stat_values.flatten())
            
            vline = ax.axvline(x=0, color='black', linewidth=1)
            plt.tight_layout()
            joint_plots.append((fig, vline))
            
            out = widgets.Output()
            with out:
                display(fig.canvas)
            plot_outputs.append(out)
        
        plots_box = widgets.VBox(plot_outputs)
        plt.ion()
    else:
        joint_plots = []
        plots_box = widgets.VBox([])
    
    def sync_widgets(change):
        slider.value = play.value
    play.observe(sync_widgets, names='value')
    
    def update_display(change):
        frame_idx = change['new']
        image_widget.value = processed_frames[frame_idx]
        
        for fig, vline in joint_plots:
            vline.set_xdata([frame_idx, frame_idx])
            fig.canvas.draw_idle()
    
    slider.observe(update_display, names='value')
    
    controls = widgets.HBox([play, slider])
    main_area = widgets.HBox([widgets.VBox([controls, image_widget]), plots_box])
    
    display(main_area)