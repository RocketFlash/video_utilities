import cv2
import numpy as np
from IPython.display import display
from PIL import Image
import ipywidgets as widgets
from ..video_frame_splitter import VideoFrame
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)


def visualize_frames(
    frames: Union[List[np.ndarray], List[VideoFrame]],
    frame_max_size: Optional[int] = None,
    frame_captioner=None,
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

    if frame_captioner is not None:
        caption_widget = widgets.Textarea(
            value='',
            description='Caption:',
            layout=widgets.Layout(
                width='400px',
                height=f'{frame_height}px'
            )
        )

        def on_button_click(b):       
            frame = frames[frame_slider.value]
            if is_video_frame_format:
                frame = frame.image

            outputs = frame_captioner(frame)
            output_strs = frame_captioner.outputs_to_string(outputs)
            caption = '\n\n'.join(output_strs) 
            caption_widget.value = caption

        caption_button = widgets.Button(
            description="Get Caption"
        )
        caption_button.on_click(on_button_click)
        wgts_output.append(caption_widget)
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

    frame_slider.observe(view_frame, names='value')
    view_frame(None)