import cv2
from IPython.display import display
from PIL import Image
import ipywidgets as widgets
from typing import (
    Union, 
    Optional,
    List, 
    Dict
)


def visualize_frames(
    frames,
    frame_max_size: Optional[int] = None,
    image_captioner=None,
):
    n_frames = len(frames)
    frame_height, frame_width, _ = frames[0].shape
    resize_scale = 1

    if frame_max_size is not None:
        max_dim = max(frame_height, frame_width)
        if max_dim > frame_max_size:
            resize_scale = frame_max_size / max_dim

    if resize_scale != 1:
        frame_height = int(frame_height * resize_scale)
        frame_width  = int(frame_width * resize_scale)

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

    if image_captioner is not None:
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
            answers_list = image_captioner(frame)
            caption = '\n\n'.join(answers_list) 
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
            widgets.HBox(wgts_interface)
        ])
    )

    def view_frame(change):
        frame = frames[frame_slider.value]

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