import numpy as np
import cv2
from IPython.display import display, clear_output
import ipywidgets as widgets
import threading
from PIL import Image
import io
from typing import (
    Union, 
    Optional,
    List, 
    Dict,
    Tuple
)

class WebcamVisualizer:
    def __init__(
        self,
        process_func = None,
        record_path: str = None,
        record_fps: Optional[int] = None,
        frame_max_size: Optional[int] = None
    ):
        self.process_func = process_func if process_func else lambda frame: frame
        self.cap = None
        self.running = False
        self.thread = None
        
        self.record_path = record_path
        self.record_fps = record_fps
        self.video_writer = None

        self.image_widget = widgets.Image(format='jpeg')
        self.stop_button = widgets.Button(description="Stop")
        self.stop_button.on_click(self.stop)
        self.display = widgets.VBox([self.image_widget, self.stop_button])
        self.frame_max_size = frame_max_size
        display(self.display)
    
    def start(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            
            if self.record_fps is None:
                webcam_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.record_fps = int(webcam_fps) if webcam_fps > 0 else 30
                print(f"Auto-detected webcam FPS: {self.record_fps}")
            
            self.running = True
            self.thread = threading.Thread(target=self.update)
            self.thread.start()
    
    def stop(self, b=None):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
    
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.frame_max_size is not None:
                frame_h_orig, frame_w_orig, _ = frame_rgb.shape
                resize_scale = self.frame_max_size / max(frame_h_orig, frame_w_orig)
                frame_rgb = cv2.resize(frame_rgb, (0, 0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
            
            if self.record_path and self.video_writer is None:
                h, w, _ = frame_rgb.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.record_path, fourcc, self.record_fps, (w, h))
                print(f"Recording at {self.record_fps} FPS")

            processed_rgb = self.process_func(frame_rgb)

            if self.video_writer:
                processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
                self.video_writer.write(processed_bgr)

            processed_rgb = frame_rgb
            pil_img = Image.fromarray(processed_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG')
            self.image_widget.value = buf.getvalue()

        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            print(f"Video successfully saved to: {self.record_path}")