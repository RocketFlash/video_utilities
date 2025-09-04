import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def add_text_to_image(
    image, 
    text, 
    position=(10, 30), 
    font=cv2.FONT_HERSHEY_SIMPLEX,      
    font_scale=1.0, 
    color=(255, 255, 255), 
    thickness=2, 
    background=True, 
    bg_color=(0, 0, 0), 
    bg_alpha=0.5, 
    padding=5
):
    """
    Add text to an OpenCV image at the specified position (default: top-left corner).
    
    Args:
        image: NumPy array representing an OpenCV image (BGR format)
        text: Text string to write on the image
        position: Tuple (x, y) for the position of the text's bottom-left corner
        font: OpenCV font type
        font_scale: Size of the font
        color: Tuple (B, G, R) for text color
        thickness: Text thickness
        background: Whether to add a background rectangle behind the text
        bg_color: Tuple (B, G, R) for background color
        bg_alpha: Transparency of the background (0.0 to 1.0)
        padding: Padding around text for the background rectangle
        
    Returns:
        Modified image as a NumPy array
    """

    img_copy = image.copy()
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    if background:
        x, y = position
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + padding
        
        if bg_alpha < 1.0:
            overlay = img_copy.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            cv2.addWeighted(overlay, bg_alpha, img_copy, 1 - bg_alpha, 0, img_copy)
        else:
            cv2.rectangle(img_copy, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    cv2.putText(img_copy, text, position, font, font_scale, color, thickness)
    
    return img_copy


def create_video_from_frames(frames, output_path, fps, codec='mp4v'):
    """
    Create an MP4 video from a list of frames.
    
    Args:
        frames: List of frames (numpy arrays, PIL Images, or VideoFrame objects)
        output_path: Path to save the output video (should end with .mp4)
        fps: Frames per second for the output video
        codec: Video codec ('mp4v', 'XVID', 'H264', 'avc1'). Default 'mp4v'
    
    Raises:
        ValueError: If frames list is empty or frames have inconsistent dimensions
        IOError: If unable to create video writer or write frames
    """
    if not frames:
        raise ValueError("Frames list cannot be empty")
    
    # Convert frames to consistent numpy format
    processed_frames = []
    expected_shape = None
    
    for i, frame in enumerate(frames):
        # Handle different input types
        if hasattr(frame, 'image'):  # VideoFrame
            img = frame.image
            if isinstance(img, Image.Image):
                img = np.array(img)
        elif isinstance(frame, Image.Image):
            img = np.array(frame)
        elif isinstance(frame, np.ndarray):
            img = frame.copy()
        else:
            raise ValueError(f"Unsupported frame type at index {i}: {type(frame)}")
        
        # Handle grayscale images
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            # If it's RGB (from PIL or other sources), convert to BGR for OpenCV
            if img.shape[2] == 3:
                # Check if it needs RGB→BGR conversion (common with PIL images)
                # This is a heuristic - you might want to add a parameter for this
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Validate consistent dimensions
        if expected_shape is None:
            expected_shape = img.shape[:2]
        elif img.shape[:2] != expected_shape:
            raise ValueError(f"Frame {i} has shape {img.shape[:2]}, expected {expected_shape}")
        
        processed_frames.append(img)
    
    height, width = expected_shape
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try different codecs in order of preference
    codecs_to_try = [codec, 'mp4v', 'XVID', 'avc1'] if codec != 'mp4v' else ['mp4v', 'XVID', 'avc1']
    
    out = None
    for codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Test if the writer is working by checking if it's opened
            if out.isOpened():
                break
            else:
                out.release()
                out = None
        except Exception:
            if out is not None:
                out.release()
                out = None
            continue
    
    if out is None:
        raise IOError(f"Could not create video writer for {output_path}. Try a different codec.")
    
    try:
        # Write each frame to the video
        for i, img in enumerate(processed_frames):
            success = out.write(img)
            if not success:
                raise IOError(f"Failed to write frame {i}")
        
        print(f"Video saved to {output_path} ({len(processed_frames)} frames, {fps} FPS)")
        
    except Exception as e:
        raise IOError(f"Error writing video: {e}")
    
    finally:
        # Always release the VideoWriter
        if out is not None:
            out.release()