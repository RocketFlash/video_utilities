import cv2

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