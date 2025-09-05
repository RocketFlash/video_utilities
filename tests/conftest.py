import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def test_results_dir():
    """Get test results directory."""
    results_dir = Path(__file__).parent / "test_results"
    results_dir.mkdir(exist_ok=True)
    return results_dir

@pytest.fixture(scope="session")
def sample_image_path(test_data_dir):
    """Path to sample test image."""
    image_path = test_data_dir / "sample_image.jpg"
    if not image_path.exists():
        # Create a test image if it doesn't exist
        test_data_dir.mkdir(exist_ok=True)
        image = create_test_image()
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return image_path

@pytest.fixture
def sample_image(sample_image_path):
    """Load sample test image as RGB numpy array."""
    image_bgr = cv2.imread(str(sample_image_path))
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

def create_test_image(width=640, height=480):
    """Create a synthetic test image with various patterns."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background
    for i in range(height):
        image[i, :, 0] = int(255 * i / height)  # Red gradient
    
    for j in range(width):
        image[:, j, 1] = int(255 * j / width)   # Green gradient
    
    # Add some geometric shapes
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(image, (300, 200), 50, (255, 0, 0), -1)
    cv2.line(image, (400, 100), (500, 300), (0, 255, 0), 5)
    
    # Add some text
    cv2.putText(image, "TEST", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image

@pytest.fixture(params=[
    (640, 480, 3),   # Color image
    (320, 240, 3),   # Smaller color image
    (640, 480, 1),   # Grayscale image
])
def various_images(request):
    """Generate various test images with different dimensions."""
    height, width, channels = request.param
    
    if channels == 3:
        image = create_test_image(width, height)
    else:
        color_image = create_test_image(width, height)
        image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    
    return image

@pytest.fixture(scope="session") 
def small_video_path(test_data_dir):
    """Create a small test video for quick testing."""
    video_path = test_data_dir / "small_video.mp4"
    
    if not video_path.exists():
        create_test_video(video_path, duration=5, fps=15, width=320, height=240)
    
    return video_path

def create_test_video(output_path, duration=30, fps=30, width=640, height=480):
    """Create a synthetic test video with sufficient length for testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use a more compatible codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    
    if not out.isOpened():
        # Try alternative codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    
    total_frames = duration * fps
    print(f"Creating test video: {duration}s, {fps}fps, {total_frames} frames")
    
    for frame_num in range(total_frames):
        # Fast method: create gradient using numpy broadcasting
        time_factor = frame_num / total_frames
        
        # Create solid background with time-based color
        hue = int((time_factor * 180) % 180)
        color = np.array([hue, 100, 200], dtype=np.uint8)
        
        # Convert HSV to BGR (much faster than pixel-by-pixel)
        hsv_frame = np.full((height, width, 3), color, dtype=np.uint8)
        frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
        
        # Add simple moving elements
        center_x = int(width/2 + 50 * np.sin(frame_num * 0.1))
        center_y = int(height/2 + 30 * np.cos(frame_num * 0.15))
        cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), -1)
        
        # Frame info
        cv2.putText(frame, f'{frame_num}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        time_sec = frame_num / fps
        cv2.putText(frame, f'{time_sec:.1f}s', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    
    # Verify the video was created successfully
    if output_path.exists() and output_path.stat().st_size > 10000:  # At least 10KB
        print(f"Test video created successfully: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
        return True
    else:
        print(f"Warning: Test video may be too small: {output_path}")
        return False
    
@pytest.fixture(scope="session")
def sample_video_path(test_data_dir):
    """Create or find a sample video for testing."""
    video_path = test_data_dir / "sample_video.mp4"
    
    if not video_path.exists():
        print("Creating test video...")
        success = create_test_video(video_path, duration=30, fps=30)
        if not success:
            pytest.skip("Could not create test video")
    
    # Verify video is usable
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        pytest.skip(f"Test video {video_path} cannot be opened")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    # Less strict validation - allow shorter videos for basic tests
    if frame_count < 50:  # Reduced from 100
        print(f"Video too short for full testing: {frame_count} frames")
        pytest.skip(f"Test video too short: {frame_count} frames, {duration:.1f}s")
    
    print(f"Using test video: {frame_count} frames, {fps} FPS, {duration:.1f}s")
    return video_path

# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
    config.addinivalue_line("markers", "integration: marks integration tests")