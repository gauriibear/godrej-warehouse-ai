"""
Unit tests for the Video Ingestion module (Stage 1).
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from app.video.reader import VideoReader, VideoMetadata


@pytest.fixture(scope="module")
def synthetic_video_path():
    """Generates a synthetic 60-frame 640x480 @ 30 FPS MP4 video for testing."""
    temp_dir = tempfile.TemporaryDirectory()
    video_path = Path(temp_dir.name) / "test_warehouse_sample.mp4"
    
    width, height, fps = 640, 480, 30.0
    total_frames = 60
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    for i in range(total_frames):
        # Create a frame with frame number stamped onto it
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add color gradient
        frame[:, :] = [(i * 4) % 255, 120, (255 - i * 4) % 255]
        cv2.putText(
            frame,
            f"Frame {i:03d}",
            (50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
        
    writer.release()
    yield video_path
    temp_dir.cleanup()


def test_metadata_extraction(synthetic_video_path):
    """Verifies resolution, framerate, total frames, and duration calculation."""
    reader = VideoReader(synthetic_video_path)
    meta = reader.metadata
    
    assert isinstance(meta, VideoMetadata)
    assert meta.width == 640
    assert meta.height == 480
    assert meta.resolution == "640x480"
    assert round(meta.fps) == 30
    assert meta.total_frames == 60
    assert pytest.approx(meta.duration_seconds, 0.1) == 2.0
    assert meta.formatted_duration == "00:02"
    assert meta.file_name == "test_warehouse_sample.mp4"
    
    meta_dict = meta.to_dict()
    assert "width" in meta_dict
    assert "height" in meta_dict
    assert "fps" in meta_dict
    assert "total_frames" in meta_dict
    
    reader.release()


def test_sequential_frame_reading(synthetic_video_path):
    """Verifies sequential frame-by-frame decoding and timestamp progression."""
    with VideoReader(synthetic_video_path) as reader:
        for expected_idx in range(60):
            success, frame, idx, timestamp = reader.read_frame()
            assert success is True
            assert frame is not None
            assert frame.shape == (480, 640, 3)
            assert idx == expected_idx
            assert pytest.approx(timestamp, 0.01) == expected_idx / 30.0
            
        # Reaching EOF should return False
        success, frame, _, _ = reader.read_frame()
        assert success is False
        assert frame is None


def test_frame_seeking(synthetic_video_path):
    """Verifies seeking to arbitrary frames and fetching frames by index."""
    with VideoReader(synthetic_video_path) as reader:
        # Fetch frame 10
        frame_10 = reader.get_frame_at(10)
        assert frame_10 is not None
        assert frame_10.shape == (480, 640, 3)
        
        # Fetch frame 40
        frame_40 = reader.get_frame_at(40)
        assert frame_40 is not None
        
        # Seek explicitly
        assert reader.seek_frame(25) is True
        success, frame, idx, timestamp = reader.read_frame()
        assert success is True
        assert idx == 25
        
        # Out-of-bounds seek
        assert reader.seek_frame(999) is False
        assert reader.seek_frame(-5) is False


def test_generator_iteration(synthetic_video_path):
    """Verifies generator yielding (frame_idx, timestamp, frame)."""
    with VideoReader(synthetic_video_path) as reader:
        frames_list = list(reader.frames(max_frames=15))
        assert len(frames_list) == 15
        
        for idx, (f_idx, ts, frame) in enumerate(frames_list):
            assert f_idx == idx
            assert pytest.approx(ts, 0.01) == idx / 30.0
            assert frame.shape == (480, 640, 3)


def test_nonexistent_file():
    """Verifies that attempting to open a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        VideoReader("non_existent_warehouse_feed.mp4")
