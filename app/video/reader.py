"""
Video Ingestion Module for Godrej Warehouse AI.
Provides frame-by-frame decoding, metadata extraction, seeking, and robust stream handling using OpenCV.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Stores structured metadata of an ingested video."""
    file_path: str
    file_name: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    resolution: str
    formatted_duration: str
    codec: str

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "formatted_duration": self.formatted_duration,
            "codec": self.codec,
        }


class VideoReader:
    """
    High-performance video reader wrapping OpenCV's VideoCapture.
    Supports frame-by-frame iteration, seeking, and metadata retrieval.
    """

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = str(video_path)
        self.path_obj = Path(video_path)
        
        if not self.path_obj.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {self.video_path}")

        self._metadata: Optional[VideoMetadata] = None
        self._current_frame_idx = 0
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Extracts resolution, fps, frame count, and duration."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Guard against zero or NaN FPS
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Fallback standard framerate
        
        duration_seconds = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
        
        # Format duration as MM:SS or HH:MM:SS
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        hours = int(minutes // 60)
        if hours > 0:
            formatted_duration = f"{hours:02d}:{minutes % 60:02d}:{seconds:02d}"
        else:
            formatted_duration = f"{minutes:02d}:{seconds:02d}"

        # Get fourcc codec
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
        if not codec:
            codec = "Unknown"

        self._metadata = VideoMetadata(
            file_path=self.video_path,
            file_name=self.path_obj.name,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration_seconds,
            resolution=f"{width}x{height}",
            formatted_duration=formatted_duration,
            codec=codec,
        )

    @property
    def metadata(self) -> VideoMetadata:
        """Returns the cached video metadata."""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """
        Reads the next sequential frame from the video.
        
        Returns:
            Tuple[bool, Optional[np.ndarray], int, float]:
                - success: bool indicating if frame was read
                - frame: BGR image numpy array or None
                - frame_idx: 0-indexed frame number
                - timestamp_seconds: elapsed time in seconds
        """
        if not self.cap.isOpened():
            return False, None, -1, 0.0

        success, frame = self.cap.read()
        if not success or frame is None:
            return False, None, self._current_frame_idx, 0.0

        current_idx = self._current_frame_idx
        fps = self.metadata.fps if self.metadata.fps > 0 else 30.0
        timestamp = current_idx / fps
        
        self._current_frame_idx += 1
        return True, frame, current_idx, timestamp

    def seek_frame(self, frame_idx: int) -> bool:
        """
        Seeks to a specific 0-indexed frame position.
        
        Args:
            frame_idx: target frame index
            
        Returns:
            bool: True if seek succeeded
        """
        if not self.cap.isOpened():
            return False
        
        total = self.metadata.total_frames
        if total > 0 and (frame_idx < 0 or frame_idx >= total):
            return False

        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if success:
            self._current_frame_idx = frame_idx
        return bool(success)

    def get_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """Fetches a specific frame by index without losing track of current position."""
        current_pos = self._current_frame_idx
        if not self.seek_frame(frame_idx):
            return None
        
        success, frame, _, _ = self.read_frame()
        # Restore position
        self.seek_frame(current_pos)
        return frame if success else None

    def frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Generator yielding (frame_idx, timestamp_seconds, frame_bgr).
        
        Args:
            max_frames: Optional upper limit of frames to yield
        """
        count = 0
        while self.cap.isOpened():
            if max_frames is not None and count >= max_frames:
                break
            success, frame, idx, timestamp = self.read_frame()
            if not success or frame is None:
                break
            yield idx, timestamp, frame
            count += 1

    def release(self) -> None:
        """Closes video capture and releases system resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
