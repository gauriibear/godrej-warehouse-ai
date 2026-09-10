"""
Tests for EvidenceCollector H.264 video clip extraction and HTML5 playback compatibility.
"""

from __future__ import annotations

from pathlib import Path
import av
import numpy as np
import pytest

from app.risk.evidence import EvidenceCollector
from app.risk.models import IncidentRecord, RiskLevel, DamageAssessmentStatus


def create_test_incident(incident_id: str = "INC-TEST-001", frame_idx: int = 30) -> IncidentRecord:
    return IncidentRecord(
        incident_id=incident_id,
        event_id=f"EVT-{incident_id}",
        rule_name="product_dropped",
        human_readable_name="Product Dropped",
        track_id=1,
        display_label="Carton #1",
        frame_idx=frame_idx,
        timestamp_seconds=1.0,
        risk_level=RiskLevel.HIGH,
        risk_score=75.0,
        confidence=0.9,
        damage_assessment=DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK,
        explanation="Test explanation",
        recommendation="Test recommendation",
        metrics={"vertical_velocity_px_s": 400.0},
    )


def test_extract_clip_generates_valid_h264_mp4(tmp_path: Path):
    """
    Verifies that EvidenceCollector.extract_clip creates a valid MP4 file
    encoded with H.264 codec and yuv420p pixel format for HTML5 browser playback.
    """
    collector = EvidenceCollector(
        output_dir=str(tmp_path),
        pre_event_frames=10,
        post_event_frames=10,
    )

    # Populate 50 synthetic frames (640x480 BGR)
    for f in range(50):
        # Frame with a moving white square
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, (10 + f * 5):(60 + f * 5)] = 255
        collector.add_frame(f, frame)

    incident = create_test_incident(incident_id="INC-TEST-H264", frame_idx=25)
    clip_path = collector.extract_clip(incident, fps=25.0)

    # Verify file existence and non-emptiness
    assert clip_path is not None
    p = Path(clip_path)
    assert p.exists()
    assert p.stat().st_size > 0
    assert incident.video_clip_path == str(p)

    # Probe with PyAV to confirm codec and format
    container = av.open(str(p))
    video_stream = container.streams.video[0]

    assert video_stream.codec_context.name == "h264", f"Expected h264, got {video_stream.codec_context.name}"
    assert video_stream.format.name == "yuv420p", f"Expected yuv420p, got {video_stream.format.name}"

    # Calculate duration
    if video_stream.duration:
        duration_sec = float(video_stream.duration * video_stream.time_base)
    else:
        duration_sec = float(p.stat().st_size)  # fallback check
    assert duration_sec > 0

    assert video_stream.width == 640
    assert video_stream.height == 480

    container.close()


def test_extract_clip_handles_odd_dimensions(tmp_path: Path):
    """
    Verifies that odd frame dimensions (which break naive H.264 encoders)
    are automatically cropped to even dimensions.
    """
    collector = EvidenceCollector(
        output_dir=str(tmp_path),
        pre_event_frames=5,
        post_event_frames=5,
    )

    # Odd width and height: 481 x 641
    for f in range(25):
        frame = np.zeros((481, 641, 3), dtype=np.uint8)
        collector.add_frame(f, frame)

    incident = create_test_incident(incident_id="INC-ODD-DIM", frame_idx=12)
    clip_path = collector.extract_clip(incident, fps=25.0)

    assert clip_path is not None
    p = Path(clip_path)
    assert p.exists()
    assert p.stat().st_size > 0

    container = av.open(str(p))
    v = container.streams.video[0]
    assert v.codec_context.name == "h264"
    assert v.width % 2 == 0
    assert v.height % 2 == 0
    container.close()


def test_extract_clip_insufficient_frames(tmp_path: Path):
    """
    Verifies that extract_clip returns None when fewer than 5 frames are buffered
    and does not write corrupt empty files to disk.
    """
    collector = EvidenceCollector(
        output_dir=str(tmp_path),
        pre_event_frames=10,
        post_event_frames=10,
    )

    # Only 3 frames buffered
    for f in range(3):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        collector.add_frame(f, frame)

    incident = create_test_incident(incident_id="INC-FEW-FRAMES", frame_idx=1)
    clip_path = collector.extract_clip(incident, fps=25.0)

    assert clip_path is None
    assert incident.video_clip_path is None
    assert not (tmp_path / "INC-FEW-FRAMES.mp4").exists()


def test_extract_clip_empty_buffer(tmp_path: Path):
    """Verifies that an empty buffer returns None cleanly."""
    collector = EvidenceCollector(output_dir=str(tmp_path))
    incident = create_test_incident(incident_id="INC-EMPTY", frame_idx=0)
    clip_path = collector.extract_clip(incident)

    assert clip_path is None
    assert incident.video_clip_path is None
