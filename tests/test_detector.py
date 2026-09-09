"""
Unit and integration tests for Stage 2 Object Detection Module.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest
from app.detection.detector import (
    Detection,
    DetectionResult,
    ObjectDetector,
    DEFAULT_WAREHOUSE_CLASS_MAPPING,
)
from app.video.reader import VideoReader


def test_detection_dataclass_properties():
    """Verifies geometric computations on the Detection dataclass."""
    det = Detection(
        bbox=(100, 150, 300, 450),
        confidence=0.88,
        class_id=0,
        class_name="person",
        raw_class_name="person",
    )

    assert det.x1 == 100
    assert det.y1 == 150
    assert det.x2 == 300
    assert det.y2 == 450
    assert det.width == 200
    assert det.height == 300
    assert det.center == (200, 300)
    assert det.foot_point == (200, 450)
    assert det.area == 60000
    assert pytest.approx(det.aspect_ratio, 0.01) == 200 / 300

    d = det.to_dict()
    assert d["bbox"] == [100, 150, 300, 450]
    assert d["confidence"] == 0.88
    assert d["class_name"] == "person"
    assert d["center"] == [200, 300]
    assert d["foot_point"] == [200, 450]


def test_detection_result_grouping():
    """Verifies counts_by_class and filtering methods."""
    det1 = Detection((10, 10, 50, 50), 0.9, 0, "person", "person")
    det2 = Detection((60, 60, 100, 100), 0.85, 0, "person", "person")
    det3 = Detection((110, 110, 150, 150), 0.75, 24, "product/carton", "backpack")

    res = DetectionResult(
        frame_idx=1,
        timestamp_seconds=0.033,
        detections=[det1, det2, det3],
        inference_time_ms=12.5,
    )

    assert res.count == 3
    assert res.counts_by_class == {"person": 2, "product/carton": 1}
    assert len(res.get_detections_by_class("person")) == 2
    assert len(res.get_detections_by_class("product/carton")) == 1
    assert len(res.get_detections_by_class("nonexistent")) == 0


def test_detector_annotation_rendering():
    """Verifies annotation overlay produces a valid modified image."""
    detector = ObjectDetector(confidence_threshold=0.25)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    det = Detection((100, 100, 250, 350), 0.92, 0, "person", "person")
    annotated = detector.annotate_frame(blank_frame, [det])

    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
    # The annotated frame should no longer be completely black
    assert np.any(annotated > 0)


def test_detector_threshold_configuration():
    """Verifies dynamic runtime adjustment of confidence threshold."""
    detector = ObjectDetector(confidence_threshold=0.30)
    assert detector.confidence_threshold == 0.30

    detector.set_confidence_threshold(0.75)
    assert detector.confidence_threshold == 0.75

    # Out of bounds clamping
    detector.set_confidence_threshold(1.5)
    assert detector.confidence_threshold == 1.0
    detector.set_confidence_threshold(-0.5)
    assert detector.confidence_threshold == 0.0


def test_detector_on_synthetic_video_frames():
    """Runs detector on generated video frames and checks output format."""
    detector = ObjectDetector(confidence_threshold=0.20)
    
    # Create a test frame
    test_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Draw an object
    cv2.rectangle(test_frame, (150, 100), (350, 400), (0, 180, 255), -1)

    result = detector.detect(test_frame, frame_idx=0, timestamp_seconds=0.0, annotate=True)

    assert isinstance(result, DetectionResult)
    assert result.frame_idx == 0
    assert result.timestamp_seconds == 0.0
    assert result.inference_time_ms > 0
    assert result.annotated_frame is not None
    assert result.annotated_frame.shape == test_frame.shape
