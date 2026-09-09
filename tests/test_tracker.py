"""
Unit and integration tests for Stage 3 Object Tracking Module.
Validates ByteTrack integration, persistent ID assignment,
trajectory accumulation, and data structure integrity.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from app.detection.detector import Detection, DetectionResult
from app.tracking.tracker import (
    TrackPoint,
    TrackedObject,
    TrackingResult,
    ObjectTracker,
)


# ───────────────────────── TrackPoint Tests ─────────────────────────────────

def test_track_point_creation():
    """Validates basic TrackPoint instantiation and serialisation."""
    tp = TrackPoint(
        frame_idx=5,
        timestamp_seconds=0.167,
        track_id=3,
        class_name="person",
        raw_class_name="person",
        bbox=(100, 150, 300, 450),
        center=(200, 300),
        confidence=0.92,
    )

    assert tp.frame_idx == 5
    assert tp.track_id == 3
    assert tp.class_name == "person"
    assert tp.center == (200, 300)
    assert tp.confidence == 0.92

    d = tp.to_dict()
    assert d["frame_idx"] == 5
    assert d["track_id"] == 3
    assert d["bbox"] == [100, 150, 300, 450]
    assert d["center"] == [200, 300]
    assert d["confidence"] == 0.92


# ──────────────────────── TrackedObject Tests ───────────────────────────────

def test_tracked_object_display_label():
    """Validates human-readable display labels."""
    obj = TrackedObject(track_id=7, class_name="product/carton", raw_class_name="backpack")
    assert "7" in obj.display_label
    assert "Product" in obj.display_label or "Carton" in obj.display_label

    obj2 = TrackedObject(track_id=3, class_name="person", raw_class_name="person")
    assert obj2.display_label == "Person #3"


def test_tracked_object_trajectory_properties():
    """Validates trajectory-derived properties on TrackedObject."""
    obj = TrackedObject(track_id=1, class_name="person", raw_class_name="person")

    # Empty trajectory edge case
    assert obj.frame_count == 0
    assert obj.last_seen_frame == -1
    assert obj.last_bbox is None
    assert obj.last_center is None
    assert obj.last_confidence == 0.0
    assert obj.center_history == []

    # Add trajectory points
    tp1 = TrackPoint(
        frame_idx=0, timestamp_seconds=0.0, track_id=1,
        class_name="person", raw_class_name="person",
        bbox=(100, 100, 200, 300), center=(150, 200), confidence=0.9,
    )
    tp2 = TrackPoint(
        frame_idx=1, timestamp_seconds=0.033, track_id=1,
        class_name="person", raw_class_name="person",
        bbox=(105, 102, 205, 302), center=(155, 202), confidence=0.88,
    )
    tp3 = TrackPoint(
        frame_idx=2, timestamp_seconds=0.067, track_id=1,
        class_name="person", raw_class_name="person",
        bbox=(110, 105, 210, 305), center=(160, 205), confidence=0.85,
    )

    obj.trajectory = [tp1, tp2, tp3]

    assert obj.frame_count == 3
    assert obj.last_seen_frame == 2
    assert obj.last_bbox == (110, 105, 210, 305)
    assert obj.last_center == (160, 205)
    assert obj.last_confidence == 0.85
    assert len(obj.center_history) == 3
    assert obj.center_history[0] == (150, 200)
    assert obj.center_history[2] == (160, 205)


def test_tracked_object_serialisation():
    """Validates to_dict produces correct structure."""
    obj = TrackedObject(track_id=5, class_name="person", raw_class_name="person")
    tp = TrackPoint(
        frame_idx=0, timestamp_seconds=0.0, track_id=5,
        class_name="person", raw_class_name="person",
        bbox=(50, 50, 150, 250), center=(100, 150), confidence=0.95,
    )
    obj.trajectory.append(tp)

    d = obj.to_dict()
    assert d["track_id"] == 5
    assert d["class_name"] == "person"
    assert d["frame_count"] == 1
    assert d["last_seen_frame"] == 0
    assert len(d["trajectory"]) == 1
    assert d["trajectory"][0]["confidence"] == 0.95


# ──────────────────────── TrackingResult Tests ──────────────────────────────

def test_tracking_result_counts_by_class():
    """Validates class counting from active track IDs."""
    obj1 = TrackedObject(track_id=1, class_name="person", raw_class_name="person")
    obj2 = TrackedObject(track_id=2, class_name="person", raw_class_name="person")
    obj3 = TrackedObject(track_id=3, class_name="product/carton", raw_class_name="backpack")

    result = TrackingResult(
        frame_idx=10,
        timestamp_seconds=0.33,
        tracked_objects=[obj1, obj2, obj3],
        active_track_ids=[1, 2, 3],
    )

    assert result.active_count == 3
    counts = result.counts_by_class
    assert counts.get("person") == 2
    assert counts.get("product/carton") == 1


# ────────────────────── ObjectTracker Unit Tests ────────────────────────────

def test_tracker_iou_computation():
    """Validates the static IoU computation."""
    # Perfect overlap
    assert ObjectTracker._iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    # No overlap
    assert ObjectTracker._iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    # Partial overlap (50% area overlap)
    iou = ObjectTracker._iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert 0.3 < iou < 0.4  # Should be 50/150 ≈ 0.333


def test_tracker_initialisation():
    """Validates clean tracker startup."""
    tracker = ObjectTracker()
    assert tracker.total_unique_tracks == 0
    assert tracker.tracked_objects == {}


def test_tracker_reset():
    """Validates tracker reset clears state."""
    tracker = ObjectTracker()
    # Simulate some state
    tracker._tracked_objects[1] = TrackedObject(
        track_id=1, class_name="person", raw_class_name="person"
    )
    assert tracker.total_unique_tracks == 1

    tracker.reset()
    assert tracker.total_unique_tracks == 0
    assert tracker.tracked_objects == {}


def test_tracker_update_with_empty_detections():
    """Validates tracker handles frames with no detections gracefully."""
    tracker = ObjectTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    det_result = DetectionResult(
        frame_idx=0,
        timestamp_seconds=0.0,
        detections=[],
        inference_time_ms=5.0,
    )

    trk_result = tracker.update(det_result, frame, annotate=True)

    assert isinstance(trk_result, TrackingResult)
    assert trk_result.active_count == 0
    assert trk_result.annotated_frame is not None
    assert trk_result.annotated_frame.shape == (480, 640, 3)


def test_tracker_builds_sv_detections():
    """Validates conversion of Detection list to supervision.Detections."""
    tracker = ObjectTracker()

    detections = [
        Detection(bbox=(10, 20, 100, 200), confidence=0.9, class_id=0,
                  class_name="person", raw_class_name="person"),
        Detection(bbox=(200, 100, 350, 300), confidence=0.8, class_id=24,
                  class_name="product/carton", raw_class_name="backpack"),
    ]

    sv_dets = tracker._build_sv_detections(detections)

    assert sv_dets.xyxy.shape == (2, 4)
    assert sv_dets.confidence.shape == (2,)
    assert sv_dets.class_id.shape == (2,)
    assert float(sv_dets.confidence[0]) == pytest.approx(0.9, abs=0.01)
    assert int(sv_dets.class_id[1]) == 24


# ──────────── Integration: Multi-Frame Tracking with Trajectory ─────────────

def test_tracker_accumulates_trajectory_across_frames():
    """
    CRITICAL TEST: Proves that feeding detections over multiple frames
    results in trajectory history being accumulated for tracked objects.
    Uses synthetic detections with slight position changes to simulate movement.
    """
    tracker = ObjectTracker(
        track_activation_threshold=0.1,
        lost_track_buffer=30,
        minimum_matching_threshold=0.3,
        minimum_consecutive_frames=1,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    num_frames = 10

    for i in range(num_frames):
        # Simulate a person moving right by 5px per frame
        x_offset = i * 5
        detections = [
            Detection(
                bbox=(100 + x_offset, 100, 200 + x_offset, 300),
                confidence=0.9,
                class_id=0,
                class_name="person",
                raw_class_name="person",
            ),
        ]

        det_result = DetectionResult(
            frame_idx=i,
            timestamp_seconds=i / 30.0,
            detections=detections,
            inference_time_ms=10.0,
        )

        trk_result = tracker.update(det_result, frame, annotate=False)

    # Verify we have at least one tracked object
    assert tracker.total_unique_tracks >= 1, "Expected at least 1 unique tracked object"

    # Find the object with the most trajectory points
    best_obj = max(tracker.tracked_objects.values(), key=lambda o: o.frame_count)

    # The trajectory should have accumulated points across frames
    assert best_obj.frame_count >= 5, (
        f"Expected trajectory with ≥5 points, got {best_obj.frame_count}"
    )

    # Verify trajectory data integrity
    for tp in best_obj.trajectory:
        assert tp.frame_idx >= 0
        assert tp.timestamp_seconds >= 0.0
        assert tp.track_id >= 0
        assert tp.class_name == "person"
        assert len(tp.bbox) == 4
        assert len(tp.center) == 2
        assert 0.0 <= tp.confidence <= 1.0

    # Verify trajectory shows movement (x coordinates should increase)
    centers = best_obj.center_history
    if len(centers) >= 2:
        x_values = [c[0] for c in centers]
        assert x_values[-1] > x_values[0], (
            f"Expected rightward movement, but x went from {x_values[0]} to {x_values[-1]}"
        )


def test_tracker_multiple_objects_get_different_ids():
    """
    Verifies that two spatially separated objects receive different track IDs.
    """
    tracker = ObjectTracker(
        track_activation_threshold=0.1,
        lost_track_buffer=30,
        minimum_matching_threshold=0.3,
        minimum_consecutive_frames=1,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(5):
        detections = [
            Detection(
                bbox=(50, 50, 150, 200),
                confidence=0.9,
                class_id=0,
                class_name="person",
                raw_class_name="person",
            ),
            Detection(
                bbox=(400, 300, 550, 450),
                confidence=0.85,
                class_id=24,
                class_name="product/carton",
                raw_class_name="backpack",
            ),
        ]

        det_result = DetectionResult(
            frame_idx=i,
            timestamp_seconds=i / 30.0,
            detections=detections,
            inference_time_ms=8.0,
        )

        tracker.update(det_result, frame, annotate=False)

    # Should have at least 2 unique tracks
    assert tracker.total_unique_tracks >= 2, (
        f"Expected ≥2 unique tracks for 2 separated objects, got {tracker.total_unique_tracks}"
    )

    # Verify different IDs
    track_ids = list(tracker.tracked_objects.keys())
    assert len(set(track_ids)) >= 2, "Expected distinct track IDs for distinct objects"


def test_tracker_annotation_produces_valid_frame():
    """Verifies that annotated tracking output is a valid image."""
    tracker = ObjectTracker(
        track_activation_threshold=0.1,
        minimum_consecutive_frames=1,
    )

    frame = np.full((480, 640, 3), 80, dtype=np.uint8)

    detections = [
        Detection(
            bbox=(100, 100, 250, 350),
            confidence=0.9,
            class_id=0,
            class_name="person",
            raw_class_name="person",
        ),
    ]

    det_result = DetectionResult(
        frame_idx=0,
        timestamp_seconds=0.0,
        detections=detections,
        inference_time_ms=10.0,
    )

    trk_result = tracker.update(det_result, frame, annotate=True)

    assert trk_result.annotated_frame is not None
    assert trk_result.annotated_frame.shape == (480, 640, 3)
    # Should have drawn something (not identical to original frame)
    if trk_result.active_count > 0:
        assert not np.array_equal(trk_result.annotated_frame, frame), \
            "Annotated frame should differ from original when tracks are active"
