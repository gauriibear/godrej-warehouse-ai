"""
Multi-Object Tracker for Godrej Warehouse AI.
Uses supervision's ByteTrack implementation to maintain persistent object IDs
across video frames and accumulate trajectory history for downstream behaviour analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from app.detection.detector import Detection, DetectionResult, CLASS_COLOR_PALETTE


# ───────────────────────────── Data Structures ──────────────────────────────

@dataclass
class TrackPoint:
    """A single observation of a tracked object at a specific frame."""
    frame_idx: int
    timestamp_seconds: float
    track_id: int
    class_name: str
    raw_class_name: str
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    center: Tuple[int, int]           # (cx, cy)
    confidence: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp_seconds, 4),
            "track_id": self.track_id,
            "class_name": self.class_name,
            "raw_class_name": self.raw_class_name,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class TrackedObject:
    """Represents an object tracked across multiple frames."""
    track_id: int
    class_name: str
    raw_class_name: str
    trajectory: List[TrackPoint] = field(default_factory=list)

    @property
    def display_label(self) -> str:
        """Human-readable label like 'Person #3' or 'Carton #7'."""
        pretty = self.class_name.replace("/", " / ").title()
        return f"{pretty} #{self.track_id}"

    @property
    def last_seen_frame(self) -> int:
        return self.trajectory[-1].frame_idx if self.trajectory else -1

    @property
    def last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self.trajectory[-1].bbox if self.trajectory else None

    @property
    def last_center(self) -> Optional[Tuple[int, int]]:
        return self.trajectory[-1].center if self.trajectory else None

    @property
    def last_confidence(self) -> float:
        return self.trajectory[-1].confidence if self.trajectory else 0.0

    @property
    def frame_count(self) -> int:
        """Number of frames this object has been observed."""
        return len(self.trajectory)

    @property
    def center_history(self) -> List[Tuple[int, int]]:
        """List of center positions over time (for trajectory visualisation)."""
        return [tp.center for tp in self.trajectory]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "display_label": self.display_label,
            "frame_count": self.frame_count,
            "last_seen_frame": self.last_seen_frame,
            "trajectory": [tp.to_dict() for tp in self.trajectory],
        }


@dataclass
class TrackingResult:
    """Output of a single tracking update: tracked objects and annotated frame."""
    frame_idx: int
    timestamp_seconds: float
    tracked_objects: List[TrackedObject] = field(default_factory=list)
    active_track_ids: List[int] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None

    @property
    def active_count(self) -> int:
        return len(self.active_track_ids)

    @property
    def counts_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for tid in self.active_track_ids:
            for obj in self.tracked_objects:
                if obj.track_id == tid:
                    counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
                    break
        return counts


# ─────────────────────────── Tracker Engine ─────────────────────────────────

class ObjectTracker:
    """
    Multi-object tracker wrapping supervision's ByteTrack.
    Provides persistent IDs, class-labelled tracks, and trajectory accumulation.

    Usage:
        tracker = ObjectTracker()
        for frame_idx, ts, frame in video_frames:
            det_result = detector.detect(frame, frame_idx, ts)
            trk_result = tracker.update(det_result, frame)
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ):
        """
        Initialises the ByteTrack tracker.

        Args:
            track_activation_threshold: Minimum detection confidence to start a track.
            lost_track_buffer: Frames to wait before deleting a lost track.
            minimum_matching_threshold: IoU threshold for matching detections to tracks.
            frame_rate: Expected video frame rate (used by ByteTrack internals).
            minimum_consecutive_frames: Consecutive detections required before a track is confirmed.
        """
        self._byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

        # Global track registry: track_id → TrackedObject
        self._tracked_objects: Dict[int, TrackedObject] = {}

        # Per-detection metadata cache for the current frame
        self._current_detections: List[Detection] = []

    @property
    def tracked_objects(self) -> Dict[int, TrackedObject]:
        """Returns the full registry of all tracked objects (active and lost)."""
        return dict(self._tracked_objects)

    @property
    def total_unique_tracks(self) -> int:
        """Total number of unique objects ever tracked."""
        return len(self._tracked_objects)

    def reset(self) -> None:
        """Resets all tracking state (useful when switching videos)."""
        self._byte_tracker.reset()
        self._tracked_objects.clear()
        self._current_detections.clear()

    def update(
        self,
        detection_result: DetectionResult,
        frame: np.ndarray,
        annotate: bool = True,
    ) -> TrackingResult:
        """
        Updates tracker with new detections from a single frame.

        Args:
            detection_result: Output from ObjectDetector.detect().
            frame: The original BGR frame (used for annotation).
            annotate: Whether to produce a visualised annotated frame.

        Returns:
            TrackingResult with updated tracked objects, active IDs, and optional annotation.
        """
        detections_list = detection_result.detections
        self._current_detections = detections_list

        if not detections_list:
            return TrackingResult(
                frame_idx=detection_result.frame_idx,
                timestamp_seconds=detection_result.timestamp_seconds,
                tracked_objects=list(self._tracked_objects.values()),
                active_track_ids=[],
                annotated_frame=frame.copy() if annotate else None,
            )

        # Build supervision Detections array from our Detection objects
        sv_detections = self._build_sv_detections(detections_list)

        # Run ByteTrack association
        sv_tracked = self._byte_tracker.update_with_detections(sv_detections)

        # Extract track IDs assigned by ByteTrack
        if sv_tracked.tracker_id is not None:
            track_ids = sv_tracked.tracker_id.tolist()
        else:
            track_ids = []

        active_ids: List[int] = []

        # Update our tracked object registry
        for i, track_id in enumerate(track_ids):
            bbox_xyxy = sv_tracked.xyxy[i].astype(int)
            x1, y1, x2, y2 = int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3])
            conf = float(sv_tracked.confidence[i]) if sv_tracked.confidence is not None else 0.0
            class_id = int(sv_tracked.class_id[i]) if sv_tracked.class_id is not None else -1

            # Find matching detection to get class name
            class_name, raw_class_name = self._resolve_class_name(
                bbox=(x1, y1, x2, y2),
                class_id=class_id,
                detections_list=detections_list,
            )

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            track_point = TrackPoint(
                frame_idx=detection_result.frame_idx,
                timestamp_seconds=detection_result.timestamp_seconds,
                track_id=track_id,
                class_name=class_name,
                raw_class_name=raw_class_name,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                confidence=conf,
            )

            if track_id not in self._tracked_objects:
                self._tracked_objects[track_id] = TrackedObject(
                    track_id=track_id,
                    class_name=class_name,
                    raw_class_name=raw_class_name,
                )

            self._tracked_objects[track_id].trajectory.append(track_point)
            active_ids.append(track_id)

        # Annotate if requested
        annotated = None
        if annotate:
            annotated = self._annotate_tracked_frame(frame, active_ids)

        return TrackingResult(
            frame_idx=detection_result.frame_idx,
            timestamp_seconds=detection_result.timestamp_seconds,
            tracked_objects=list(self._tracked_objects.values()),
            active_track_ids=active_ids,
            annotated_frame=annotated,
        )

    def _build_sv_detections(self, detections: List[Detection]) -> sv.Detections:
        """Converts our Detection list into a supervision.Detections object."""
        xyxy = np.array(
            [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] for d in detections],
            dtype=np.float32,
        )
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    def _resolve_class_name(
        self,
        bbox: Tuple[int, int, int, int],
        class_id: int,
        detections_list: List[Detection],
    ) -> Tuple[str, str]:
        """
        Resolves class_name and raw_class_name for a tracked detection
        by matching class_id and proximity.
        """
        # First try: match by class_id and IoU overlap
        best_iou = 0.0
        best_det = None

        for det in detections_list:
            if det.class_id == class_id:
                iou = self._iou(bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

        if best_det is not None:
            return best_det.class_name, best_det.raw_class_name

        # Fallback: any detection with highest IoU
        for det in detections_list:
            iou = self._iou(bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = det

        if best_det is not None:
            return best_det.class_name, best_det.raw_class_name

        return "unknown", "unknown"

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """Computes intersection-over-union between two (x1, y1, x2, y2) bounding boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0

        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _annotate_tracked_frame(
        self,
        frame: np.ndarray,
        active_ids: List[int],
        line_thickness: int = 2,
        font_scale: float = 0.5,
        trail_length: int = 30,
    ) -> np.ndarray:
        """
        Draws bounding boxes with track IDs, class labels, confidence,
        and trajectory trails on the frame.
        """
        annotated = frame.copy()

        for track_id in active_ids:
            obj = self._tracked_objects.get(track_id)
            if obj is None or not obj.trajectory:
                continue

            latest = obj.trajectory[-1]
            x1, y1, x2, y2 = latest.bbox
            color = CLASS_COLOR_PALETTE.get(
                obj.class_name,
                CLASS_COLOR_PALETTE.get("default", (180, 180, 180)),
            )

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            # Draw centroid
            cv2.circle(annotated, latest.center, 4, color, -1)

            # Draw trajectory trail
            centers = obj.center_history[-trail_length:]
            for i in range(1, len(centers)):
                alpha = i / len(centers)  # Fade from dim to bright
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(annotated, centers[i - 1], centers[i], trail_color, max(1, line_thickness - 1))

            # Label: "Person #3 87%"
            label = f"{obj.display_label} {latest.confidence * 100:.0f}%"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_y1 = max(0, y1 - text_h - 8)
            label_y2 = y1
            label_x2 = min(frame.shape[1], x1 + text_w + 10)

            # Label background
            cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)

            # Text colour (adaptive contrast)
            brightness = sum(color)
            text_color = (0, 0, 0) if brightness > 400 else (255, 255, 255)

            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return annotated
