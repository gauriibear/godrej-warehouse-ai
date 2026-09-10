"""
Behaviour Engine Coordinator for Godrej Warehouse AI.
Executes temporal kinematic and spatial reasoning rules across tracked object trajectories,
manages event cooldowns/deduplication, and provides visual alert rendering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.tracking.tracker import ObjectTracker, TrackedObject, TrackPoint


# ───────────────────────────── Configurations ───────────────────────────────

@dataclass
class BehaviourConfig:
    """
    Centralized configurable heuristics for warehouse behaviour rules.
    Values represent pixel/video coordinate heuristics calibrated for typical CCTV setups.
    """
    # General temporal settings
    cooldown_frames: int = 60                   # Frame window before same track can re-trigger same rule (2.0s @ 30fps)
    min_trajectory_frames: int = 4              # Minimum trajectory observations needed for kinematic estimation

    # Worker interaction & proximity heuristics
    worker_holding_iou_threshold: float = 0.08  # IoU above which an object is considered worker-held
    worker_holding_distance_px: float = 90.0    # Centroid distance (px) indicating worker manual holding reach
    worker_contact_distance_px: float = 80.0    # Proximity threshold (px) from worker foot to carton

    # Product Dropped heuristics
    drop_min_vertical_velocity_px_s: float = 350.0  # Min downward pixel velocity (px/s) for freefall
    drop_impact_stationary_frames: int = 3          # Consecutive rest frames required to confirm ground impact
    drop_min_height_px: float = 50.0                # Minimum vertical descent distance (px)

    # Product Thrown heuristics
    throw_min_horizontal_velocity_px_s: float = 300.0  # Min lateral pixel speed (px/s) for ballistic motion
    throw_min_airborne_frames: int = 5                 # Consecutive frames detached in flight

    # Product Dragged heuristics
    drag_min_horizontal_velocity_px_s: float = 80.0    # Min horizontal floor speed (px/s)
    drag_min_duration_frames: int = 10                 # Consecutive frames of sustained floor sliding
    drag_max_worker_distance_px: float = 180.0         # Max reach distance (px) to associated walking worker

    # Product Pushed / Kicked heuristics
    push_impulse_velocity_spike_px_s: float = 250.0    # Velocity surge from rest indicating an abrupt kick/push
    push_prior_rest_velocity_px_s: float = 40.0        # Max speed prior to impulse to qualify as resting

    # Product Rolled heuristics
    roll_min_horizontal_velocity_px_s: float = 75.0    # Min translational speed (px/s)
    roll_min_aspect_ratio_flips: int = 2               # Minimum wide/tall AR reversals to confirm tumbling

    # Stacking & Overhang heuristics
    stacking_max_offset_ratio: float = 0.35            # Max center-of-mass offset ratio before stack is deemed unstable
    stacking_min_duration_frames: int = 10             # Stationary frames required before flagging unstable stack
    pallet_overhang_max_ratio: float = 0.15            # Max allowed overhang width percentage outside pallet boundary

    # Product Outside Designated Area heuristics (Scenario #8)
    designated_zones: List[Tuple[int, int, int, int]] = field(
        default_factory=lambda: [(100, 150, 1180, 650)]
    )                                                  # Permitted storage/pallet/loading zone(s) (x1, y1, x2, y2)
    outside_area_min_duration_frames: int = 10         # Consecutive frames outside zone before violation triggers
    outside_area_margin_px: float = 20.0               # Boundary tolerance margin (px) to prevent edge jitter false alarms
    outside_area_prolonged_frames: int = 25            # Sustained duration threshold for risk escalation
    outside_area_high_distance_px: float = 80.0        # Pixel displacement threshold for severe pathway obstruction risk

    # Product Handled Without Required Equipment heuristics (Scenario #9)
    required_equipment_types: List[str] = field(
        default_factory=lambda: [
            "equipment/forklift",
            "forklift",
            "equipment/machinery",
            "trolley",
            "pallet_jack",
        ]
    )                                                  # Equipment classes that must be present during handling
    equipment_worker_handling_distance_px: float = 120.0  # Max distance between worker and carton for active handling
    equipment_presence_distance_px: float = 250.0         # Max proximity for equipment to be considered assisting
    equipment_min_duration_frames: int = 10               # Consecutive frames of manual unassisted handling
    equipment_prolonged_frames: int = 25                  # Frames of sustained unassisted handling for risk escalation

    # Unsafe Loading/Unloading Sequence heuristics (Scenario #10)
    loading_zones: List[Tuple[int, int, int, int]] = field(
        default_factory=lambda: [(100, 150, 1180, 650)]
    )                                                  # Permitted loading/unloading area(s)
    sequence_max_window_frames: int = 45               # Maximum frame window to evaluate the sequence
    sequence_min_approach_frames: int = 4              # Minimum motion frames required to establish approach phase
    sequence_approach_speed_px_s: float = 60.0         # Minimum velocity (px/s) for approach into loading zone
    sequence_min_stabilize_frames: int = 5             # Consecutive frames required for safe stabilization
    sequence_stabilize_max_speed_px_s: float = 35.0    # Maximum speed to qualify as stabilized positioning
    sequence_unsafe_action_speed_px_s: float = 180.0   # Velocity threshold indicating an abrupt action / release
    sequence_rapid_execution_frames: int = 15          # Fast out-of-order execution threshold for risk escalation


# ───────────────────────────── Behaviour Event ──────────────────────────────

@dataclass
class BehaviourEvent:
    """Represents a confirmed material handling violation event."""
    event_id: str                      # Unique identifier, e.g. "EVT-DROP-7-F42"
    rule_name: str                     # "product_dropped", "product_thrown", etc.
    track_id: int                      # Primary affected object track ID (e.g. carton ID)
    display_label: str                 # "Carton #7"
    secondary_track_id: Optional[int] = None  # Linked entity track ID (e.g. worker ID or pallet ID)
    secondary_label: Optional[str] = None     # "Person #3" or None
    class_name: str = "carton"
    frame_idx: int = 0
    timestamp_seconds: float = 0.0
    confidence: float = 0.85
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def human_readable_name(self) -> str:
        """User-friendly title for the violation."""
        return self.rule_name.replace("_", " ").title()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "rule_name": self.rule_name,
            "human_readable_name": self.human_readable_name,
            "track_id": self.track_id,
            "display_label": self.display_label,
            "secondary_track_id": self.secondary_track_id,
            "secondary_label": self.secondary_label,
            "class_name": self.class_name,
            "frame_idx": self.frame_idx,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
            "metrics": self.metrics,
        }


# ──────────────────────────── Base Rule Class ───────────────────────────────

class BaseBehaviourRule(ABC):
    """Abstract base class for all temporal and spatial behaviour rules."""

    name: str = "base_rule"

    @abstractmethod
    def evaluate(
        self,
        tracked_objects: Dict[int, TrackedObject],
        active_track_ids: List[int],
        current_frame_idx: int,
        timestamp: float,
        video_fps: float,
        config: BehaviourConfig,
    ) -> List[BehaviourEvent]:
        """
        Evaluates the current state of tracked objects and trajectories.
        Returns a list of detected violation events (if any).
        """
        pass

    @staticmethod
    def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """Intersection over Union between two bounding boxes (x1, y1, x2, y2)."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter <= 0:
            return 0.0

        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def euclidean_dist(pt_a: Tuple[int, int], pt_b: Tuple[int, int]) -> float:
        """Euclidean distance in pixel space."""
        return math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])

    @staticmethod
    def is_package(class_name: str) -> bool:
        """Checks if class represents a package or carton."""
        name = class_name.lower()
        return "carton" in name or "product" in name or "box" in name or "package" in name

    @staticmethod
    def is_person(class_name: str) -> bool:
        """Checks if class represents a person."""
        return "person" in class_name.lower()

    @staticmethod
    def is_pallet(class_name: str) -> bool:
        """Checks if class represents a pallet or rack."""
        name = class_name.lower()
        return "pallet" in name or "structure" in name or "rack" in name


# ─────────────────────────── Coordinator Engine ─────────────────────────────

class BehaviourEngine:
    """
    Coordinates evaluation of all registered material handling behaviour rules,
    enforces temporal deduplication windows, and handles visual alert drawing.
    """

    def __init__(self, config: Optional[BehaviourConfig] = None, rules: Optional[List[BaseBehaviourRule]] = None):
        self.config = config or BehaviourConfig()
        self.rules: List[BaseBehaviourRule] = []

        if rules is not None:
            self.rules = rules
        else:
            self._load_default_rules()

        # Cooldown tracker: (track_id, rule_name) -> last_fired_frame_idx
        self._recent_events: Dict[Tuple[int, str], int] = {}

        # All recorded events across the run
        self._event_history: List[BehaviourEvent] = []

        # Active visual alert buffer: list of (event, expire_frame)
        self._active_visual_alerts: List[Tuple[BehaviourEvent, int]] = []

    def _load_default_rules(self) -> None:
        """Loads and initializes standard behaviour rules."""
        try:
            from app.behaviour.drop import DropRule
            from app.behaviour.throw import ThrowRule
            from app.behaviour.drag import DragRule
            from app.behaviour.push import PushRule
            from app.behaviour.roll import RollRule
            from app.behaviour.stacking import StackingRule
            from app.behaviour.outside_area import OutsideDesignatedAreaRule
            from app.behaviour.equipment import NoRequiredEquipmentRule
            from app.behaviour.sequence import UnsafeLoadingSequenceRule

            self.rules = [
                DropRule(),
                ThrowRule(),
                DragRule(),
                PushRule(),
                RollRule(),
                StackingRule(),
                OutsideDesignatedAreaRule(),
                NoRequiredEquipmentRule(),
                UnsafeLoadingSequenceRule(),
            ]
        except ImportError:
            # Rules will be populated as they are defined
            self.rules = []

    @property
    def all_events(self) -> List[BehaviourEvent]:
        """Returns full list of confirmed behaviour events."""
        return list(self._event_history)

    def reset(self) -> None:
        """Resets engine state between video runs."""
        self._recent_events.clear()
        self._event_history.clear()
        self._active_visual_alerts.clear()

    def evaluate_frame(
        self,
        tracker: ObjectTracker,
        current_frame_idx: int,
        timestamp: float,
        fps: float = 30.0,
    ) -> List[BehaviourEvent]:
        """
        Evaluates all active rules across current tracked objects.
        Returns newly confirmed (non-deduplicated) events for this frame.
        """
        all_objects = tracker.tracked_objects
        # Determine active tracks at this frame
        active_ids = [
            tid for tid, obj in all_objects.items()
            if obj.last_seen_frame == current_frame_idx
        ]

        fps = max(1.0, float(fps))
        raw_events: List[BehaviourEvent] = []

        for rule in self.rules:
            try:
                rule_events = rule.evaluate(
                    tracked_objects=all_objects,
                    active_track_ids=active_ids,
                    current_frame_idx=current_frame_idx,
                    timestamp=timestamp,
                    video_fps=fps,
                    config=self.config,
                )
                if rule_events:
                    raw_events.extend(rule_events)
            except Exception:
                # Rule execution should never crash the tracking pipeline
                continue

        # Filter duplicates through cooldown window
        new_confirmed_events: List[BehaviourEvent] = []
        for event in raw_events:
            dedup_key = (event.track_id, event.rule_name)
            last_frame = self._recent_events.get(dedup_key, -999999)

            if (current_frame_idx - last_frame) >= self.config.cooldown_frames:
                self._recent_events[dedup_key] = current_frame_idx
                self._event_history.append(event)
                new_confirmed_events.append(event)
                # Keep active in visual alert buffer for 15 frames (0.5s @ 30fps)
                self._active_visual_alerts.append((event, current_frame_idx + 15))

        # Purge expired visual alerts
        self._active_visual_alerts = [
            (ev, exp) for ev, exp in self._active_visual_alerts
            if exp >= current_frame_idx
        ]

        return new_confirmed_events

    def annotate_events(
        self,
        frame: np.ndarray,
        events: List[BehaviourEvent],
        current_frame_idx: int,
        tracked_objects: Optional[Dict[int, TrackedObject]] = None,
    ) -> np.ndarray:
        """
        Draws prominent warning banners and highlighted red bounding boxes
        for active violations on the video frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Use current alerts from buffer
        current_alerts = [ev for ev, exp in self._active_visual_alerts if exp >= current_frame_idx]
        if not current_alerts and not events:
            return annotated

        # 1. Draw top alert banner
        active_event = current_alerts[-1] if current_alerts else events[-1]
        banner_height = 45
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 180), -1)  # Deep crimson
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        banner_text = (
            f"VIOLATION: {active_event.human_readable_name.upper()} | "
            f"{active_event.display_label} | Time: {active_event.timestamp_seconds:.2f}s"
        )
        cv2.putText(
            annotated,
            banner_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 2. Highlight bounding box of offending object in crimson
        for ev in current_alerts:
            # If tracked object is currently available, get latest box
            box = ev.bbox
            if tracked_objects and ev.track_id in tracked_objects:
                obj = tracked_objects[ev.track_id]
                if obj.last_bbox and obj.last_seen_frame == current_frame_idx:
                    box = obj.last_bbox

            x1, y1, x2, y2 = box
            # Red violation box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Sub-badge with metric
            metric_text = ""
            if "vertical_velocity_px_s" in ev.metrics:
                metric_text = f"Drop: {ev.metrics['vertical_velocity_px_s']:.0f} px/s"
            elif "horizontal_velocity_px_s" in ev.metrics:
                metric_text = f"Speed: {ev.metrics['horizontal_velocity_px_s']:.0f} px/s"
            elif "offset_ratio" in ev.metrics:
                metric_text = f"Offset: {ev.metrics['offset_ratio'] * 100:.0f}%"
            elif "overhang_ratio" in ev.metrics:
                metric_text = f"Overhang: {ev.metrics['overhang_ratio'] * 100:.0f}%"
            elif "distance_outside_px" in ev.metrics:
                metric_text = f"Outside: {ev.metrics['distance_outside_px']:.0f} px"
            elif "consecutive_handling_frames" in ev.metrics:
                metric_text = f"NoEquip: {ev.metrics['consecutive_handling_frames']}f"
            elif "sequence_duration_frames" in ev.metrics:
                metric_text = f"UnsafeSeq: {ev.metrics['sequence_duration_frames']}f"

            if metric_text:
                tag_y = min(h - 5, y2 + 18)
                cv2.rectangle(annotated, (x1, y2), (x1 + 140, tag_y + 4), (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    metric_text,
                    (x1 + 4, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return annotated
