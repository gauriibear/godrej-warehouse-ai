"""
Temporal rule detector for unsafe loading and unloading sequences.
Detects when a material handling or placement action is executed out of order,
specifically occurring before the required safe positioning/stabilization step.
Godrej Scenario #10: Unsafe Loading/Unloading Sequence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class UnsafeLoadingSequenceRule(BaseBehaviourRule):
    """
    Detects unsafe loading/unloading sequences where material handling actions
    (abrupt transfer, drop, impulse, or release) occur out of sequence before
    the required safe positioning and stabilization step has been completed.
    """

    name: str = "unsafe_loading_unloading_sequence"

    def __init__(
        self,
        loading_zones: Optional[List[Tuple[int, int, int, int]]] = None,
        max_sequence_window_frames: Optional[int] = None,
        min_approach_frames: Optional[int] = None,
        approach_speed_threshold_px_s: Optional[float] = None,
        min_stabilize_frames: Optional[int] = None,
        stabilize_max_speed_px_s: Optional[float] = None,
        unsafe_action_speed_px_s: Optional[float] = None,
    ):
        """
        Args:
            loading_zones: Optional list of rectangular loading bay/zone bounds (x1, y1, x2, y2).
            max_sequence_window_frames: Maximum frame window for evaluating the sequence.
            min_approach_frames: Minimum frames of active motion required to establish approach phase.
            approach_speed_threshold_px_s: Minimum pixel velocity during approach into loading area.
            min_stabilize_frames: Consecutive frames required for safe stabilization before handling.
            stabilize_max_speed_px_s: Maximum velocity to qualify as stabilized positioning.
            unsafe_action_speed_px_s: Velocity threshold indicating an abrupt action / release.
        """
        self.custom_zones = loading_zones
        self.custom_max_window = max_sequence_window_frames
        self.custom_min_approach = min_approach_frames
        self.custom_approach_speed = approach_speed_threshold_px_s
        self.custom_min_stabilize = min_stabilize_frames
        self.custom_stabilize_speed = stabilize_max_speed_px_s
        self.custom_action_speed = unsafe_action_speed_px_s

    @staticmethod
    def is_point_in_zone(point: Tuple[int, int], zone: Tuple[int, int, int, int], margin: float = 30.0) -> bool:
        """Checks if a point (x, y) lies inside or within margin of a rectangular zone."""
        x, y = point
        zx1, zy1, zx2, zy2 = zone
        return (zx1 - margin) <= x <= (zx2 + margin) and (zy1 - margin) <= y <= (zy2 + margin)

    def evaluate(
        self,
        tracked_objects: Dict[int, TrackedObject],
        active_track_ids: List[int],
        current_frame_idx: int,
        timestamp: float,
        video_fps: float,
        config: BehaviourConfig,
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []

        # Resolve configurable heuristics
        zones = (
            self.custom_zones
            if self.custom_zones is not None
            else getattr(config, "loading_zones", [(100, 150, 1180, 650)])
        )
        max_window = (
            self.custom_max_window
            if self.custom_max_window is not None
            else getattr(config, "sequence_max_window_frames", 45)
        )
        min_approach = (
            self.custom_min_approach
            if self.custom_min_approach is not None
            else getattr(config, "sequence_min_approach_frames", 4)
        )
        approach_speed_thresh = (
            self.custom_approach_speed
            if self.custom_approach_speed is not None
            else getattr(config, "sequence_approach_speed_px_s", 60.0)
        )
        min_stabilize = (
            self.custom_min_stabilize
            if self.custom_min_stabilize is not None
            else getattr(config, "sequence_min_stabilize_frames", 5)
        )
        stabilize_speed_thresh = (
            self.custom_stabilize_speed
            if self.custom_stabilize_speed is not None
            else getattr(config, "sequence_stabilize_max_speed_px_s", 35.0)
        )
        action_speed_thresh = (
            self.custom_action_speed
            if self.custom_action_speed is not None
            else getattr(config, "sequence_unsafe_action_speed_px_s", 180.0)
        )

        # Collect active workers for secondary association
        active_workers = [
            obj for tid, obj in tracked_objects.items()
            if tid in active_track_ids and self.is_person(obj.class_name) and obj.last_bbox is not None
        ]

        for tid in active_track_ids:
            carton = tracked_objects.get(tid)
            if carton is None or not self.is_package(carton.class_name):
                continue

            traj = carton.trajectory
            if len(traj) < (min_approach + 2):
                continue

            # Limit evaluation to the temporal window
            window = traj[-max_window:]
            win_len = len(window)
            if win_len < (min_approach + 2):
                continue

            # Check spatial zone applicability
            if zones:
                in_any_zone = any(
                    self.is_point_in_zone(pt.center, zone)
                    for pt in window
                    for zone in zones
                )
                if not in_any_zone:
                    continue

            # Compute step velocities across the window
            velocities: List[float] = []
            vert_velocities: List[float] = []
            for i in range(1, win_len):
                prev_p = window[i - 1]
                curr_p = window[i]
                dt = max(0.001, curr_p.timestamp_seconds - prev_p.timestamp_seconds)
                dx = curr_p.center[0] - prev_p.center[0]
                dy = curr_p.center[1] - prev_p.center[1]
                speed = math.hypot(dx, dy) / dt
                vy = dy / dt
                velocities.append(speed)
                vert_velocities.append(vy)

            num_steps = len(velocities)
            if num_steps < min_approach + 1:
                continue

            # 1. Action phase detection (occurring at or near current frame: last 3 steps)
            action_idx: Optional[int] = None
            action_type: str = "abrupt_handling"
            action_speed: float = 0.0

            action_search_range = range(max(0, num_steps - 3), num_steps)
            for step_i in action_search_range:
                spd = velocities[step_i]
                vy = vert_velocities[step_i]
                if spd >= action_speed_thresh or vy >= action_speed_thresh:
                    action_idx = step_i
                    action_speed = spd
                    action_type = "abrupt_release_or_movement"
                    break

            if action_idx is None:
                # No active abrupt action in current/recent frames
                continue

            # 2. Prior Approach phase detection
            # Must find an approach period preceding the action within the window
            approach_start_idx: Optional[int] = None
            consecutive_approach = 0

            for i in range(action_idx):
                if velocities[i] >= approach_speed_thresh:
                    consecutive_approach += 1
                    if consecutive_approach >= min_approach:
                        approach_start_idx = i - min_approach + 1
                else:
                    consecutive_approach = 0

            if approach_start_idx is None:
                # Isolated action without the required preceding approach phase
                continue

            # 3. Check for the Required Safe Step: Stabilized Positioning
            # In a safe sequence, between approach and action, the product must come to rest
            # for at least min_stabilize consecutive frames.
            consecutive_stable = 0
            max_stable_seen = 0

            # Scan the interval between approach completion and action
            interval_start = approach_start_idx + min_approach
            for i in range(interval_start, action_idx):
                if velocities[i] <= stabilize_speed_thresh:
                    consecutive_stable += 1
                    max_stable_seen = max(max_stable_seen, consecutive_stable)
                else:
                    consecutive_stable = 0

            if max_stable_seen >= min_stabilize:
                # The product was properly stabilized before handling -> SAFE SEQUENCE
                continue

            # 4. Find closest interacting worker (if any)
            interacting_worker: Optional[TrackedObject] = None
            min_w_dist = float("inf")
            c_box = carton.last_bbox
            c_center = carton.last_center
            if c_box and c_center:
                for worker in active_workers:
                    if worker.last_bbox and worker.last_center:
                        w_dist = min(
                            self.box_distance(c_box, worker.last_bbox),
                            self.euclidean_dist(c_center, worker.last_center),
                        )
                        if w_dist <= 160.0 and w_dist < min_w_dist:
                            min_w_dist = w_dist
                            interacting_worker = worker

            # 5. Build confirmed UnsafeLoadingSequence event
            seq_duration = (win_len - 1) - approach_start_idx
            confidence = min(0.95, max(0.72, 0.80 + (action_speed / (action_speed_thresh * 2.0)) * 0.15))

            steps_observed = [
                "approach_loading_zone",
                "abrupt_handling_without_stabilization",
            ]
            skipped_step = "stabilized_positioning"

            event = BehaviourEvent(
                event_id=f"EVT-UNSAFESEQ-{carton.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=carton.track_id,
                display_label=carton.display_label,
                secondary_track_id=interacting_worker.track_id if interacting_worker else None,
                secondary_label=interacting_worker.display_label if interacting_worker else None,
                class_name=carton.class_name,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=round(confidence, 3),
                bbox=carton.last_bbox or (0, 0, 0, 0),
                metrics={
                    "sequence_duration_frames": seq_duration,
                    "sequence_steps": steps_observed,
                    "skipped_step": skipped_step,
                    "action_speed_px_s": round(action_speed, 1),
                    "action_type": action_type,
                    "max_stable_frames_observed": max_stable_seen,
                    "required_stable_frames": min_stabilize,
                    "window_frames": max_window,
                    "explanation": (
                        "An unsafe loading/unloading sequence was detected because the "
                        "material handling action occurred before the required safe positioning step."
                    ),
                    "recommendation": (
                        "Follow the prescribed loading/unloading sequence and ensure the "
                        "product is safely positioned before continuing the next handling step."
                    ),
                },
            )
            events.append(event)

        return events

    @staticmethod
    def box_distance(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """Computes minimum Euclidean distance between two bounding boxes."""
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        dx = max(0, max(xa1 - xb2, xb1 - xa2))
        dy = max(0, max(ya1 - yb2, yb1 - ya2))
        return math.hypot(dx, dy)
