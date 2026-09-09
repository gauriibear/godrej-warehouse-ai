"""
Temporal rule detector for product throwing and ballistic motion events.
Detects significant horizontal pixel velocity combined with vertical arc kinematics
while detached from all personnel.
"""

from __future__ import annotations

from typing import Dict, List

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class ThrowRule(BaseBehaviourRule):
    """
    Detects product throwing violations based on ballistic horizontal speed heuristics,
    vertical curvature, and sustained worker detachment.
    """

    name: str = "product_thrown"

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

        # Active workers
        active_workers = [
            obj for tid, obj in tracked_objects.items()
            if tid in active_track_ids and self.is_person(obj.class_name) and obj.last_bbox is not None
        ]

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or not self.is_package(obj.class_name):
                continue

            traj = obj.trajectory
            min_frames = config.throw_min_airborne_frames
            if len(traj) < min_frames:
                continue

            window = traj[-min_frames:]
            dt = window[-1].timestamp_seconds - window[0].timestamp_seconds
            if dt <= 0:
                dt = (min_frames - 1) / video_fps

            dx = window[-1].center[0] - window[0].center[0]
            dy = window[-1].center[1] - window[0].center[1]
            horizontal_speed = abs(dx) / dt

            # Must satisfy minimum horizontal speed heuristic
            if horizontal_speed < config.throw_min_horizontal_velocity_px_s:
                continue

            # Must have vertical movement or acceleration (ballistic arc)
            # Check for downward acceleration or non-trivial vertical travel
            vertical_travel = abs(dy)
            # Also check intermediate vertical velocities
            mid = len(window) // 2
            dt1 = max(0.001, window[mid].timestamp_seconds - window[0].timestamp_seconds)
            dt2 = max(0.001, window[-1].timestamp_seconds - window[mid].timestamp_seconds)
            vy1 = (window[mid].center[1] - window[0].center[1]) / dt1
            vy2 = (window[-1].center[1] - window[mid].center[1]) / dt2
            ay = (vy2 - vy1) / (dt1 + dt2)

            is_ballistic = (ay > -50.0) or (vertical_travel >= 15.0)
            if not is_ballistic:
                continue

            # Check that carton is detached from all workers across the flight window
            last_pt = window[-1]
            carton_box = last_pt.bbox
            carton_center = last_pt.center
            near_worker = False

            for worker in active_workers:
                worker_box = worker.last_bbox
                if worker_box is None:
                    continue
                if self.compute_iou(carton_box, worker_box) >= config.worker_holding_iou_threshold:
                    near_worker = True
                    break
                if worker.last_center and self.euclidean_dist(carton_center, worker.last_center) <= config.worker_holding_distance_px:
                    near_worker = True
                    break

            if near_worker:
                continue

            # Trigger confirmed throw event
            event = BehaviourEvent(
                event_id=f"EVT-THROW-{obj.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=obj.track_id,
                display_label=obj.display_label,
                class_name=obj.class_name,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=min(0.95, horizontal_speed / (config.throw_min_horizontal_velocity_px_s * 1.4)),
                bbox=last_pt.bbox,
                metrics={
                    "horizontal_velocity_px_s": round(horizontal_speed, 1),
                    "vertical_acceleration_px_s2": round(ay, 1),
                    "airborne_frames": min_frames,
                },
            )
            events.append(event)

        return events
