"""
Temporal rule detector for product dragging on floor events.
Detects sustained horizontal floor sliding in close proximity to a moving worker without lifting.
"""

from __future__ import annotations

from typing import Dict, List

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class DragRule(BaseBehaviourRule):
    """
    Detects product dragging violations based on floor-level horizontal velocity,
    stable ground contact, and proximity to an active walking worker.
    """

    name: str = "product_dragged"

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

        # Find active workers
        active_workers = [
            obj for tid, obj in tracked_objects.items()
            if tid in active_track_ids and self.is_person(obj.class_name) and obj.last_bbox is not None
        ]

        min_duration = config.drag_min_duration_frames

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or not self.is_package(obj.class_name):
                continue

            traj = obj.trajectory
            if len(traj) < min_duration:
                continue

            window = traj[-min_duration:]
            dt = window[-1].timestamp_seconds - window[0].timestamp_seconds
            if dt <= 0:
                dt = (min_duration - 1) / video_fps

            # 1. Floor contact check: y2 vertical variance must be small (sliding on floor)
            y2_vals = [tp.bbox[3] for tp in window]
            y2_variance = max(y2_vals) - min(y2_vals)
            if y2_variance > 25.0:
                continue

            # 2. Sustained horizontal sliding speed
            dx = window[-1].center[0] - window[0].center[0]
            h_speed = abs(dx) / dt
            if h_speed < config.drag_min_horizontal_velocity_px_s:
                continue

            # 3. Linked worker proximity check
            last_pt = window[-1]
            carton_center = last_pt.center
            carton_y = carton_center[1]

            linked_worker = None
            closest_dist = float("inf")

            for worker in active_workers:
                worker_box = worker.last_bbox
                worker_center = worker.last_center
                if worker_box is None or worker_center is None:
                    continue

                dist = self.euclidean_dist(carton_center, worker_center)
                if dist <= config.drag_max_worker_distance_px:
                    # Verify carton is positioned low relative to worker (not carried at waist/chest)
                    worker_h = max(1, worker_box[3] - worker_box[1])
                    if carton_y >= (worker_box[1] + 0.3 * worker_h):
                        if dist < closest_dist:
                            closest_dist = dist
                            linked_worker = worker

            if linked_worker is None:
                continue

            # Confirmed dragging event
            event = BehaviourEvent(
                event_id=f"EVT-DRAG-{obj.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=obj.track_id,
                display_label=obj.display_label,
                secondary_track_id=linked_worker.track_id,
                secondary_label=linked_worker.display_label,
                class_name=obj.class_name,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=min(0.95, h_speed / (config.drag_min_horizontal_velocity_px_s * 1.5)),
                bbox=last_pt.bbox,
                metrics={
                    "horizontal_velocity_px_s": round(h_speed, 1),
                    "duration_frames": min_duration,
                    "worker_distance_px": round(closest_dist, 1),
                },
            )
            events.append(event)

        return events
