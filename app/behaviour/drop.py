"""
Temporal rule detector for product drop events.
Detects rapid downward vertical pixel velocity followed by sudden ground impact deceleration
while unheld by any worker.
"""

from __future__ import annotations

from typing import Dict, List
import uuid

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class DropRule(BaseBehaviourRule):
    """
    Detects product dropping violations based on vertical descent velocity heuristics,
    subsequent ground impact arrest, and worker detachment.
    """

    name: str = "product_dropped"

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

        # Collect active workers for proximity and holding suppression
        active_workers = [
            obj for tid, obj in tracked_objects.items()
            if tid in active_track_ids and self.is_person(obj.class_name) and obj.last_bbox is not None
        ]

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or not self.is_package(obj.class_name):
                continue

            traj = obj.trajectory
            if len(traj) < config.min_trajectory_frames:
                continue

            # Inspect sliding window of recent observations
            window_size = min(len(traj), 12)
            recent_points = traj[-window_size:]

            # 1. Check for impact / deceleration at current frame
            # The object must be currently stationary or near-stationary at its lowest point
            stationary_frames = config.drop_impact_stationary_frames
            if len(recent_points) < stationary_frames + 2:
                continue

            last_pt = recent_points[-1]
            tail_points = recent_points[-stationary_frames:]
            y2_tail = [tp.bbox[3] for tp in tail_points]
            y2_variance = max(y2_tail) - min(y2_tail)

            # Tail must be arrested at bottom (little vertical motion)
            if y2_variance > 18:
                continue

            # 2. Check for rapid downward descent leading up to this rest
            # Search preceding points for peak downward velocity
            max_downward_speed = 0.0
            found_descent = False
            total_descent = 0.0

            preceding_points = recent_points[:-stationary_frames + 1]
            if len(preceding_points) >= 2:
                y_start = preceding_points[0].bbox[3]
                y_end = preceding_points[-1].bbox[3]
                total_descent = y_end - y_start

                for i in range(1, len(preceding_points)):
                    dt = preceding_points[i].timestamp_seconds - preceding_points[i - 1].timestamp_seconds
                    if dt <= 0:
                        dt = 1.0 / video_fps
                    dy = preceding_points[i].bbox[3] - preceding_points[i - 1].bbox[3]
                    speed = dy / dt
                    if speed > max_downward_speed:
                        max_downward_speed = speed

                if (
                    max_downward_speed >= config.drop_min_vertical_velocity_px_s
                    and total_descent >= config.drop_min_height_px
                ):
                    found_descent = True

            if not found_descent:
                continue

            # 3. False-positive check: Is a worker holding or lowering this package?
            carton_box = last_pt.bbox
            carton_center = last_pt.center
            held_by_worker = False

            for worker in active_workers:
                worker_box = worker.last_bbox
                if worker_box is None:
                    continue
                # Bounding box IoU check
                if self.compute_iou(carton_box, worker_box) >= config.worker_holding_iou_threshold:
                    held_by_worker = True
                    break
                # Centroid proximity check
                if worker.last_center and self.euclidean_dist(carton_center, worker.last_center) <= config.worker_holding_distance_px:
                    held_by_worker = True
                    break

            if held_by_worker:
                continue

            # 4. Trigger confirmed drop event
            event = BehaviourEvent(
                event_id=f"EVT-DROP-{obj.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=obj.track_id,
                display_label=obj.display_label,
                class_name=obj.class_name,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=min(0.98, max_downward_speed / (config.drop_min_vertical_velocity_px_s * 1.5)),
                bbox=last_pt.bbox,
                metrics={
                    "vertical_velocity_px_s": round(max_downward_speed, 1),
                    "drop_height_px": round(total_descent, 1),
                    "held_by_worker": False,
                },
            )
            events.append(event)

        return events
