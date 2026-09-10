"""
Temporal rule detector for product dragging on floor events.
Detects sustained horizontal floor sliding in close proximity to a moving worker without lifting.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class DragRule(BaseBehaviourRule):
    """
    Detects product dragging violations using multi-frame temporal and spatial evidence:
    - sustained lateral motion across multiple frames
    - ground-level trajectory with limited vertical variation
    - directional consistency over the movement window
    - optional worker association for support
    - rejection of carried, stationary and short random displacements
    """

    name: str = "product_dragged"

    @staticmethod
    def _motion_features(window: List) -> Dict[str, float]:
        if len(window) < 2:
            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "horizontal_disp": 0.0,
                "vertical_disp": 0.0,
                "direction_consistency": 0.0,
                "y_variance": 0.0,
                "duration": 0.0,
            }

        points = window
        speeds: List[float] = []
        angles: List[float] = []

        for i in range(1, len(points)):
            prev = points[i - 1].center
            curr = points[i].center
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            dt = max(1e-3, points[i].timestamp_seconds - points[i - 1].timestamp_seconds)
            speed = math.hypot(dx, dy) / dt
            speeds.append(speed)
            if abs(dx) > 0 or abs(dy) > 0:
                angles.append(math.atan2(dy, dx))

        horizontal_disp = abs(points[-1].center[0] - points[0].center[0])
        vertical_disp = abs(points[-1].center[1] - points[0].center[1])
        y_vals = [tp.bbox[3] for tp in points]
        y_variance = max(y_vals) - min(y_vals)

        direction_consistency = 1.0
        if len(angles) >= 2:
            diffs = []
            for j in range(1, len(angles)):
                delta = abs((angles[j] - angles[j - 1] + math.pi) % (2 * math.pi) - math.pi)
                diffs.append(delta)
            if diffs:
                mean_delta = sum(diffs) / len(diffs)
                direction_consistency = max(0.0, 1.0 - (mean_delta / math.pi))

        total_dt = max(1e-3, points[-1].timestamp_seconds - points[0].timestamp_seconds)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        return {
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "horizontal_disp": horizontal_disp,
            "vertical_disp": vertical_disp,
            "direction_consistency": direction_consistency,
            "y_variance": y_variance,
            "duration": total_dt,
        }

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

        active_workers = [
            obj for tid, obj in tracked_objects.items()
            if tid in active_track_ids and self.is_person(obj.class_name) and obj.last_bbox is not None
        ]

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or not self.is_package(obj.class_name):
                continue

            traj = obj.trajectory
            if len(traj) < max(6, config.drag_min_duration_frames):
                continue

            window = traj[-max(8, config.drag_min_duration_frames):]
            features = self._motion_features(window)

            if features["horizontal_disp"] < 25.0:
                continue

            if features["y_variance"] > 30.0:
                continue

            avg_speed = features["avg_speed"]
            max_speed = features["max_speed"]
            horizontal_speed = features["horizontal_disp"] / max(features["duration"], 1e-3)
            direction_consistency = features["direction_consistency"]

            if horizontal_speed < config.drag_min_horizontal_velocity_px_s and avg_speed < config.drag_min_horizontal_velocity_px_s:
                continue

            if direction_consistency < 0.45:
                continue

            last_pt = window[-1]
            carton_center = last_pt.center
            carton_y = last_pt.center[1]

            linked_worker = None
            closest_dist = float("inf")
            worker_support = 0.0

            for worker in active_workers:
                worker_box = worker.last_bbox
                worker_center = worker.last_center
                if worker_box is None or worker_center is None:
                    continue

                dist = self.euclidean_dist(carton_center, worker_center)
                if dist <= config.drag_max_worker_distance_px:
                    worker_h = max(1, worker_box[3] - worker_box[1])
                    if carton_y >= (worker_box[1] + 0.3 * worker_h):
                        if dist < closest_dist:
                            closest_dist = dist
                            linked_worker = worker

            worker_support = 1.0 if linked_worker is not None else 0.25
            if linked_worker is None and horizontal_speed < config.drag_min_horizontal_velocity_px_s * 1.8:
                continue

            final_evidence = min(
                0.99,
                0.45
                + min(0.35, max(0.0, horizontal_speed / (config.drag_min_horizontal_velocity_px_s * 4.0)))
                + min(0.20, max(0.0, direction_consistency * 0.2))
                + (0.15 if linked_worker is not None else 0.0)
            )

            if final_evidence < 0.55:
                continue

            event = BehaviourEvent(
                event_id=f"EVT-DRAG-{obj.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=obj.track_id,
                display_label=obj.display_label,
                secondary_track_id=linked_worker.track_id if linked_worker is not None else None,
                secondary_label=linked_worker.display_label if linked_worker is not None else None,
                class_name=obj.class_name,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=round(min(0.97, final_evidence), 3),
                bbox=last_pt.bbox,
                metrics={
                    "horizontal_velocity_px_s": round(horizontal_speed, 1),
                    "avg_velocity_px_s": round(avg_speed, 1),
                    "max_velocity_px_s": round(max_speed, 1),
                    "duration_frames": len(window),
                    "direction_consistency": round(direction_consistency, 3),
                    "horizontal_displacement_px": round(features["horizontal_disp"], 1),
                    "worker_distance_px": round(closest_dist, 1) if linked_worker is not None else None,
                    "evidence_score": round(final_evidence, 3),
                },
            )
            events.append(event)

        return events
