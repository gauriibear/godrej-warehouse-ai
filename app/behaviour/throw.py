"""
Temporal rule detector for product throwing and ballistic motion events.
Detects significant horizontal pixel velocity combined with vertical arc kinematics
while detached from all personnel.
"""

from __future__ import annotations

import math
from typing import Dict, List

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class ThrowRule(BaseBehaviourRule):
    """
    Detects rejecting / throwing motion using release-separation and motion evidence,
    rather than requiring a near-perfect ballistic trajectory. The rule is designed to
    catch abrupt worker release, rapid forward motion, and short independent flight segments.
    """

    name: str = "product_thrown"

    @staticmethod
    def _window_metrics(window: List) -> Dict[str, float]:
        if len(window) < 2:
            return {
                "horizontal_disp": 0.0,
                "vertical_disp": 0.0,
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "accel": 0.0,
                "direction_change": 0.0,
            }

        speeds: List[float] = []
        horz_disp = abs(window[-1].center[0] - window[0].center[0])
        vert_disp = abs(window[-1].center[1] - window[0].center[1])

        for i in range(1, len(window)):
            prev = window[i - 1].center
            curr = window[i].center
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            dt = max(1e-3, window[i].timestamp_seconds - window[i - 1].timestamp_seconds)
            speed = math.hypot(dx, dy) / dt
            speeds.append(speed)

        max_speed = max(speeds) if speeds else 0.0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        # Estimate acceleration from the speed jump across the motion window.
        accel = 0.0
        if len(speeds) >= 2:
            accel = max(speeds) - min(speeds)

        direction_change = 0.0
        if len(window) >= 3:
            angles = []
            for i in range(1, len(window)):
                prev = window[i - 1].center
                curr = window[i].center
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                if abs(dx) > 0 or abs(dy) > 0:
                    angles.append(math.atan2(dy, dx))
            if len(angles) >= 2:
                diffs = [abs((a - b + math.pi) % (2 * math.pi) - math.pi) for a, b in zip(angles[:-1], angles[1:])]
                direction_change = sum(diffs) / len(diffs)

        return {
            "horizontal_disp": horz_disp,
            "vertical_disp": vert_disp,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "accel": accel,
            "direction_change": direction_change,
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
            if len(traj) < max(4, config.throw_min_airborne_frames):
                continue

            window = traj[-max(6, config.throw_min_airborne_frames):]
            features = self._window_metrics(window)
            horizontal_disp = features["horizontal_disp"]
            vertical_disp = features["vertical_disp"]
            avg_speed = features["avg_speed"]
            max_speed = features["max_speed"]
            accel = features["accel"]
            direction_change = features["direction_change"]

            if horizontal_disp < 30.0 and max_speed < config.throw_min_horizontal_velocity_px_s:
                continue

            if avg_speed < config.throw_min_horizontal_velocity_px_s * 0.7:
                continue

            # Reject ordinary drops/placements: they are mostly downward with low horizontal propulsion.
            if horizontal_disp < 35.0 and vertical_disp > 1.5 * horizontal_disp:
                continue

            last_pt = window[-1]
            carton_box = last_pt.bbox
            carton_center = last_pt.center

            nearest_worker = None
            nearest_worker_dist = float("inf")
            for worker in active_workers:
                worker_box = worker.last_bbox
                if worker_box is None:
                    continue
                if self.compute_iou(carton_box, worker_box) >= config.worker_holding_iou_threshold:
                    nearest_worker = worker
                    nearest_worker_dist = 0.0
                    break
                if worker.last_center is not None:
                    dist = self.euclidean_dist(carton_center, worker.last_center)
                    if dist < nearest_worker_dist:
                        nearest_worker_dist = dist
                        nearest_worker = worker

            worker_separation = 1.0
            if nearest_worker is not None:
                worker_separation = 0.0 if nearest_worker_dist <= config.worker_holding_distance_px else 1.0

            evidence = 0.0
            evidence += min(0.45, max_speed / (config.throw_min_horizontal_velocity_px_s * 2.5))
            evidence += min(0.20, max(0.0, horizontal_disp / 120.0))
            evidence += min(0.20, max(0.0, accel / 250.0))
            evidence += 0.10 if worker_separation > 0.5 else 0.0
            evidence += min(0.15, max(0.0, (math.pi / 2.0 - direction_change) / math.pi))

            if nearest_worker is not None and nearest_worker_dist <= config.worker_holding_distance_px:
                continue

            if evidence < 0.65:
                continue

            event = BehaviourEvent(
                event_id=f"EVT-THROW-{obj.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=obj.track_id,
                display_label=obj.display_label,
                class_name=obj.class_name,
                secondary_track_id=nearest_worker.track_id if nearest_worker is not None else None,
                secondary_label=nearest_worker.display_label if nearest_worker is not None else None,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=round(min(0.97, evidence), 3),
                bbox=last_pt.bbox,
                metrics={
                    "horizontal_velocity_px_s": round(max_speed, 1),
                    "avg_velocity_px_s": round(avg_speed, 1),
                    "horizontal_displacement_px": round(horizontal_disp, 1),
                    "vertical_displacement_px": round(vertical_disp, 1),
                    "acceleration_px_s": round(accel, 1),
                    "direction_change_rad": round(direction_change, 3),
                    "worker_distance_px": round(nearest_worker_dist, 1) if nearest_worker is not None else None,
                    "evidence_score": round(evidence, 3),
                },
            )
            events.append(event)

        return events
