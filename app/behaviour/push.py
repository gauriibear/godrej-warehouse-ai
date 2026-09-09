"""
Temporal rule detector for product pushing and ground kick events.
Detects sudden impulse acceleration spikes on floor-resting packages initiated by worker foot contact.
"""

from __future__ import annotations

from typing import Dict, List

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class PushRule(BaseBehaviourRule):
    """
    Detects product pushing and ground kick violations based on impulse velocity spikes from rest
    occurring in immediate proximity to a worker's foot contact point.
    """

    name: str = "product_pushed"

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

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or not self.is_package(obj.class_name):
                continue

            traj = obj.trajectory
            if len(traj) < 4:
                continue

            # Look at recent frames: [t-3, t-2, t-1, t]
            window = traj[-4:]
            last_pt = window[-1]

            # 1. Compute velocity across last step (t-1 -> t)
            dt_curr = max(0.001, last_pt.timestamp_seconds - window[-2].timestamp_seconds)
            dx_curr = last_pt.center[0] - window[-2].center[0]
            dy_curr = last_pt.center[1] - window[-2].center[1]
            current_speed = (dx_curr**2 + dy_curr**2)**0.5 / dt_curr

            if current_speed < config.push_impulse_velocity_spike_px_s:
                continue

            # 2. Verify previous state was stationary / low-speed
            dt_prev = max(0.001, window[-2].timestamp_seconds - window[0].timestamp_seconds)
            dx_prev = window[-2].center[0] - window[0].center[0]
            dy_prev = window[-2].center[1] - window[0].center[1]
            prior_speed = (dx_prev**2 + dy_prev**2)**0.5 / dt_prev

            if prior_speed > config.push_prior_rest_velocity_px_s:
                continue

            # 3. Contact initiation check: was a worker foot close to this carton?
            carton_box = last_pt.bbox
            kicking_worker = None
            min_foot_dist = float("inf")

            for worker in active_workers:
                worker_box = worker.last_bbox
                if worker_box is None:
                    continue

                # Worker foot point is bottom-center of worker box
                w_foot = ((worker_box[0] + worker_box[2]) // 2, worker_box[3])
                # Compute distance from foot to carton center or edge
                dist = self.euclidean_dist(w_foot, last_pt.center)

                if dist <= config.worker_contact_distance_px:
                    # Make sure worker is not holding carton in arms
                    if self.compute_iou(carton_box, worker_box) < config.worker_holding_iou_threshold:
                        if dist < min_foot_dist:
                            min_foot_dist = dist
                            kicking_worker = worker

            if kicking_worker is None:
                continue

            # Confirmed push / kick event
            event = BehaviourEvent(
                event_id=f"EVT-PUSH-{obj.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=obj.track_id,
                display_label=obj.display_label,
                secondary_track_id=kicking_worker.track_id,
                secondary_label=kicking_worker.display_label,
                class_name=obj.class_name,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=min(0.95, current_speed / (config.push_impulse_velocity_spike_px_s * 1.5)),
                bbox=last_pt.bbox,
                metrics={
                    "impulse_velocity_px_s": round(current_speed, 1),
                    "prior_velocity_px_s": round(prior_speed, 1),
                    "foot_distance_px": round(min_foot_dist, 1),
                },
            )
            events.append(event)

        return events
