"""
Temporal rule detector for product rolling and aspect-ratio rotation events.
Detects translating packages exhibiting cyclical bounding-box aspect ratio oscillations characteristic of tumbling.
"""

from __future__ import annotations

from typing import Dict, List

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class RollRule(BaseBehaviourRule):
    """
    Detects product rolling violations based on lateral floor translation
    concurrent with periodic aspect ratio reversals.
    """

    name: str = "product_rolled"

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

        # Find active workers for suppression
        active_workers = [
            obj for tid, obj in tracked_objects.items()
            if tid in active_track_ids and self.is_person(obj.class_name) and obj.last_bbox is not None
        ]

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or not self.is_package(obj.class_name):
                continue

            traj = obj.trajectory
            window_size = min(len(traj), 20)
            if window_size < 8:
                continue

            window = traj[-window_size:]
            dt = window[-1].timestamp_seconds - window[0].timestamp_seconds
            if dt <= 0:
                dt = (window_size - 1) / video_fps

            # 1. Check translational velocity across the floor
            dx = window[-1].center[0] - window[0].center[0]
            h_speed = abs(dx) / dt
            if h_speed < config.roll_min_horizontal_velocity_px_s:
                continue

            # 2. Check for aspect ratio flips across the window
            # Compute aspect ratio (w / h) for each observation
            aspect_ratios = []
            for tp in window:
                w = max(1, tp.bbox[2] - tp.bbox[0])
                h = max(1, tp.bbox[3] - tp.bbox[1])
                aspect_ratios.append(w / h)

            # Count flips between wide (ar >= 1.15) and tall (ar <= 0.85) states
            flips = 0
            current_state = None  # "wide" or "tall"

            for ar in aspect_ratios:
                if ar >= 1.15:
                    new_state = "wide"
                elif ar <= 0.85:
                    new_state = "tall"
                else:
                    new_state = None

                if new_state is not None:
                    if current_state is not None and new_state != current_state:
                        flips += 1
                    current_state = new_state

            if flips < config.roll_min_aspect_ratio_flips:
                continue

            # 3. Suppress if held by worker
            last_pt = window[-1]
            carton_box = last_pt.bbox
            held_by_worker = False

            for worker in active_workers:
                worker_box = worker.last_bbox
                if worker_box is not None:
                    if self.compute_iou(carton_box, worker_box) >= config.worker_holding_iou_threshold:
                        held_by_worker = True
                        break

            if held_by_worker:
                continue

            # Confirmed rolling violation event
            event = BehaviourEvent(
                event_id=f"EVT-ROLL-{obj.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=obj.track_id,
                display_label=obj.display_label,
                class_name=obj.class_name,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=min(0.92, 0.70 + flips * 0.1),
                bbox=last_pt.bbox,
                metrics={
                    "horizontal_velocity_px_s": round(h_speed, 1),
                    "aspect_ratio_flips": flips,
                    "window_frames": window_size,
                },
            )
            events.append(event)

        return events
