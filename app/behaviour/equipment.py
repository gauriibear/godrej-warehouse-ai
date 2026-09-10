"""
Spatial and temporal rule detector for products handled without required equipment.
Detects when a worker manually manipulates a product/carton without the presence
of designated handling equipment (e.g. forklift, pallet jack, trolley).
Godrej Scenario #9: Product Handled Without Required Equipment.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class NoRequiredEquipmentRule(BaseBehaviourRule):
    """
    Detects when a package/carton is being actively handled by a worker
    without the required handling equipment present in the handling zone.
    """

    name: str = "product_handled_without_required_equipment"

    def __init__(
        self,
        required_equipment: Optional[List[str]] = None,
        max_handling_distance_px: Optional[float] = None,
        equipment_proximity_distance_px: Optional[float] = None,
        min_duration_frames: Optional[int] = None,
    ):
        """
        Args:
            required_equipment: Optional list of class names (e.g. ["forklift", "equipment/forklift"]).
            max_handling_distance_px: Max distance between worker and carton to qualify as manual handling.
            equipment_proximity_distance_px: Max distance from carton/worker for equipment to be considered assisting.
            min_duration_frames: Number of consecutive frames of manual handling required before triggering.
        """
        self.custom_equipment = required_equipment
        self.custom_handling_distance = max_handling_distance_px
        self.custom_equipment_distance = equipment_proximity_distance_px
        self.custom_min_duration = min_duration_frames

    @staticmethod
    def is_matching_equipment(class_name: str, required_types: List[str]) -> bool:
        """Checks if an object class matches any required equipment pattern."""
        cls_lower = class_name.lower()
        for req in required_types:
            req_lower = req.lower()
            if req_lower in cls_lower or cls_lower in req_lower:
                return True
        return False

    @staticmethod
    def box_distance(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        """
        Computes minimum Euclidean distance between two bounding boxes (0 if overlapping).
        """
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b

        dx = max(0, max(xa1 - xb2, xb1 - xa2))
        dy = max(0, max(ya1 - yb2, yb1 - ya2))
        return math.hypot(dx, dy)

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

        # Resolve configurable thresholds
        req_equipment = (
            self.custom_equipment
            if self.custom_equipment is not None
            else getattr(
                config,
                "required_equipment_types",
                ["equipment/forklift", "forklift", "equipment/machinery", "trolley", "pallet_jack"],
            )
        )
        max_handling_dist = (
            self.custom_handling_distance
            if self.custom_handling_distance is not None
            else getattr(config, "equipment_worker_handling_distance_px", 120.0)
        )
        equip_prox_dist = (
            self.custom_equipment_distance
            if self.custom_equipment_distance is not None
            else getattr(config, "equipment_presence_distance_px", 250.0)
        )
        min_duration = (
            self.custom_min_duration
            if self.custom_min_duration is not None
            else getattr(config, "equipment_min_duration_frames", 10)
        )

        # Collect active cartons, workers, and potential handling equipment
        active_cartons: List[TrackedObject] = []
        active_workers: List[TrackedObject] = []
        active_equipment: List[TrackedObject] = []

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or obj.last_bbox is None:
                continue
            if self.is_package(obj.class_name):
                active_cartons.append(obj)
            elif self.is_person(obj.class_name):
                active_workers.append(obj)
            elif self.is_matching_equipment(obj.class_name, req_equipment):
                active_equipment.append(obj)

        if not active_cartons or not active_workers:
            # Requires both a carton and an active worker for manual handling
            return events

        for carton in active_cartons:
            c_traj = carton.trajectory
            if len(c_traj) < min_duration:
                continue

            c_box = carton.last_bbox
            c_center = carton.last_center
            if c_box is None or c_center is None:
                continue

            # 1. Check if any required equipment is in proximity to the carton at current frame
            equipment_present = False
            for eq in active_equipment:
                if eq.last_bbox is not None and eq.last_center is not None:
                    dist_to_carton = self.box_distance(c_box, eq.last_bbox)
                    center_dist = self.euclidean_dist(c_center, eq.last_center)
                    if dist_to_carton <= equip_prox_dist or center_dist <= equip_prox_dist:
                        equipment_present = True
                        break

            if equipment_present:
                # Equipment is assisting or nearby; no violation
                continue

            # 2. Find interacting worker in proximity to this carton
            interacting_worker: Optional[TrackedObject] = None
            min_worker_dist = float("inf")

            for worker in active_workers:
                if worker.last_bbox is None or worker.last_center is None:
                    continue
                w_box = worker.last_bbox
                w_center = worker.last_center

                dist_b = self.box_distance(c_box, w_box)
                dist_c = self.euclidean_dist(c_center, w_center)
                effective_dist = min(dist_b, dist_c)

                if effective_dist <= max_handling_dist:
                    if effective_dist < min_worker_dist:
                        min_worker_dist = effective_dist
                        interacting_worker = worker

            if interacting_worker is None:
                # Carton is resting untouched; no manual handling
                continue

            # 3. Temporal persistence + actual handling motion: a resting carton near a worker
            # must not trigger solely because it is close by. Require at least a meaningful
            # displacement of the carton or worker over the same handling window.
            consecutive_handling = 0
            handling_motion_px = 0.0
            motion_frames = 0
            w_traj = interacting_worker.trajectory
            w_frame_map = {tp.frame_idx: tp for tp in w_traj}

            previous_c_tp = None
            previous_w_tp = None
            for c_tp in reversed(c_traj):
                w_tp = w_frame_map.get(c_tp.frame_idx)
                if w_tp is None:
                    break

                dist_b = self.box_distance(c_tp.bbox, w_tp.bbox)
                dist_c = self.euclidean_dist(c_tp.center, w_tp.center)
                if min(dist_b, dist_c) <= max_handling_dist:
                    consecutive_handling += 1
                    if previous_c_tp is not None and previous_w_tp is not None:
                        carton_delta = self.euclidean_dist(c_tp.center, previous_c_tp.center)
                        worker_delta = self.euclidean_dist(w_tp.center, previous_w_tp.center)
                        motion_step = max(carton_delta, worker_delta)
                        if motion_step > 1.5:
                            handling_motion_px += motion_step
                            motion_frames += 1
                    previous_c_tp = c_tp
                    previous_w_tp = w_tp
                else:
                    break

            if consecutive_handling < min_duration:
                continue

            if motion_frames < max(2, min_duration // 4) or handling_motion_px < 12.0:
                continue

            confidence = min(0.95, max(0.70, 0.78 + (consecutive_handling / 40.0) * 0.16))
            req_names = ", ".join(req_equipment)

            event = BehaviourEvent(
                event_id=f"EVT-NOEQUIP-{carton.track_id}-{interacting_worker.track_id}-F{current_frame_idx}",
                rule_name=self.name,
                track_id=carton.track_id,
                display_label=carton.display_label,
                secondary_track_id=interacting_worker.track_id,
                secondary_label=interacting_worker.display_label,
                class_name=carton.class_name,
                frame_idx=current_frame_idx,
                timestamp_seconds=timestamp,
                confidence=round(confidence, 3),
                bbox=c_box,
                metrics={
                    "consecutive_handling_frames": consecutive_handling,
                    "handling_motion_px": round(handling_motion_px, 1),
                    "worker_distance_px": round(min_worker_dist, 1),
                    "required_equipment": req_equipment,
                    "equipment_present": False,
                    "explanation": (
                        f"Product being handled manually by {interacting_worker.display_label} for "
                        f"{consecutive_handling} consecutive frames without required equipment ({req_names})."
                    ),
                },
            )
            events.append(event)

        return events
