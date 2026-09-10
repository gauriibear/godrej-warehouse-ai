"""
Spatial and temporal rule detector for products placed outside designated areas.
Detects when a carton/package is positioned outside the authorized storage/loading/pallet zone
for a sustained duration.
Godrej Scenario #8: Product Outside Designated Area.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class OutsideDesignatedAreaRule(BaseBehaviourRule):
    """
    Detects when a package/carton is placed or left outside a permitted designated zone.
    Applies spatial boundary tolerance margins and temporal persistence checks
    to prevent false alarms from transient crossings or boundary jitter.
    """

    name: str = "product_outside_designated_area"

    def __init__(
        self,
        designated_zones: Optional[List[Tuple[int, int, int, int]]] = None,
        margin_px: Optional[float] = None,
        min_duration_frames: Optional[int] = None,
    ):
        """
        Args:
            designated_zones: Optional list of (x1, y1, x2, y2) permitted rectangular zones.
                              If None, falls back to config.designated_zones.
            margin_px: Optional boundary tolerance in pixels. If None, falls back to config.
            min_duration_frames: Optional consecutive frames required outside before triggering.
        """
        self.custom_zones = designated_zones
        self.custom_margin = margin_px
        self.custom_min_duration = min_duration_frames

    @staticmethod
    def is_outside_zone(
        bbox: Tuple[int, int, int, int],
        center: Tuple[int, int],
        zone: Tuple[int, int, int, int],
        margin_px: float,
    ) -> Tuple[bool, float]:
        """
        Evaluates whether an object is outside a given rectangular zone taking margin into account.
        Returns (is_outside, distance_outside_px).
        """
        zx1, zy1, zx2, zy2 = zone
        cx = center[0] if center else (bbox[0] + bbox[2]) / 2.0
        cy = center[1] if center else (bbox[1] + bbox[3]) / 2.0

        # Effective zone expanded by tolerance margin
        eff_x1 = zx1 - margin_px
        eff_y1 = zy1 - margin_px
        eff_x2 = zx2 + margin_px
        eff_y2 = zy2 + margin_px

        # Distance outside the expanded boundary
        dx = max(0.0, eff_x1 - cx, cx - eff_x2)
        dy = max(0.0, eff_y1 - cy, cy - eff_y2)
        dist = math.hypot(dx, dy)

        is_outside = dist > 0.0
        return is_outside, dist

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

        # Resolve zones, margin, and duration threshold from custom parameters or config
        zones = self.custom_zones or getattr(config, "designated_zones", [(100, 150, 1180, 650)])
        if not zones:
            return events

        margin = (
            self.custom_margin
            if self.custom_margin is not None
            else getattr(config, "outside_area_margin_px", 20.0)
        )
        min_duration = (
            self.custom_min_duration
            if self.custom_min_duration is not None
            else getattr(config, "outside_area_min_duration_frames", 10)
        )

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or not self.is_package(obj.class_name):
                continue

            traj = obj.trajectory
            if len(traj) < min_duration:
                continue

            # Check consecutive observations outside permitted zones (walking backwards from tail)
            consecutive_outside = 0
            latest_min_dist = float("inf")
            closest_zone = zones[0]

            for tp in reversed(traj):
                # Object is permitted if it is inside ANY configured zone
                outside_all = True
                min_dist_for_tp = float("inf")
                closest_zone_for_tp = zones[0]

                for z in zones:
                    is_out, dist = self.is_outside_zone(tp.bbox, tp.center, z, margin)
                    if not is_out:
                        outside_all = False
                        break
                    if dist < min_dist_for_tp:
                        min_dist_for_tp = dist
                        closest_zone_for_tp = z

                if outside_all:
                    consecutive_outside += 1
                    if consecutive_outside == 1:
                        latest_min_dist = min_dist_for_tp
                        closest_zone = closest_zone_for_tp
                else:
                    # Trajectory point was inside or within tolerance; stop consecutive count
                    break

            if consecutive_outside >= min_duration:
                last_box = obj.last_bbox or (0, 0, 0, 0)
                last_center = obj.last_center or (0, 0)
                confidence = min(0.95, max(0.70, 0.75 + (latest_min_dist / 250.0) * 0.20))

                event = BehaviourEvent(
                    event_id=f"EVT-OUTSIDE-{obj.track_id}-F{current_frame_idx}",
                    rule_name=self.name,
                    track_id=obj.track_id,
                    display_label=obj.display_label,
                    secondary_track_id=None,
                    secondary_label=None,
                    class_name=obj.class_name,
                    frame_idx=current_frame_idx,
                    timestamp_seconds=timestamp,
                    confidence=round(confidence, 3),
                    bbox=last_box,
                    metrics={
                        "consecutive_outside_frames": consecutive_outside,
                        "distance_outside_px": round(latest_min_dist, 1),
                        "designated_zone": list(closest_zone),
                        "margin_px": margin,
                        "object_center": list(last_center),
                        "explanation": (
                            f"Product positioned {latest_min_dist:.0f}px outside designated area "
                            f"for {consecutive_outside} consecutive frames."
                        ),
                    },
                )
                events.append(event)

        return events
