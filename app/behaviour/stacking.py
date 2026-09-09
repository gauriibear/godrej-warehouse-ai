"""
Spatial and temporal rule detector for improper stacking, unstable stacking, and pallet overhang.
Detects center-of-gravity horizontal offsets between stacked cartons and cartons exceeding pallet boundaries.
"""

from __future__ import annotations

from typing import Dict, List

from app.behaviour.engine import BaseBehaviourRule, BehaviourConfig, BehaviourEvent
from app.tracking.tracker import TrackedObject


class StackingRule(BaseBehaviourRule):
    """
    Detects stacking violations:
    1. unstable_stacking: Upper carton center-of-mass deviates horizontally from lower support box.
    2. pallet_overhang: Carton horizontal boundary protrudes beyond pallet edges.
    """

    name: str = "stacking_rules"

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

        # Partition active objects into cartons and pallets
        active_cartons: List[TrackedObject] = []
        active_pallets: List[TrackedObject] = []

        for tid in active_track_ids:
            obj = tracked_objects.get(tid)
            if obj is None or obj.last_bbox is None:
                continue
            if self.is_package(obj.class_name):
                active_cartons.append(obj)
            elif self.is_pallet(obj.class_name):
                active_pallets.append(obj)

        # ── 1. Unstable Stacking Check (Carton on Carton) ───────────────────
        for upper in active_cartons:
            u_box = upper.last_bbox
            u_center = upper.last_center
            if u_box is None or u_center is None:
                continue

            u_w = max(1, u_box[2] - u_box[0])
            u_y2 = u_box[3]

            for lower in active_cartons:
                if upper.track_id == lower.track_id:
                    continue

                l_box = lower.last_bbox
                l_center = lower.last_center
                if l_box is None or l_center is None:
                    continue

                l_w = max(1, l_box[2] - l_box[0])
                l_y1 = l_box[1]

                # Vertical contact check: bottom of upper box is close to top of lower box
                if abs(u_y2 - l_y1) > 35.0:
                    continue

                # Horizontal interval overlap check
                overlap_x1 = max(u_box[0], l_box[0])
                overlap_x2 = min(u_box[2], l_box[2])
                if overlap_x2 <= overlap_x1:
                    continue

                # Center of gravity horizontal offset ratio
                offset_px = abs(u_center[0] - l_center[0])
                offset_ratio = offset_px / l_w

                if offset_ratio >= config.stacking_max_offset_ratio:
                    # Verify persistence (minimum 3 frames to avoid transient placement noise)
                    min_frames = min(upper.frame_count, lower.frame_count)
                    if min_frames >= 3:
                        event = BehaviourEvent(
                            event_id=f"EVT-STACK-{upper.track_id}-{lower.track_id}-F{current_frame_idx}",
                            rule_name="unstable_stacking",
                            track_id=upper.track_id,
                            display_label=upper.display_label,
                            secondary_track_id=lower.track_id,
                            secondary_label=lower.display_label,
                            class_name=upper.class_name,
                            frame_idx=current_frame_idx,
                            timestamp_seconds=timestamp,
                            confidence=min(0.95, offset_ratio / (config.stacking_max_offset_ratio * 1.5)),
                            bbox=u_box,
                            metrics={
                                "offset_ratio": round(offset_ratio, 3),
                                "offset_px": round(offset_px, 1),
                                "base_width_px": l_w,
                            },
                        )
                        events.append(event)

        # ── 2. Pallet Overhang Check (Carton on Pallet) ─────────────────────
        for carton in active_cartons:
            c_box = carton.last_bbox
            if c_box is None:
                continue

            c_w = max(1, c_box[2] - c_box[0])
            c_y2 = c_box[3]

            for pallet in active_pallets:
                p_box = pallet.last_bbox
                if p_box is None:
                    continue

                p_y1 = p_box[1]

                # Vertical contact check: carton bottom is near pallet top
                if abs(c_y2 - p_y1) > 40.0 and not (p_box[1] <= c_y2 <= p_box[3]):
                    continue

                # Horizontal overhang check
                overhang_left = max(0, p_box[0] - c_box[0])
                overhang_right = max(0, c_box[2] - p_box[2])
                max_overhang = max(overhang_left, overhang_right)
                overhang_ratio = max_overhang / c_w

                if overhang_ratio >= config.pallet_overhang_max_ratio:
                    event = BehaviourEvent(
                        event_id=f"EVT-OVERHANG-{carton.track_id}-{pallet.track_id}-F{current_frame_idx}",
                        rule_name="pallet_overhang",
                        track_id=carton.track_id,
                        display_label=carton.display_label,
                        secondary_track_id=pallet.track_id,
                        secondary_label=pallet.display_label,
                        class_name=carton.class_name,
                        frame_idx=current_frame_idx,
                        timestamp_seconds=timestamp,
                        confidence=min(0.95, overhang_ratio / (config.pallet_overhang_max_ratio * 1.5)),
                        bbox=c_box,
                        metrics={
                            "overhang_ratio": round(overhang_ratio, 3),
                            "overhang_px": round(max_overhang, 1),
                            "carton_width_px": c_w,
                        },
                    )
                    events.append(event)

        return events
