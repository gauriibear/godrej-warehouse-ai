"""
Risk Engine for Godrej Warehouse AI.
Stage 5: Rule-based risk scoring, metric adjustments, repeat infraction escalation,
explainable hazard reasoning, and process recommendations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.behaviour.engine import BehaviourEvent
from app.risk.models import (
    DamageAssessmentStatus,
    IncidentRecord,
    RiskConfig,
    RiskLevel,
)


class RiskEngine:
    """
    Transparent, rule-based operational risk engine that evaluates detected
    material handling behaviour events and produces explainable incident records.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.incidents: List[IncidentRecord] = []
        self.track_incident_counts: Dict[int, int] = {}
        self.track_prior_rules: Dict[int, List[str]] = {}

    def reset(self) -> None:
        """Resets engine state between video processing sessions."""
        self.incidents.clear()
        self.track_incident_counts.clear()
        self.track_prior_rules.clear()

    @property
    def all_incidents(self) -> List[IncidentRecord]:
        """Returns all recorded incidents in the current session."""
        return list(self.incidents)

    def evaluate_event(
        self,
        event: BehaviourEvent,
        session_history: Optional[List[BehaviourEvent]] = None,
    ) -> IncidentRecord:
        """
        Evaluates a confirmed BehaviourEvent and produces an explainable IncidentRecord.
        Computes calibrated risk score [0, 100], determines RiskLevel,
        and generates context-aware narrative explanations and recommendations.
        """
        # 1. Base score lookup
        base_score = self.config.base_scores.get(event.rule_name, 50.0)
        score = base_score
        reason_components: List[str] = [
            f"Base severity for {event.human_readable_name} ({base_score:.0f} pts)."
        ]

        metrics = event.metrics or {}

        # 2. Metric adjustments based on kinematic heuristics
        if event.rule_name == "product_dropped":
            vy = metrics.get("vertical_velocity_px_s", 0.0)
            drop_h = metrics.get("drop_height_px", 0.0)
            if vy >= self.config.drop_high_velocity_px_s:
                score += self.config.drop_high_velocity_penalty
                reason_components.append(
                    f"High vertical descent velocity ({vy:.0f} px/s >= {self.config.drop_high_velocity_px_s:.0f} px/s: "
                    f"+{self.config.drop_high_velocity_penalty:.0f} pts)."
                )
            if drop_h >= self.config.drop_high_height_px:
                score += self.config.drop_high_height_penalty
                reason_components.append(
                    f"Substantial vertical fall distance ({drop_h:.0f} px >= {self.config.drop_high_height_px:.0f} px: "
                    f"+{self.config.drop_high_height_penalty:.0f} pts)."
                )

        elif event.rule_name == "product_thrown":
            vx = metrics.get("horizontal_velocity_px_s", 0.0)
            air_f = metrics.get("airborne_frames", 0)
            if vx >= self.config.throw_high_velocity_px_s:
                score += self.config.throw_high_velocity_penalty
                reason_components.append(
                    f"High lateral ballistic speed ({vx:.0f} px/s >= {self.config.throw_high_velocity_px_s:.0f} px/s: "
                    f"+{self.config.throw_high_velocity_penalty:.0f} pts)."
                )
            if air_f >= self.config.throw_high_airborne_frames:
                score += self.config.throw_high_airborne_penalty
                reason_components.append(
                    f"Prolonged airborne detachment ({air_f} frames: +{self.config.throw_high_airborne_penalty:.0f} pts)."
                )

        elif event.rule_name == "product_pushed":
            spike = metrics.get("impulse_spike_px_s", 0.0)
            if spike >= self.config.push_high_impulse_px_s:
                score += self.config.push_high_impulse_penalty
                reason_components.append(
                    f"High-velocity impulse spike ({spike:.0f} px/s >= {self.config.push_high_impulse_px_s:.0f} px/s: "
                    f"+{self.config.push_high_impulse_penalty:.0f} pts, escalating hazard)."
                )

        elif event.rule_name == "product_dragged":
            drag_f = metrics.get("drag_duration_frames", 0)
            if drag_f >= self.config.drag_prolonged_frames:
                score += self.config.drag_prolonged_penalty
                reason_components.append(
                    f"Prolonged floor contact sliding ({drag_f} frames >= {self.config.drag_prolonged_frames}: "
                    f"+{self.config.drag_prolonged_penalty:.0f} pts)."
                )

        elif event.rule_name == "unstable_stacking":
            offset_r = metrics.get("offset_ratio", 0.0)
            if offset_r >= self.config.stacking_critical_offset_ratio:
                score += self.config.stacking_critical_offset_penalty
                reason_components.append(
                    f"Severe center-of-mass offset ({offset_r * 100:.0f}% base width >= 50%: "
                    f"+{self.config.stacking_critical_offset_penalty:.0f} pts, structural tipping risk)."
                )

        elif event.rule_name == "pallet_overhang":
            oh_r = metrics.get("overhang_ratio", 0.0)
            if oh_r >= self.config.overhang_critical_ratio:
                score += self.config.overhang_critical_penalty
                reason_components.append(
                    f"Excessive pallet perimeter protrusion ({oh_r * 100:.0f}% width >= 25%: "
                    f"+{self.config.overhang_critical_penalty:.0f} pts, snagging/shear hazard)."
                )

        elif event.rule_name == "product_rolled":
            flips = metrics.get("aspect_ratio_flips", 0)
            if flips >= self.config.roll_high_flips:
                score += self.config.roll_high_flips_penalty
                reason_components.append(
                    f"Repeated multi-revolution tumble ({flips} flips >= 4: "
                    f"+{self.config.roll_high_flips_penalty:.0f} pts)."
                )

        elif event.rule_name == "product_outside_designated_area":
            dur = metrics.get("consecutive_outside_frames", 0)
            dist = metrics.get("distance_outside_px", 0.0)
            if dur >= self.config.outside_area_prolonged_frames:
                score += self.config.outside_area_prolonged_penalty
                reason_components.append(
                    f"Prolonged positioning outside designated zone ({dur} frames >= {self.config.outside_area_prolonged_frames}: "
                    f"+{self.config.outside_area_prolonged_penalty:.0f} pts)."
                )
            if dist >= self.config.outside_area_high_distance_px:
                score += self.config.outside_area_high_distance_penalty
                reason_components.append(
                    f"Substantial displacement outside authorized perimeter ({dist:.0f} px >= {self.config.outside_area_high_distance_px:.0f} px: "
                    f"+{self.config.outside_area_high_distance_penalty:.0f} pts, transit obstruction risk)."
                )

        elif event.rule_name == "product_handled_without_required_equipment":
            dur = metrics.get("consecutive_handling_frames", 0)
            if dur >= self.config.equipment_prolonged_frames:
                score += self.config.equipment_prolonged_penalty
                reason_components.append(
                    f"Prolonged manual handling without required equipment ({dur} frames >= {self.config.equipment_prolonged_frames}: "
                    f"+{self.config.equipment_prolonged_penalty:.0f} pts, increased risk of drops/sprains)."
                )

        elif event.rule_name == "unsafe_loading_unloading_sequence":
            seq_dur = metrics.get("sequence_duration_frames", 0)
            act_spd = metrics.get("action_speed_px_s", 0.0)
            if seq_dur <= self.config.sequence_rapid_frames:
                score += self.config.sequence_rapid_penalty
                reason_components.append(
                    f"Rapid out-of-order execution ({seq_dur} frames <= {self.config.sequence_rapid_frames}: "
                    f"+{self.config.sequence_rapid_penalty:.0f} pts, severe procedural violation)."
                )
            if act_spd >= self.config.sequence_high_action_speed_px_s:
                score += self.config.sequence_high_action_speed_penalty
                reason_components.append(
                    f"High-velocity abrupt action during sequence ({act_spd:.0f} px/s >= {self.config.sequence_high_action_speed_px_s:.0f} px/s: "
                    f"+{self.config.sequence_high_action_speed_penalty:.0f} pts, impact hazard)."
                )

        # 3. Repeated violation penalty on the same entity
        prior_count = self.track_incident_counts.get(event.track_id, 0)
        if prior_count == 1:
            score += self.config.repeat_violation_penalty_2nd
            reason_components.append(
                f"Repeat infraction (2nd incident involving {event.display_label}: "
                f"+{self.config.repeat_violation_penalty_2nd:.0f} pts)."
            )
        elif prior_count >= 2:
            score += self.config.repeat_violation_penalty_3rd_plus
            prior_rules = ", ".join(self.track_prior_rules.get(event.track_id, []))
            reason_components.append(
                f"Chronic repeat violations ({prior_count + 1}th incident involving {event.display_label}, "
                f"prior: [{prior_rules}]: +{self.config.repeat_violation_penalty_3rd_plus:.0f} pts, escalated)."
            )

        # 4. Confidence adjustments
        if event.confidence >= 0.85:
            score += 3.0
        elif event.confidence < 0.40:
            score -= 8.0
            reason_components.append(
                f"Lower detection confidence ({event.confidence * 100:.0f}%): -8 pts damping."
            )

        # 5. Bound score strictly in [0, 100]
        score = max(0.0, min(100.0, score))

        # 6. Map to 4-tier Risk Level
        if score < self.config.low_max_score:
            risk_level = RiskLevel.LOW
        elif score < self.config.medium_max_score:
            risk_level = RiskLevel.MEDIUM
        elif score < self.config.high_max_score:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        # 7. Generate explanation narrative
        explanation = (
            f"Classified as {risk_level.value} Risk (Score: {score:.1f}/100). "
            + " ".join(reason_components)
        )

        # 8. Generate constructive mitigation recommendation
        base_recommendation = self.config.recommendations.get(
            event.rule_name,
            "Review handling procedures and verify container structural integrity."
        )
        if prior_count >= 1:
            base_recommendation += (
                f" Note: {event.display_label} has multiple recorded violations; "
                "prioritize for physical pre-dispatch QA inspection."
            )

        # 9. Format structured Incident ID
        incident_id = f"INC-{event.event_id.replace('EVT-', '')}"

        # 10. Construct IncidentRecord
        incident = IncidentRecord(
            incident_id=incident_id,
            event_id=event.event_id,
            rule_name=event.rule_name,
            human_readable_name=event.human_readable_name,
            risk_level=risk_level,
            risk_score=score,
            timestamp_seconds=event.timestamp_seconds,
            frame_idx=event.frame_idx,
            track_id=event.track_id,
            display_label=event.display_label,
            secondary_track_id=event.secondary_track_id,
            secondary_label=event.secondary_label,
            class_name=event.class_name,
            confidence=event.confidence,
            bbox=event.bbox,
            metrics=event.metrics,
            damage_assessment=DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK,
            explanation=explanation,
            recommendation=base_recommendation,
        )

        # Update history
        self.incidents.append(incident)
        self.track_incident_counts[event.track_id] = prior_count + 1
        if event.track_id not in self.track_prior_rules:
            self.track_prior_rules[event.track_id] = []
        self.track_prior_rules[event.track_id].append(event.human_readable_name)

        return incident
