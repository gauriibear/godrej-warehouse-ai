"""
Data models and configuration for Godrej Warehouse AI Risk Engine & Incident Evidence.
Stage 5: Risk classification, incident record definitions, and damage assessment distinctions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Tuple


class RiskLevel(str, Enum):
    """Four-tier operational risk classification levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def color_hex(self) -> str:
        """Hex color code for dashboard and UI rendering."""
        colors = {
            "LOW": "#2563EB",       # Blue
            "MEDIUM": "#D97706",    # Amber
            "HIGH": "#DC2626",      # Red
            "CRITICAL": "#991B1B",  # Dark Red
        }
        return colors.get(self.value, "#6B7280")


class DamageAssessmentStatus(str, Enum):
    """
    Explicit separation of observed kinematics, potential risk, and confirmed physical damage.
    Godrej Responsible AI policy: Never claim damage occurred without physical proof.
    """
    OBSERVED_BEHAVIOUR = "OBSERVED_BEHAVIOUR"
    POTENTIAL_DAMAGE_RISK = "POTENTIAL_DAMAGE_RISK"
    CONFIRMED_DAMAGE = "CONFIRMED_DAMAGE"

    @property
    def human_readable(self) -> str:
        names = {
            "OBSERVED_BEHAVIOUR": "Observed Handling Behaviour",
            "POTENTIAL_DAMAGE_RISK": "Potential Damage Risk (Unconfirmed)",
            "CONFIRMED_DAMAGE": "Confirmed Physical Damage",
        }
        return names.get(self.value, self.value)


@dataclass
class RiskConfig:
    """
    Configurable parameters and baseline severities for rule-based risk evaluation.
    All kinematic values represent pixel/video coordinate heuristics.
    """
    # Score classification boundaries [0, 100]
    low_max_score: float = 40.0
    medium_max_score: float = 70.0
    high_max_score: float = 85.0

    # Base severity scores per behaviour rule
    base_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "product_dragged": 45.0,
            "product_rolled": 50.0,
            "pallet_overhang": 55.0,
            "product_pushed": 60.0,
            "unstable_stacking": 72.0,
            "product_dropped": 75.0,
            "product_thrown": 80.0,
        }
    )

    # Metric escalation increments
    drop_high_velocity_px_s: float = 500.0
    drop_high_velocity_penalty: float = 10.0
    drop_high_height_px: float = 120.0
    drop_high_height_penalty: float = 5.0

    throw_high_velocity_px_s: float = 450.0
    throw_high_velocity_penalty: float = 10.0
    throw_high_airborne_frames: int = 8
    throw_high_airborne_penalty: float = 5.0

    push_high_impulse_px_s: float = 350.0
    push_high_impulse_penalty: float = 15.0

    drag_prolonged_frames: int = 25
    drag_prolonged_penalty: float = 10.0

    stacking_critical_offset_ratio: float = 0.50
    stacking_critical_offset_penalty: float = 14.0

    overhang_critical_ratio: float = 0.25
    overhang_critical_penalty: float = 15.0

    roll_high_flips: int = 4
    roll_high_flips_penalty: float = 8.0

    # Repeat infraction penalties on same entity
    repeat_violation_penalty_2nd: float = 10.0
    repeat_violation_penalty_3rd_plus: float = 20.0

    # Actionable damage prevention and process recommendations
    recommendations: Dict[str, str] = field(
        default_factory=lambda: {
            "product_dropped": (
                "Inspect carton integrity and contents for internal shock or rupture before dispatch; "
                "reinforce two-handed lifting protocol and verify adequate grip gear."
            ),
            "product_thrown": (
                "Quarantine affected merchandise for mandatory packaging and contents audit; "
                "review transfer zone layout to eliminate long unassisted manual passing."
            ),
            "product_dragged": (
                "Inspect bottom flaps and tape seams for friction tear or floor contamination; "
                "utilize hand trolleys or roller conveyors for floor transitions."
            ),
            "product_pushed": (
                "Verify packaging corners and lower structural integrity; "
                "ensure designated walkways remain clear and use push-carts instead of manual kicks."
            ),
            "product_rolled": (
                "Check interior product orientation and seals for crushing; "
                "handle cylindrical or bulky items with pallet jacks or cradle dollies."
            ),
            "unstable_stacking": (
                "Restack column immediately before transit; align carton center-of-mass within 15% base margin "
                "and apply stretch wrap reinforcement."
            ),
            "pallet_overhang": (
                "Reposition carton flush within pallet perimeter edges; eliminate overhang exceeding 5% "
                "to prevent forklift shearing and rack snagging."
            ),
        }
    )


@dataclass
class IncidentRecord:
    """
    Standardized record capturing a verified material handling incident,
    including risk classification, explainability, evidence paths, and mitigation recommendations.
    """
    incident_id: str
    event_id: str
    rule_name: str
    human_readable_name: str
    risk_level: RiskLevel
    risk_score: float
    timestamp_seconds: float
    frame_idx: int
    track_id: int
    display_label: str
    secondary_track_id: Optional[int] = None
    secondary_label: Optional[str] = None
    class_name: str = "carton"
    confidence: float = 0.85
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    metrics: Dict[str, Any] = field(default_factory=dict)
    damage_assessment: DamageAssessmentStatus = DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK
    explanation: str = ""
    recommendation: str = ""
    evidence_image_path: Optional[str] = None
    video_clip_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts incident record to JSON-serializable dictionary."""
        return {
            "incident_id": self.incident_id,
            "event_id": self.event_id,
            "rule_name": self.rule_name,
            "human_readable_name": self.human_readable_name,
            "risk_level": self.risk_level.value if isinstance(self.risk_level, RiskLevel) else str(self.risk_level),
            "risk_score": round(self.risk_score, 1),
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "frame_idx": self.frame_idx,
            "track_id": self.track_id,
            "display_label": self.display_label,
            "secondary_track_id": self.secondary_track_id,
            "secondary_label": self.secondary_label,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
            "metrics": self.metrics,
            "damage_assessment": (
                self.damage_assessment.value
                if isinstance(self.damage_assessment, DamageAssessmentStatus)
                else str(self.damage_assessment)
            ),
            "damage_assessment_label": (
                self.damage_assessment.human_readable
                if isinstance(self.damage_assessment, DamageAssessmentStatus)
                else str(self.damage_assessment)
            ),
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "evidence_image_path": self.evidence_image_path,
            "video_clip_path": self.video_clip_path,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes incident record to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
