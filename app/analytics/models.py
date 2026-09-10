"""
Models and dataclasses for Warehouse Risk & Behaviour Analytics.
Converts detected video incidents into warehouse-level operational insights,
including behaviour frequencies, risk tier distributions, repeat entity tracking,
spatial zone analysis, temporal clustering, and targeted prevention priorities.
Strictly adheres to Godrej Responsible AI damage assessment principles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Union

from app.risk.models import DamageAssessmentStatus, IncidentRecord, RiskLevel


# Canonical warehouse handling behaviours predefined in the Godrej AI pipeline
SUPPORTED_BEHAVIOURS: List[str] = [
    "product_dropped",
    "product_thrown",
    "product_dragged",
    "product_pushed",
    "product_rolled",
    "unstable_stacking",
    "pallet_overhang",
    "product_outside_designated_area",
    "product_handled_without_required_equipment",
    "unsafe_loading_unloading_sequence",
]


@dataclass
class BehaviourFrequencyItem:
    """Frequency and distribution metric for a specific warehouse behaviour."""
    rule_name: str
    display_name: str
    count: int
    percentage: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "display_name": self.display_name,
            "count": self.count,
            "percentage": self.percentage,
        }


@dataclass
class RepeatTrackItem:
    """Tracks physical entities that accumulated multiple handling infractions."""
    track_id: int
    display_label: str
    incident_count: int
    behaviours: List[str]
    incident_ids: List[str]
    max_risk_level: str
    average_risk_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "display_label": self.display_label,
            "incident_count": self.incident_count,
            "behaviours": self.behaviours,
            "incident_ids": self.incident_ids,
            "max_risk_level": self.max_risk_level,
            "average_risk_score": self.average_risk_score,
        }


@dataclass
class LocationAnalyticsItem:
    """Incident density and risk profile for an identified warehouse location or zone."""
    location: str
    incident_count: int
    percentage: float
    risk_distribution: Dict[str, int]
    average_risk_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "incident_count": self.incident_count,
            "percentage": self.percentage,
            "risk_distribution": self.risk_distribution,
            "average_risk_score": self.average_risk_score,
        }


@dataclass
class TemporalCluster:
    """A high-density temporal window where multiple incidents occurred in close succession."""
    cluster_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    incident_count: int
    incident_ids: List[str]
    dominant_behaviour: Optional[str] = None
    dominant_risk_level: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "incident_count": self.incident_count,
            "incident_ids": self.incident_ids,
            "dominant_behaviour": self.dominant_behaviour,
            "dominant_risk_level": self.dominant_risk_level,
        }


@dataclass
class PreventionInsight:
    """Deterministic, actionable operational recommendation derived from incident patterns."""
    category: str
    title: str
    observation: str
    recommendation: str
    severity: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "title": self.title,
            "observation": self.observation,
            "recommendation": self.recommendation,
            "severity": self.severity,
        }


@dataclass
class WarehouseAnalytics:
    """
    Standardized, warehouse-level operational analytics result.
    Synthesizes behaviour distributions, risk concentration, repeated entities,
    zone data, temporal activity, and prevention priorities.
    """
    total_incidents: int
    behaviour_counts: Dict[str, int]
    behaviour_percentages: Dict[str, float]
    most_frequent_behaviours: List[str]
    least_frequent_behaviours: List[str]
    risk_counts: Dict[str, int]
    risk_percentages: Dict[str, float]
    average_risk_score: float
    max_risk_score: float
    highest_risk_incident: Optional[Dict[str, Any]]
    repeat_tracks: List[RepeatTrackItem]
    total_repeat_incidents: int
    highest_repeat_track: Optional[RepeatTrackItem]
    location_counts: Union[Dict[str, int], str]
    highest_risk_location: str
    location_analysis: Union[Dict[str, LocationAnalyticsItem], str]
    temporal_analysis: Dict[str, Any]
    prevention_insights: List[PreventionInsight]
    damage_assessment_distinction: str = (
        "Observed Behaviour -> Potential Damage Risk -> Confirmed Damage Only When Physical Evidence Exists"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes analytics result into a clean, frontend-friendly dictionary structure."""
        loc_counts = (
            self.location_counts
            if isinstance(self.location_counts, (dict, str))
            else str(self.location_counts)
        )
        loc_analysis = (
            {k: v.to_dict() for k, v in self.location_analysis.items()}
            if isinstance(self.location_analysis, dict)
            else self.location_analysis
        )

        return {
            "summary": {
                "total_incidents": self.total_incidents,
                "average_risk_score": self.average_risk_score,
                "highest_risk_score": self.max_risk_score,
                "total_repeat_incidents": self.total_repeat_incidents,
            },
            "total_incidents": self.total_incidents,
            "average_risk_score": self.average_risk_score,
            "max_risk_score": self.max_risk_score,
            "highest_risk_incident": self.highest_risk_incident,
            "behaviour_frequency": self.behaviour_counts,
            "behaviour_counts": self.behaviour_counts,
            "behaviour_percentages": self.behaviour_percentages,
            "most_frequent_behaviours": self.most_frequent_behaviours,
            "least_frequent_behaviours": self.least_frequent_behaviours,
            "risk_distribution": self.risk_counts,
            "risk_counts": self.risk_counts,
            "risk_percentages": self.risk_percentages,
            "repeat_violations": [t.to_dict() for t in self.repeat_tracks],
            "repeat_tracks": [t.to_dict() for t in self.repeat_tracks],
            "total_repeat_incidents": self.total_repeat_incidents,
            "highest_repeat_track": self.highest_repeat_track.to_dict() if self.highest_repeat_track else None,
            "location_analysis": loc_analysis,
            "location_counts": loc_counts,
            "highest_risk_location": self.highest_risk_location,
            "temporal_analysis": self.temporal_analysis,
            "prevention_insights": [i.to_dict() for i in self.prevention_insights],
            "damage_assessment_distinction": self.damage_assessment_distinction,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes analytics result to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
