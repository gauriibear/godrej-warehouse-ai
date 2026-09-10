"""
Unit and integration tests for Warehouse Risk & Behaviour Analytics (app/analytics/).
Validates behaviour frequency, risk distributions, repeat entity tracking,
location/zone analysis, temporal clustering, prevention insights, serialization,
and AI Operations Assistant integration.
Strictly verifies:
1. Risk alone NEVER becomes confirmed damage.
2. Missing location data NEVER generates fake locations.
3. Frequency ties NEVER invent a single winner.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.analytics import (
    AnalyticsEngine,
    analyze_incidents,
    BEHAVIOUR_PREVENTION_MAP,
    SUPPORTED_BEHAVIOURS,
    WarehouseAnalytics,
)
from app.analytics.models import (
    BehaviourFrequencyItem,
    LocationAnalyticsItem,
    PreventionInsight,
    RepeatTrackItem,
    TemporalCluster,
)
from app.assistant.assistant import OperationsAssistant, QueryIntent
from app.risk.models import DamageAssessmentStatus, IncidentRecord, RiskConfig, RiskLevel
from scripts.generate_evidence import generate_synthetic_evidence


def _make_incident(
    incident_id: str = "INC-1",
    rule_name: str = "product_dragged",
    human_readable_name: str = "Product Dragged",
    risk_level: RiskLevel = RiskLevel.HIGH,
    risk_score: float = 75.0,
    track_id: int = 1,
    display_label: str = "Carton #1",
    timestamp_seconds: float = 2.0,
    frame_idx: int = 60,
    metrics: dict | None = None,
    damage_assessment: DamageAssessmentStatus = DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK,
) -> IncidentRecord:
    """Helper to construct a deterministic IncidentRecord for testing."""
    return IncidentRecord(
        incident_id=incident_id,
        event_id=f"EVT-{incident_id}",
        rule_name=rule_name,
        human_readable_name=human_readable_name,
        risk_level=risk_level,
        risk_score=risk_score,
        track_id=track_id,
        display_label=display_label,
        timestamp_seconds=timestamp_seconds,
        frame_idx=frame_idx,
        class_name="carton",
        confidence=0.90,
        bbox=(100, 100, 200, 200),
        metrics=metrics or {},
        damage_assessment=damage_assessment,
        explanation=f"Explanation for {incident_id}",
        recommendation=f"Recommendation for {incident_id}",
    )


# =============================================================================
# 1. Empty Dataset & Single Incident
# =============================================================================

def test_empty_incident_list():
    """Verifies analytics engine handles an empty incident list gracefully."""
    engine = AnalyticsEngine([])
    result = engine.analyze()

    assert result.total_incidents == 0
    assert result.behaviour_counts == {}
    assert result.behaviour_percentages == {}
    assert result.most_frequent_behaviours == []
    assert result.least_frequent_behaviours == []
    assert result.risk_counts == {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    assert result.average_risk_score == 0.0
    assert result.max_risk_score == 0.0
    assert result.highest_risk_incident is None
    assert result.repeat_tracks == []
    assert result.total_repeat_incidents == 0
    assert result.highest_repeat_track is None
    assert result.location_counts == "location_data_unavailable"
    assert result.highest_risk_location == "location_data_unavailable"
    assert result.location_analysis == "location_data_unavailable"
    assert len(result.prevention_insights) == 1
    assert result.prevention_insights[0].category == "GENERAL"

    # Serialization
    d = result.to_dict()
    assert d["summary"]["total_incidents"] == 0
    assert d["location_analysis"] == "location_data_unavailable"
    j = result.to_json()
    assert '"total_incidents": 0' in j


def test_single_incident():
    """Verifies analytics calculation for a single isolated incident."""
    inc = _make_incident(
        incident_id="INC-DROP-1",
        rule_name="product_dropped",
        human_readable_name="Product Dropped",
        risk_level=RiskLevel.HIGH,
        risk_score=75.0,
        track_id=1,
    )
    result = analyze_incidents([inc])

    assert result.total_incidents == 1
    assert result.behaviour_counts == {"product_dropped": 1}
    assert result.behaviour_percentages == {"product_dropped": 100.0}
    assert result.most_frequent_behaviours == ["product_dropped"]
    assert result.least_frequent_behaviours == ["product_dropped"]
    assert result.risk_counts["HIGH"] == 1
    assert result.average_risk_score == 75.0
    assert result.max_risk_score == 75.0
    assert result.highest_risk_incident is not None
    assert result.highest_risk_incident["incident_id"] == "INC-DROP-1"
    assert result.repeat_tracks == []


# =============================================================================
# 2. Behaviour Frequency & Deterministic Tie-Breaking
# =============================================================================

def test_multiple_behaviours_frequency_ranking():
    """Verifies counting, percentages, and frequency ranking across behaviours."""
    incidents = [
        _make_incident("INC-1", "product_dragged", "Product Dragged"),
        _make_incident("INC-2", "product_dragged", "Product Dragged"),
        _make_incident("INC-3", "product_dragged", "Product Dragged"),
        _make_incident("INC-4", "product_dropped", "Product Dropped"),
        _make_incident("INC-5", "pallet_overhang", "Pallet Overhang"),
    ]
    result = analyze_incidents(incidents)

    assert result.total_incidents == 5
    assert result.behaviour_counts["product_dragged"] == 3
    assert result.behaviour_counts["product_dropped"] == 1
    assert result.behaviour_counts["pallet_overhang"] == 1
    assert result.behaviour_percentages["product_dragged"] == 60.0
    assert result.behaviour_percentages["product_dropped"] == 20.0
    assert result.behaviour_percentages["pallet_overhang"] == 20.0
    assert result.most_frequent_behaviours == ["product_dragged"]
    # Least frequent tie: pallet_overhang < product_dropped alphabetically
    assert result.least_frequent_behaviours == ["pallet_overhang", "product_dropped"]


def test_frequency_ties_never_invent_winner():
    """
    STRICT REQUIREMENT: If two or more behaviours have the same highest frequency,
    return all tied behaviours. NEVER invent a single winner.
    """
    # 2-way tie
    incidents_2 = [
        _make_incident("INC-1", "product_dropped", "Product Dropped"),
        _make_incident("INC-2", "product_dragged", "Product Dragged"),
    ]
    res_2 = analyze_incidents(incidents_2)
    assert len(res_2.most_frequent_behaviours) == 2
    # Alphabetical order: product_dragged < product_dropped
    assert res_2.most_frequent_behaviours == ["product_dragged", "product_dropped"]

    # 3-way tie
    incidents_3 = [
        _make_incident("INC-1", "product_dropped", "Product Dropped"),
        _make_incident("INC-2", "product_dragged", "Product Dragged"),
        _make_incident("INC-3", "pallet_overhang", "Pallet Overhang"),
    ]
    res_3 = analyze_incidents(incidents_3)
    assert len(res_3.most_frequent_behaviours) == 3
    assert res_3.most_frequent_behaviours == [
        "pallet_overhang",
        "product_dragged",
        "product_dropped",
    ]


# =============================================================================
# 3. Risk Distribution
# =============================================================================

def test_risk_distribution_and_scoring():
    """Verifies counts per risk tier, percentages, average score, and highest risk event."""
    incidents = [
        _make_incident("INC-1", "product_dragged", risk_level=RiskLevel.LOW, risk_score=20.0),
        _make_incident("INC-2", "product_pushed", risk_level=RiskLevel.MEDIUM, risk_score=50.0),
        _make_incident("INC-3", "product_dropped", risk_level=RiskLevel.HIGH, risk_score=80.0),
        _make_incident("INC-4", "unstable_stacking", risk_level=RiskLevel.CRITICAL, risk_score=95.0),
    ]
    result = analyze_incidents(incidents)

    assert result.risk_counts["LOW"] == 1
    assert result.risk_counts["MEDIUM"] == 1
    assert result.risk_counts["HIGH"] == 1
    assert result.risk_counts["CRITICAL"] == 1
    assert result.risk_percentages["LOW"] == 25.0
    assert result.risk_percentages["MEDIUM"] == 25.0
    assert result.risk_percentages["HIGH"] == 25.0
    assert result.risk_percentages["CRITICAL"] == 25.0

    # Average: (20 + 50 + 80 + 95) / 4 = 245 / 4 = 61.25
    assert result.average_risk_score == 61.25
    assert result.max_risk_score == 95.0
    assert result.highest_risk_incident is not None
    assert result.highest_risk_incident["incident_id"] == "INC-4"


# =============================================================================
# 4. Repeat Violations (Without Assuming Human Identity)
# =============================================================================

def test_repeat_track_detection():
    """Verifies detection of multiple violations on the same tracked physical object."""
    incidents = [
        _make_incident("INC-1", "product_dragged", track_id=5, display_label="Carton #5", risk_score=60.0),
        _make_incident("INC-2", "product_dropped", track_id=5, display_label="Carton #5", risk_score=80.0),
        _make_incident("INC-3", "product_pushed", track_id=5, display_label="Carton #5", risk_score=70.0),
        _make_incident("INC-4", "pallet_overhang", track_id=8, display_label="Pallet #8", risk_score=50.0),
        _make_incident("INC-5", "pallet_overhang", track_id=8, display_label="Pallet #8", risk_score=55.0),
        _make_incident("INC-6", "product_thrown", track_id=12, display_label="Carton #12", risk_score=90.0),  # Single
    ]
    result = analyze_incidents(incidents)

    assert len(result.repeat_tracks) == 2
    assert result.total_repeat_incidents == 5

    # Track #5 has 3 infractions -> highest repeat track
    assert result.highest_repeat_track is not None
    assert result.highest_repeat_track.track_id == 5
    assert result.highest_repeat_track.incident_count == 3
    assert result.highest_repeat_track.display_label == "Carton #5"
    assert "product_dragged" in result.highest_repeat_track.behaviours
    assert "product_dropped" in result.highest_repeat_track.behaviours

    # Single-incident track #12 must NOT be included in repeat_tracks
    repeat_tids = [t.track_id for t in result.repeat_tracks]
    assert 12 not in repeat_tids


# =============================================================================
# 5. Location / Zone Analytics
# =============================================================================

def test_missing_location_never_generates_fake_location():
    """
    STRICT REQUIREMENT: If location data is not available, return
    'location_data_unavailable'. DO NOT fabricate locations.
    Bounding box coordinates must NOT be treated as zone names.
    """
    incidents = [
        _make_incident(
            "INC-1",
            "product_outside_designated_area",
            metrics={"designated_zone": [100, 150, 800, 650]},  # Coordinate box, not a named zone
        ),
        _make_incident("INC-2", "product_dragged", metrics={}),
    ]
    result = analyze_incidents(incidents)

    assert result.location_counts == "location_data_unavailable"
    assert result.highest_risk_location == "location_data_unavailable"
    assert result.location_analysis == "location_data_unavailable"


def test_available_location_data_aggregation():
    """Verifies location grouping, incident counts, risk distribution, and highest risk zone."""
    incidents = [
        _make_incident("INC-1", "product_dragged", risk_level=RiskLevel.MEDIUM, risk_score=50.0, metrics={"zone": "Bay 1"}),
        _make_incident("INC-2", "product_dropped", risk_level=RiskLevel.HIGH, risk_score=80.0, metrics={"zone": "Bay 1"}),
        _make_incident("INC-3", "unstable_stacking", risk_level=RiskLevel.CRITICAL, risk_score=95.0, metrics={"zone": "Bay 2"}),
    ]
    result = analyze_incidents(incidents)

    assert isinstance(result.location_counts, dict)
    assert result.location_counts["Bay 1"] == 2
    assert result.location_counts["Bay 2"] == 1
    assert isinstance(result.location_analysis, dict)
    assert "Bay 1" in result.location_analysis
    assert "Bay 2" in result.location_analysis

    # Bay 2 has higher average risk score (95.0 vs 65.0)
    assert result.highest_risk_location == "Bay 2"
    assert result.location_analysis["Bay 2"].average_risk_score == 95.0
    assert result.location_analysis["Bay 1"].average_risk_score == 65.0


# =============================================================================
# 6. Temporal Analytics & Clustering
# =============================================================================

def test_temporal_timeline_and_clustering():
    """Verifies chronological timeline and temporal cluster detection (5-second window)."""
    incidents = [
        _make_incident("INC-1", "product_dragged", timestamp_seconds=1.0, frame_idx=30),
        _make_incident("INC-2", "product_dropped", timestamp_seconds=3.0, frame_idx=90),   # within 5s of INC-1 -> cluster
        _make_incident("INC-3", "product_pushed", timestamp_seconds=20.0, frame_idx=600), # isolated
        _make_incident("INC-4", "product_pushed", timestamp_seconds=22.0, frame_idx=660), # within 5s of INC-3 -> cluster
    ]
    result = analyze_incidents(incidents)
    temp = result.temporal_analysis

    assert len(temp["timeline"]) == 4
    assert temp["duration_seconds"] == 21.0

    # Two clusters should be detected
    clusters = temp["clusters"]
    assert len(clusters) == 2
    assert clusters[0]["incident_count"] == 2
    assert "INC-1" in clusters[0]["incident_ids"]
    assert "INC-2" in clusters[0]["incident_ids"]

    assert clusters[1]["incident_count"] == 2
    assert "INC-3" in clusters[1]["incident_ids"]
    assert "INC-4" in clusters[1]["incident_ids"]


# =============================================================================
# 7. Prevention Insights & Godrej Damage Policy Distinction
# =============================================================================

def test_prevention_insights_generation():
    """Verifies that actionable, deterministic prevention recommendations are derived from patterns."""
    incidents = [
        _make_incident("INC-1", "product_dragged", risk_level=RiskLevel.HIGH, risk_score=80.0, track_id=7),
        _make_incident("INC-2", "product_dragged", risk_level=RiskLevel.HIGH, risk_score=85.0, track_id=7),
        _make_incident("INC-3", "product_dragged", risk_level=RiskLevel.CRITICAL, risk_score=95.0, track_id=7),
    ]
    result = analyze_incidents(incidents)
    insights = result.prevention_insights

    assert len(insights) >= 2
    categories = [ins.category for ins in insights]
    assert "BEHAVIOUR_TREND" in categories
    assert "REPEAT_VIOLATION" in categories

    # Verify drag prevention recommendation is populated
    drag_insight = next(ins for ins in insights if ins.category == "BEHAVIOUR_TREND")
    assert "Product Dragged" in drag_insight.title
    assert "training" in drag_insight.recommendation.lower() or "lifting" in drag_insight.recommendation.lower()


def test_risk_alone_never_claims_confirmed_damage():
    """
    STRICT REQUIREMENT: Risk alone NEVER becomes confirmed damage.
    Preserves: Observed Behaviour -> Potential Damage Risk -> Confirmed Damage only with physical QA.
    """
    incidents = [
        _make_incident(
            "INC-CRIT-99",
            "unstable_stacking",
            risk_level=RiskLevel.CRITICAL,
            risk_score=99.0,
            damage_assessment=DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK,
        )
    ]
    result = analyze_incidents(incidents)

    # Must state distinction
    assert "Observed Behaviour -> Potential Damage Risk" in result.damage_assessment_distinction

    # Must never claim confirmed damage occurred
    serialized = result.to_json()
    assert "CONFIRMED_PHYSICAL_DAMAGE" not in serialized
    for ins in result.prevention_insights:
        assert "confirmed physical damage" not in ins.recommendation.lower()
        assert "confirmed physical damage" not in ins.observation.lower()


# =============================================================================
# 8. Dict & JSON Serialization
# =============================================================================

def test_to_dict_and_to_json_serialization():
    """Verifies clean, frontend-friendly serialization matching requirements."""
    incidents = [
        _make_incident("INC-1", "product_dragged", risk_level=RiskLevel.HIGH, risk_score=75.0, track_id=3),
        _make_incident("INC-2", "product_dragged", risk_level=RiskLevel.HIGH, risk_score=85.0, track_id=3),
    ]
    analytics = analyze_incidents(incidents)

    d = analytics.to_dict()
    assert "summary" in d
    assert d["summary"]["total_incidents"] == 2
    assert "behaviour_frequency" in d
    assert "risk_distribution" in d
    assert "repeat_violations" in d
    assert "location_analysis" in d
    assert "prevention_insights" in d

    # JSON export and re-parse
    j_str = analytics.to_json()
    parsed = json.loads(j_str)
    assert parsed["summary"]["total_incidents"] == 2
    assert parsed["behaviour_frequency"]["product_dragged"] == 2


# =============================================================================
# 9. AI Operations Assistant Integration
# =============================================================================

@pytest.fixture
def sample_assistant(tmp_path):
    """Provides an assistant instance initialized with synthetic warehouse evidence."""
    incidents = generate_synthetic_evidence(
        output_dir=str(tmp_path),
        generate_clips=False,
        verbose=False,
    )
    return OperationsAssistant.from_incidents(incidents)


def test_assistant_analytics_location_query(sample_assistant):
    """Verifies assistant handles location analytics questions and reports missing telemetry gracefully."""
    queries = [
        "Which location has the most incidents?",
        "Which location has the highest risk?",
        "Show location analytics",
    ]
    for q in queries:
        res = sample_assistant.ask(q)
        assert res.intent == QueryIntent.ANALYTICS_LOCATION
        # Synthetic incidents do not have named zone metadata -> must report unavailable
        assert "Location/Zone Analytics Unavailable" in res.answer or "location_data_unavailable" in res.answer


def test_assistant_analytics_location_query_with_data():
    """Verifies assistant formats location analytics when spatial metadata is present."""
    incidents = [
        _make_incident("INC-1", "product_dragged", risk_level=RiskLevel.HIGH, risk_score=80.0, metrics={"zone": "Bay 3"}),
        _make_incident("INC-2", "product_dropped", risk_level=RiskLevel.CRITICAL, risk_score=95.0, metrics={"zone": "Bay 3"}),
    ]
    asst = OperationsAssistant.from_incidents(incidents)
    res = asst.ask("Which location has the most incidents?")

    assert res.intent == QueryIntent.ANALYTICS_LOCATION
    assert "Bay 3" in res.answer
    assert "Highest-risk location: **Bay 3**" in res.answer
    assert res.data["highest_risk_location"] == "Bay 3"


def test_assistant_analytics_prevention_priorities(sample_assistant):
    """Verifies assistant answers supervisor focus and prevention priority queries."""
    queries = [
        "What should the supervisor focus on?",
        "What are the main prevention priorities?",
        "What should warehouse managers prioritize to reduce damage?",
    ]
    for q in queries:
        res = sample_assistant.ask(q)
        assert res.intent == QueryIntent.PREVENTION_PRIORITIES, f"Failed for query: '{q}', got {res.intent}"
        assert "Warehouse Supervisor Focus & Prevention Priorities" in res.answer
        assert len(res.data["priorities"]) > 0
        assert "Potential Damage Risk" in res.answer


def test_assistant_analytics_risk_distribution(sample_assistant):
    """Verifies assistant answers risk distribution and tier count queries."""
    queries = [
        "Show me the risk distribution.",
        "How many critical incidents occurred?",
        "How many high risk incidents were there?",
    ]
    for q in queries:
        res = sample_assistant.ask(q)
        assert res.intent == QueryIntent.ANALYTICS_RISK, f"Failed for query: '{q}', got {res.intent}"
        assert "Warehouse Handling Risk Distribution" in res.answer
        assert "CRITICAL Risk" in res.answer
        assert "HIGH Risk" in res.answer
        assert "Average Risk Score" in res.answer


def test_assistant_repeat_violation_query(sample_assistant):
    """Verifies assistant answers general repeat violation query using analytics."""
    res = sample_assistant.ask("Which product or track had repeated violations?")
    assert res.intent == QueryIntent.REPEAT_HISTORY
    assert "Repeat Violation" in res.answer or "Track" in res.answer


# =============================================================================
# 10. Real Evidence Manifest Test
# =============================================================================

def test_analytics_on_real_manifest():
    """Runs AnalyticsEngine on the real warehouse incident evidence manifest."""
    manifest_path = Path("outputs/evidence/incidents.json")
    if not manifest_path.exists():
        pytest.skip("outputs/evidence/incidents.json does not exist")

    engine = AnalyticsEngine.from_manifest(manifest_path)
    result = engine.analyze()

    assert result.total_incidents > 0
    assert result.average_risk_score > 0.0
    assert result.max_risk_score > 0.0
    assert len(result.prevention_insights) > 0
    assert result.location_counts == "location_data_unavailable"
