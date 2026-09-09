"""
Unit and integration tests for Stage 5: Risk Engine & Incident Evidence.
Validates risk classification, metric escalation, repeat penalty, explainability,
damage distinction, evidence capture, and manifest generation.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from app.behaviour.engine import BehaviourEvent
from app.risk.models import (
    DamageAssessmentStatus,
    IncidentRecord,
    RiskConfig,
    RiskLevel,
)
from app.risk.engine import RiskEngine
from app.risk.evidence import EvidenceCollector


def make_event(
    rule_name: str = "product_dropped",
    track_id: int = 7,
    display_label: str = "Carton #7",
    metrics: dict = None,
    confidence: float = 0.85,
    frame_idx: int = 45,
    timestamp: float = 1.5,
    bbox: tuple = (100, 200, 250, 350),
) -> BehaviourEvent:
    """Helper to create deterministic BehaviourEvent instances."""
    return BehaviourEvent(
        event_id=f"EVT-{rule_name.upper()}-{track_id}-F{frame_idx}",
        rule_name=rule_name,
        track_id=track_id,
        display_label=display_label,
        class_name="carton",
        frame_idx=frame_idx,
        timestamp_seconds=timestamp,
        confidence=confidence,
        bbox=bbox,
        metrics=metrics or {},
    )


# ── 1. Classification Boundaries ──────────────────────────────────────────

def test_low_risk_classification():
    """Proves an event with dampened confidence / low score maps to LOW."""
    engine = RiskEngine()
    # Configure low base score for testing LOW boundary
    config = RiskConfig()
    config.base_scores["minor_wobble"] = 30.0
    engine.config = config

    event = make_event(rule_name="minor_wobble", confidence=0.35)
    incident = engine.evaluate_event(event)

    assert incident.risk_level == RiskLevel.LOW
    assert incident.risk_score < 40.0


def test_medium_risk_classification():
    """Proves product_dragged and product_rolled map to MEDIUM baseline."""
    engine = RiskEngine()

    drag_event = make_event(
        rule_name="product_dragged",
        metrics={"drag_duration_frames": 12, "horizontal_velocity_px_s": 85.0},
    )
    inc_drag = engine.evaluate_event(drag_event)
    assert inc_drag.risk_level == RiskLevel.MEDIUM
    assert 40.0 <= inc_drag.risk_score < 70.0

    roll_event = make_event(
        rule_name="product_rolled",
        track_id=2,
        metrics={"aspect_ratio_flips": 2, "speed_px_s": 90.0},
    )
    inc_roll = engine.evaluate_event(roll_event)
    assert inc_roll.risk_level == RiskLevel.MEDIUM


def test_high_risk_classification():
    """Proves product_dropped, product_thrown, and unstable_stacking map to HIGH baseline."""
    engine = RiskEngine()

    drop_event = make_event(
        rule_name="product_dropped",
        track_id=1,
        metrics={"vertical_velocity_px_s": 380.0, "drop_height_px": 80.0},
    )
    inc_drop = engine.evaluate_event(drop_event)
    assert inc_drop.risk_level == RiskLevel.HIGH
    assert 70.0 <= inc_drop.risk_score < 85.0

    throw_event = make_event(
        rule_name="product_thrown",
        track_id=2,
        metrics={"horizontal_velocity_px_s": 320.0, "airborne_frames": 6},
    )
    inc_throw = engine.evaluate_event(throw_event)
    assert inc_throw.risk_level == RiskLevel.HIGH


# ── 2. Metric Severity Escalation ──────────────────────────────────────────

def test_metric_severity_escalation():
    """Tests that high drop speed and high descent distance increase the score."""
    engine = RiskEngine()

    # Normal drop
    normal_drop = make_event(
        rule_name="product_dropped",
        metrics={"vertical_velocity_px_s": 380.0, "drop_height_px": 60.0},
        confidence=0.80,
    )
    inc_normal = engine.evaluate_event(normal_drop)

    # Extreme drop exceeding velocity (>= 500 px/s) and height (>= 120 px) thresholds
    extreme_drop = make_event(
        rule_name="product_dropped",
        track_id=8,
        metrics={"vertical_velocity_px_s": 560.0, "drop_height_px": 140.0},
        confidence=0.80,
    )
    inc_extreme = engine.evaluate_event(extreme_drop)

    assert inc_extreme.risk_score > inc_normal.risk_score
    # Base 75 + 10 (velocity) + 5 (height) = 90 -> CRITICAL
    assert inc_extreme.risk_level == RiskLevel.CRITICAL


def test_critical_risk_escalation_from_stacking():
    """Tests that a severe stacking offset ratio (>= 50%) escalates to CRITICAL."""
    engine = RiskEngine()

    event = make_event(
        rule_name="unstable_stacking",
        metrics={"offset_ratio": 0.55, "offset_px": 65.0, "stationary_frames": 15},
    )
    incident = engine.evaluate_event(event)

    # Base 72 + 14 (critical offset) + 3 (high conf) = 89 -> CRITICAL
    assert incident.risk_score >= 85.0
    assert incident.risk_level == RiskLevel.CRITICAL


def test_push_impulse_escalation_to_high():
    """Tests that a strong push/kick impulse (>= 350 px/s) escalates MEDIUM to HIGH."""
    engine = RiskEngine()

    event = make_event(
        rule_name="product_pushed",
        metrics={"impulse_spike_px_s": 380.0, "prior_rest_speed_px_s": 20.0},
    )
    incident = engine.evaluate_event(event)

    # Base 60 + 15 (impulse penalty) + 3 (high conf) = 78 -> HIGH
    assert incident.risk_level == RiskLevel.HIGH


# ── 3. Repeated Violation Escalation ───────────────────────────────────────

def test_repeated_violation_escalation():
    """Tests that consecutive infractions involving the same entity track ID escalate score."""
    engine = RiskEngine()

    # 1st violation for Track #7
    ev1 = make_event(rule_name="product_dragged", track_id=7)
    inc1 = engine.evaluate_event(ev1)
    score1 = inc1.risk_score

    # 2nd violation for Track #7 (+10 penalty)
    ev2 = make_event(rule_name="product_pushed", track_id=7, frame_idx=80)
    inc2 = engine.evaluate_event(ev2)
    assert inc2.risk_score >= score1

    # 3rd violation for Track #7 (+20 penalty, escalates to CRITICAL)
    ev3 = make_event(rule_name="product_dropped", track_id=7, frame_idx=120)
    inc3 = engine.evaluate_event(ev3)
    assert inc3.risk_level == RiskLevel.CRITICAL
    assert "Chronic repeat violations" in inc3.explanation


# ── 4. Explainability & Mitigation Recommendations ─────────────────────────

def test_explanation_and_recommendation_generation():
    """Tests that explainability narrative and actionable non-punitive recommendations are populated."""
    engine = RiskEngine()

    event = make_event(
        rule_name="product_dropped",
        metrics={"vertical_velocity_px_s": 520.0, "drop_height_px": 130.0},
    )
    incident = engine.evaluate_event(event)

    assert len(incident.explanation) > 20
    assert "CRITICAL" in incident.explanation
    assert "520 px/s" in incident.explanation

    assert len(incident.recommendation) > 20
    assert "Inspect carton integrity" in incident.recommendation


# ── 5. Damage Distinction (Godrej Policy) ──────────────────────────────────

def test_no_false_claim_of_confirmed_damage():
    """
    Verifies that the system strictly adheres to the Godrej distinction:
    Observed Behaviour -> Potential Damage Risk -> Confirmed Damage (only if verified).
    """
    engine = RiskEngine()
    event = make_event(rule_name="product_thrown")
    incident = engine.evaluate_event(event)

    # Must be POTENTIAL_DAMAGE_RISK, never CONFIRMED_DAMAGE
    assert incident.damage_assessment == DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK
    assert incident.damage_assessment != DamageAssessmentStatus.CONFIRMED_DAMAGE
    assert "Potential Damage Risk" in incident.to_dict()["damage_assessment_label"]


# ── 6. Serialization & Dataclass Methods ───────────────────────────────────

def test_incident_record_serialisation():
    """Verifies to_dict and to_json serialize correctly."""
    engine = RiskEngine()
    event = make_event()
    incident = engine.evaluate_event(event)

    d = incident.to_dict()
    assert d["incident_id"].startswith("INC-")
    assert d["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert isinstance(d["risk_score"], float)
    assert "damage_assessment" in d

    json_str = incident.to_json()
    parsed = json.loads(json_str)
    assert parsed["incident_id"] == incident.incident_id


# ── 7. Evidence Collector Keyframe, Video Clip & Manifest ──────────────────

def test_evidence_collector_keyframe_and_buffer(tmp_path):
    """Tests frame ring buffer, keyframe snapshot saving, and manifest generation."""
    collector = EvidenceCollector(output_dir=str(tmp_path), pre_event_frames=5, post_event_frames=5)

    # Feed 15 synthetic video frames into buffer
    dummy_frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    for f in range(15):
        collector.add_frame(f, dummy_frame)

    engine = RiskEngine()
    event = make_event(frame_idx=8, bbox=(50, 50, 150, 150))
    incident = engine.evaluate_event(event)

    # Save keyframe
    keyframe_path = collector.save_keyframe(incident, dummy_frame, annotate=True)
    assert Path(keyframe_path).exists()
    assert Path(keyframe_path).stat().st_size > 0

    # Extract video clip from buffer
    clip_path = collector.extract_clip(incident, fps=25.0)
    assert clip_path is not None
    assert Path(clip_path).exists()
    assert Path(clip_path).stat().st_size > 0

    # Save manifest
    manifest_path = collector.save_manifest([incident])
    assert Path(manifest_path).exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert len(manifest_data) == 1
    assert manifest_data[0]["incident_id"] == incident.incident_id


# ── 8. End-to-End Stage 4 to Stage 5 Pipeline Integration ──────────────────

def test_full_stage4_to_stage5_pipeline(tmp_path):
    """Verifies that an event emitted from Stage 4 smoothly ingests into Stage 5 RiskEngine and EvidenceCollector."""
    risk_engine = RiskEngine()
    evidence_collector = EvidenceCollector(output_dir=str(tmp_path))

    # Synthetic Stage 4 BehaviourEvent
    event = BehaviourEvent(
        event_id="EVT-OVERHANG-4-F90",
        rule_name="pallet_overhang",
        track_id=4,
        display_label="Carton #4",
        secondary_track_id=1,
        secondary_label="Pallet #1",
        class_name="carton",
        frame_idx=90,
        timestamp_seconds=3.0,
        confidence=0.91,
        bbox=(200, 300, 450, 500),
        metrics={"overhang_ratio": 0.28, "overhang_px": 55.0},
    )

    incident = risk_engine.evaluate_event(event)
    assert incident.risk_level == RiskLevel.HIGH
    assert incident.secondary_label == "Pallet #1"

    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    img_path = evidence_collector.save_keyframe(incident, frame)
    assert Path(img_path).exists()
