"""
Unit and integration tests for Godrej Scenario #8: Product Outside Designated Area.
Validates spatial zone checks, boundary tolerance margins, temporal duration persistence,
independent object tracking, metric generation, RiskEngine integration, and Godrej damage policy compliance.
"""

from typing import List, Tuple
import pytest

from app.behaviour.engine import BehaviourConfig, BehaviourEngine, BehaviourEvent
from app.behaviour.outside_area import OutsideDesignatedAreaRule
from app.risk import DamageAssessmentStatus, RiskConfig, RiskEngine, RiskLevel
from app.tracking.tracker import TrackedObject, TrackPoint, ObjectTracker
from app.detection.detector import Detection, DetectionResult


def make_track_point(
    frame_idx: int,
    timestamp: float,
    track_id: int,
    class_name: str,
    bbox: Tuple[int, int, int, int],
    center: Tuple[int, int] = None,
    confidence: float = 0.90,
) -> TrackPoint:
    if center is None:
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return TrackPoint(
        frame_idx=frame_idx,
        timestamp_seconds=timestamp,
        track_id=track_id,
        class_name=class_name,
        raw_class_name=class_name,
        bbox=bbox,
        center=center,
        confidence=confidence,
    )


def make_tracked_object(track_id: int, class_name: str, points: List[TrackPoint]) -> TrackedObject:
    obj = TrackedObject(track_id=track_id, class_name=class_name, raw_class_name=class_name)
    obj.trajectory = list(points)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial & Temporal Tests for OutsideDesignatedAreaRule
# ─────────────────────────────────────────────────────────────────────────────

def test_product_fully_inside_designated_area_no_event():
    """Verifies that a product fully inside the permitted designated zone produces 0 events."""
    rule = OutsideDesignatedAreaRule(
        designated_zones=[(100, 100, 500, 500)],
        margin_px=15.0,
        min_duration_frames=5,
    )
    config = BehaviourConfig()

    # Product centered at (300, 300) well inside [100, 500]
    points = [
        make_track_point(f, f * 0.033, 1, "carton", (250, 250, 350, 350))
        for f in range(10)
    ]
    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=9,
        timestamp=0.30,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 0


def test_product_near_boundary_within_tolerance_no_event():
    """
    Verifies that a product close to the boundary (or slightly outside the nominal border
    but within the configurable tolerance margin) produces 0 events.
    """
    # Zone: [100, 100, 500, 500], margin: 20px -> effective right boundary is 520px
    rule = OutsideDesignatedAreaRule(
        designated_zones=[(100, 100, 500, 500)],
        margin_px=20.0,
        min_duration_frames=5,
    )
    config = BehaviourConfig()

    # Center is at (510, 300) -> 10px beyond nominal border (500), but within 20px margin
    points = [
        make_track_point(f, f * 0.033, 1, "carton", (460, 250, 560, 350))  # center x = 510
        for f in range(10)
    ]
    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=9,
        timestamp=0.30,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 0


def test_product_outside_area_briefly_no_event():
    """
    Verifies temporal reasoning: a product that moves outside the permitted area
    for only a few frames (e.g. 3 frames < min_duration 10) does NOT trigger an event.
    """
    rule = OutsideDesignatedAreaRule(
        designated_zones=[(100, 100, 500, 500)],
        margin_px=15.0,
        min_duration_frames=10,
    )
    config = BehaviourConfig()

    # Frames 0..6: inside zone
    points = [
        make_track_point(f, f * 0.033, 1, "carton", (250, 250, 350, 350))
        for f in range(7)
    ]
    # Frames 7..9 (3 frames): outside zone at x=700
    for f in range(7, 10):
        points.append(make_track_point(f, f * 0.033, 1, "carton", (650, 250, 750, 350)))

    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=9,
        timestamp=0.30,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 0


def test_product_outside_area_for_required_consecutive_frames_triggers_event():
    """
    Verifies that a product sustained outside the permitted area for at least
    min_duration_frames consecutive frames triggers the violation event.
    """
    rule = OutsideDesignatedAreaRule(
        designated_zones=[(100, 100, 500, 500)],
        margin_px=15.0,
        min_duration_frames=10,
    )
    config = BehaviourConfig()

    # Frames 0..11 (12 consecutive frames) positioned outside at center x=700
    points = [
        make_track_point(f, f * 0.033, 1, "carton", (650, 250, 750, 350))
        for f in range(12)
    ]
    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=11,
        timestamp=0.36,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.rule_name == "product_outside_designated_area"
    assert ev.track_id == 1
    assert ev.display_label == "Carton #1"
    assert ev.frame_idx == 11
    assert ev.confidence >= 0.70


def test_different_object_ids_handled_independently():
    """
    Verifies that multiple tracked objects are evaluated independently:
    Carton #1 (outside for 10 frames) triggers an event, while
    Carton #2 (inside the designated zone) produces no event.
    """
    rule = OutsideDesignatedAreaRule(
        designated_zones=[(100, 100, 500, 500)],
        margin_px=15.0,
        min_duration_frames=10,
    )
    config = BehaviourConfig()

    # Carton #1: outside for 10 frames (center x=700)
    pts1 = [
        make_track_point(f, f * 0.033, 1, "carton", (650, 250, 750, 350))
        for f in range(10)
    ]
    carton1 = make_tracked_object(1, "carton", pts1)

    # Carton #2: inside for 10 frames (center x=300)
    pts2 = [
        make_track_point(f, f * 0.033, 2, "carton", (250, 250, 350, 350))
        for f in range(10)
    ]
    carton2 = make_tracked_object(2, "carton", pts2)

    events = rule.evaluate(
        tracked_objects={1: carton1, 2: carton2},
        active_track_ids=[1, 2],
        current_frame_idx=9,
        timestamp=0.30,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 1
    assert events[0].track_id == 1


def test_multiple_designated_zones_support():
    """
    Verifies that when multiple permitted zones are configured (e.g. storage zone + dispatch zone),
    an object located in EITHER zone is permitted without violation.
    """
    rule = OutsideDesignatedAreaRule(
        designated_zones=[(100, 100, 400, 400), (600, 100, 900, 400)],
        margin_px=15.0,
        min_duration_frames=5,
    )
    config = BehaviourConfig()

    # Product is outside Zone 1, but inside Zone 2 (center x=750, y=250)
    points = [
        make_track_point(f, f * 0.033, 1, "carton", (700, 200, 800, 300))
        for f in range(8)
    ]
    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=7,
        timestamp=0.23,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 0


def test_event_contains_correct_metrics_and_explanation():
    """Verifies that the generated BehaviourEvent contains detailed metrics and narrative explanation."""
    rule = OutsideDesignatedAreaRule(
        designated_zones=[(100, 100, 500, 500)],
        margin_px=10.0,
        min_duration_frames=8,
    )
    config = BehaviourConfig()

    # Product center at (610, 300). Effective boundary is 500 + 10 = 510.
    # Distance outside = 610 - 510 = 100px.
    points = [
        make_track_point(f, f * 0.033, 3, "carton", (560, 250, 660, 350))
        for f in range(12)
    ]
    carton = make_tracked_object(3, "carton", points)

    events = rule.evaluate(
        tracked_objects={3: carton},
        active_track_ids=[3],
        current_frame_idx=11,
        timestamp=0.36,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.metrics["consecutive_outside_frames"] == 12
    assert ev.metrics["distance_outside_px"] == pytest.approx(100.0, abs=1.0)
    assert ev.metrics["designated_zone"] == [100, 100, 500, 500]
    assert "outside designated area" in ev.metrics["explanation"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk Engine Integration & Godrej Damage Policy Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_risk_engine_handles_outside_designated_area():
    """
    Verifies that RiskEngine evaluates product_outside_designated_area events:
    - Base score assigned (MEDIUM tier)
    - Metric escalation for prolonged staging or high distance
    - Damage assessment status is strictly POTENTIAL_DAMAGE_RISK
    - Actionable recommendation provided without claiming physical damage
    """
    risk_engine = RiskEngine(config=RiskConfig())

    # Standard outside event: 15 frames, 40px outside
    event_standard = BehaviourEvent(
        event_id="EVT-OUTSIDE-5-F20",
        rule_name="product_outside_designated_area",
        track_id=5,
        display_label="Carton #5",
        class_name="carton",
        frame_idx=20,
        timestamp_seconds=0.66,
        confidence=0.85,
        bbox=(550, 250, 650, 350),
        metrics={"consecutive_outside_frames": 15, "distance_outside_px": 40.0},
    )

    inc_standard = risk_engine.evaluate_event(event_standard)
    assert inc_standard.risk_level == RiskLevel.MEDIUM
    assert 50.0 <= inc_standard.risk_score <= 65.0
    assert inc_standard.damage_assessment == DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK
    assert "Reposition the product within the designated" in inc_standard.recommendation
    assert "confirmed damage" not in inc_standard.explanation.lower()

    # Severe prolonged & distant outside event: 30 frames, 110px outside
    event_severe = BehaviourEvent(
        event_id="EVT-OUTSIDE-6-F50",
        rule_name="product_outside_designated_area",
        track_id=6,
        display_label="Carton #6",
        class_name="carton",
        frame_idx=50,
        timestamp_seconds=1.66,
        confidence=0.88,
        bbox=(800, 250, 900, 350),
        metrics={"consecutive_outside_frames": 30, "distance_outside_px": 110.0},
    )

    inc_severe = risk_engine.evaluate_event(event_severe)
    # Escalated with both prolonged (+12) and high distance (+10) penalties -> score >= 70 (HIGH)
    assert inc_severe.risk_score > inc_standard.risk_score
    assert inc_severe.risk_level == RiskLevel.HIGH
    assert "Prolonged" in inc_severe.explanation
    assert "Substantial displacement" in inc_severe.explanation
    assert inc_severe.damage_assessment == DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK


# ─────────────────────────────────────────────────────────────────────────────
# 3. BehaviourEngine Full Integration
# ─────────────────────────────────────────────────────────────────────────────

def test_behaviour_engine_full_integration_with_outside_rule():
    """Verifies that BehaviourEngine default rule list includes OutsideDesignatedAreaRule."""
    engine = BehaviourEngine()
    rule_names = [r.name for r in engine.rules]
    assert "product_outside_designated_area" in rule_names
    assert len(engine.rules) >= 7
