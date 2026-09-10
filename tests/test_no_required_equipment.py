"""
Unit and integration tests for Godrej Scenario #9: Product Handled Without Required Equipment.
Validates worker-carton spatial proximity, equipment presence suppression, temporal duration persistence,
independent object tracking, metric generation, RiskEngine integration, and Godrej damage policy compliance.
"""

from typing import List, Optional, Tuple
import pytest

from app.behaviour.engine import BehaviourConfig, BehaviourEngine, BehaviourEvent
from app.behaviour.equipment import NoRequiredEquipmentRule
from app.risk import DamageAssessmentStatus, RiskConfig, RiskEngine, RiskLevel
from app.tracking.tracker import TrackedObject, TrackPoint


def make_track_point(
    frame_idx: int,
    timestamp: float,
    track_id: int,
    class_name: str,
    bbox: Tuple[int, int, int, int],
    center: Optional[Tuple[int, int]] = None,
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
# 1. Spatial & Temporal Tests for NoRequiredEquipmentRule
# ─────────────────────────────────────────────────────────────────────────────

def test_equipment_present_suppresses_violation():
    """
    Verifies that when a worker handles a carton in the presence of required
    handling equipment (e.g. forklift within 250px), no violation is triggered.
    """
    rule = NoRequiredEquipmentRule(
        required_equipment=["forklift", "equipment/forklift"],
        max_handling_distance_px=120.0,
        equipment_proximity_distance_px=250.0,
        min_duration_frames=10,
    )
    config = BehaviourConfig()

    # Carton at (400, 400, 500, 500)
    carton_points = [
        make_track_point(f, f * 0.033, 1, "carton", (400, 400, 500, 500))
        for f in range(15)
    ]
    carton = make_tracked_object(1, "carton", carton_points)

    # Worker at (360, 400, 440, 520) - close proximity (handling)
    worker_points = [
        make_track_point(f, f * 0.033, 2, "person", (360, 400, 440, 520))
        for f in range(15)
    ]
    worker = make_tracked_object(2, "person", worker_points)

    # Forklift at (520, 380, 680, 520) - 20px from carton (equipment present)
    forklift_points = [
        make_track_point(f, f * 0.033, 3, "equipment/forklift", (520, 380, 680, 520))
        for f in range(15)
    ]
    forklift = make_tracked_object(3, "equipment/forklift", forklift_points)

    events = rule.evaluate(
        tracked_objects={1: carton, 2: worker, 3: forklift},
        active_track_ids=[1, 2, 3],
        current_frame_idx=14,
        timestamp=0.46,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 0


def test_manual_handling_without_equipment_triggers_event():
    """
    Verifies that when a worker handles a carton manually for >= min_duration frames
    with no required handling equipment nearby, a violation event is generated.
    """
    rule = NoRequiredEquipmentRule(
        required_equipment=["forklift", "equipment/forklift"],
        max_handling_distance_px=120.0,
        equipment_proximity_distance_px=250.0,
        min_duration_frames=10,
    )
    config = BehaviourConfig()

    carton_points = [
        make_track_point(f, f * 0.033, 1, "carton", (400, 400, 500, 500))
        for f in range(15)
    ]
    carton = make_tracked_object(1, "carton", carton_points)

    worker_points = [
        make_track_point(f, f * 0.033, 2, "person", (360, 400, 440, 520))
        for f in range(15)
    ]
    worker = make_tracked_object(2, "person", worker_points)

    events = rule.evaluate(
        tracked_objects={1: carton, 2: worker},
        active_track_ids=[1, 2],
        current_frame_idx=14,
        timestamp=0.46,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.rule_name == "product_handled_without_required_equipment"
    assert ev.track_id == 1
    assert ev.secondary_track_id == 2
    assert ev.metrics["consecutive_handling_frames"] >= 10
    assert ev.metrics["equipment_present"] is False


def test_brief_handling_below_duration_no_event():
    """
    Verifies that brief worker interaction below the min_duration_frames threshold
    (e.g. 5 frames < 10 frames) does NOT trigger an equipment violation.
    """
    rule = NoRequiredEquipmentRule(
        required_equipment=["forklift", "equipment/forklift"],
        max_handling_distance_px=120.0,
        equipment_proximity_distance_px=250.0,
        min_duration_frames=10,
    )
    config = BehaviourConfig()

    # Carton present for 15 frames
    carton_points = [
        make_track_point(f, f * 0.033, 1, "carton", (400, 400, 500, 500))
        for f in range(15)
    ]
    carton = make_tracked_object(1, "carton", carton_points)

    # Worker only present / close for frames 10..14 (5 frames)
    worker_points = [
        make_track_point(f, f * 0.033, 2, "person", (360, 400, 440, 520))
        for f in range(10, 15)
    ]
    worker = make_tracked_object(2, "person", worker_points)

    events = rule.evaluate(
        tracked_objects={1: carton, 2: worker},
        active_track_ids=[1, 2],
        current_frame_idx=14,
        timestamp=0.46,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 0


def test_stationary_untouched_carton_without_equipment_no_event():
    """
    Verifies that a resting carton with no worker nearby produces 0 events,
    even if handling equipment is completely absent from the scene.
    """
    rule = NoRequiredEquipmentRule(
        required_equipment=["forklift", "equipment/forklift"],
        max_handling_distance_px=120.0,
        equipment_proximity_distance_px=250.0,
        min_duration_frames=10,
    )
    config = BehaviourConfig()

    # Carton at (100, 100, 200, 200)
    carton_points = [
        make_track_point(f, f * 0.033, 1, "carton", (100, 100, 200, 200))
        for f in range(15)
    ]
    carton = make_tracked_object(1, "carton", carton_points)

    # Worker far away at (800, 500, 880, 650) -> distance > 600px
    worker_points = [
        make_track_point(f, f * 0.033, 2, "person", (800, 500, 880, 650))
        for f in range(15)
    ]
    worker = make_tracked_object(2, "person", worker_points)

    events = rule.evaluate(
        tracked_objects={1: carton, 2: worker},
        active_track_ids=[1, 2],
        current_frame_idx=14,
        timestamp=0.46,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 0


def test_custom_required_equipment_classes():
    """
    Verifies that required equipment types are fully configurable.
    If 'overhead_crane' is required, a standard forklift does not satisfy the requirement.
    """
    rule = NoRequiredEquipmentRule(
        required_equipment=["overhead_crane"],
        max_handling_distance_px=120.0,
        equipment_proximity_distance_px=250.0,
        min_duration_frames=5,
    )
    config = BehaviourConfig()

    carton_points = [
        make_track_point(f, f * 0.033, 1, "carton", (400, 400, 500, 500))
        for f in range(10)
    ]
    carton = make_tracked_object(1, "carton", carton_points)

    worker_points = [
        make_track_point(f, f * 0.033, 2, "person", (360, 400, 440, 520))
        for f in range(10)
    ]
    worker = make_tracked_object(2, "person", worker_points)

    # A forklift is nearby, but the rule requires an 'overhead_crane'
    forklift_points = [
        make_track_point(f, f * 0.033, 3, "equipment/forklift", (420, 400, 560, 520))
        for f in range(10)
    ]
    forklift = make_tracked_object(3, "equipment/forklift", forklift_points)

    events = rule.evaluate(
        tracked_objects={1: carton, 2: worker, 3: forklift},
        active_track_ids=[1, 2, 3],
        current_frame_idx=9,
        timestamp=0.30,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 1
    assert events[0].rule_name == "product_handled_without_required_equipment"
    assert "overhead_crane" in events[0].metrics["required_equipment"]


def test_multiple_objects_handled_independently():
    """
    Verifies that multiple cartons are evaluated independently by track ID:
    - Carton #1: manual handling without equipment (triggers violation)
    - Carton #2: manual handling with forklift present (suppressed)
    - Carton #3: resting carton without worker (suppressed)
    """
    rule = NoRequiredEquipmentRule(
        required_equipment=["forklift"],
        max_handling_distance_px=120.0,
        equipment_proximity_distance_px=250.0,
        min_duration_frames=8,
    )
    config = BehaviourConfig()

    # Carton #1 + Worker #10 (unassisted)
    c1_pts = [make_track_point(f, f * 0.033, 1, "carton", (100, 200, 180, 280)) for f in range(10)]
    w10_pts = [make_track_point(f, f * 0.033, 10, "person", (120, 200, 190, 310)) for f in range(10)]
    c1 = make_tracked_object(1, "carton", c1_pts)
    w10 = make_tracked_object(10, "person", w10_pts)

    # Carton #2 + Worker #20 + Forklift #30 (assisted)
    c2_pts = [make_track_point(f, f * 0.033, 2, "carton", (700, 200, 780, 280)) for f in range(10)]
    w20_pts = [make_track_point(f, f * 0.033, 20, "person", (720, 200, 790, 310)) for f in range(10)]
    fl30_pts = [make_track_point(f, f * 0.033, 30, "forklift", (710, 190, 850, 330)) for f in range(10)]
    c2 = make_tracked_object(2, "carton", c2_pts)
    w20 = make_tracked_object(20, "person", w20_pts)
    fl30 = make_tracked_object(30, "forklift", fl30_pts)

    # Carton #3 (stationary alone)
    c3_pts = [make_track_point(f, f * 0.033, 3, "carton", (400, 500, 480, 580)) for f in range(10)]
    c3 = make_tracked_object(3, "carton", c3_pts)

    all_objs = {1: c1, 2: c2, 3: c3, 10: w10, 20: w20, 30: fl30}
    active_ids = [1, 2, 3, 10, 20, 30]

    events = rule.evaluate(
        tracked_objects=all_objs,
        active_track_ids=active_ids,
        current_frame_idx=9,
        timestamp=0.30,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 1
    assert events[0].track_id == 1
    assert events[0].secondary_track_id == 10


def test_event_metrics_and_explanation_contents():
    """
    Verifies that generated BehaviourEvent contains proper metrics:
    consecutive_handling_frames, worker_distance_px, required_equipment list, and explanation string.
    """
    rule = NoRequiredEquipmentRule(
        required_equipment=["forklift", "trolley"],
        max_handling_distance_px=100.0,
        equipment_proximity_distance_px=200.0,
        min_duration_frames=6,
    )
    config = BehaviourConfig()

    carton_points = [
        make_track_point(f, f * 0.033, 5, "carton", (300, 300, 400, 400))
        for f in range(8)
    ]
    worker_points = [
        make_track_point(f, f * 0.033, 7, "person", (290, 300, 370, 420))
        for f in range(8)
    ]
    carton = make_tracked_object(5, "carton", carton_points)
    worker = make_tracked_object(7, "person", worker_points)

    events = rule.evaluate(
        tracked_objects={5: carton, 7: worker},
        active_track_ids=[5, 7],
        current_frame_idx=7,
        timestamp=0.23,
        video_fps=30.0,
        config=config,
    )
    assert len(events) == 1
    ev = events[0]
    metrics = ev.metrics
    assert "consecutive_handling_frames" in metrics
    assert metrics["consecutive_handling_frames"] == 8
    assert "worker_distance_px" in metrics
    assert "required_equipment" in metrics
    assert metrics["equipment_present"] is False
    assert "explanation" in metrics
    assert "Person #7" in metrics["explanation"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. RiskEngine Integration & Godrej Policy Compliance Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_risk_engine_handles_no_required_equipment():
    """
    Verifies that RiskEngine evaluates product_handled_without_required_equipment:
    - Base score is 58.0 (MEDIUM tier)
    - Prolonged handling (>= 25 frames) adds penalty +12.0 pts
    - Produces constructive mitigation recommendation
    - Strictly sets damage_assessment to POTENTIAL_DAMAGE_RISK
    """
    risk_engine = RiskEngine(config=RiskConfig())

    event = BehaviourEvent(
        event_id="EVT-NOEQUIP-9-15-F40",
        rule_name="product_handled_without_required_equipment",
        track_id=9,
        display_label="Carton #9",
        secondary_track_id=15,
        secondary_label="Person #15",
        class_name="carton",
        frame_idx=40,
        timestamp_seconds=1.33,
        confidence=0.88,
        bbox=(400, 400, 500, 500),
        metrics={
            "consecutive_handling_frames": 28,  # >= 25 frames -> prolonged penalty +12 pts
            "worker_distance_px": 35.0,
            "required_equipment": ["forklift", "trolley"],
            "equipment_present": False,
        },
    )

    incident = risk_engine.evaluate_event(event)

    assert incident.rule_name == "product_handled_without_required_equipment"
    assert incident.damage_assessment == DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK
    assert incident.damage_assessment != DamageAssessmentStatus.CONFIRMED_DAMAGE
    assert incident.risk_score >= 70.0  # 58 base + 12 penalty + 3 confidence = 73.0 (HIGH)
    assert incident.risk_level == RiskLevel.HIGH
    assert "equipment" in incident.recommendation.lower()
    assert "Prolonged manual handling" in incident.explanation


def test_behaviour_engine_full_integration_with_equipment_rule():
    """
    Verifies that BehaviourEngine coordinates NoRequiredEquipmentRule:
    - Rule is loaded by default
    - Engine detects violation and applies cooldown deduplication
    """
    engine = BehaviourEngine()
    rule_names = [r.name for r in engine.rules]
    assert "product_handled_without_required_equipment" in rule_names

    # Run through engine evaluate_frame simulation
    from app.tracking.tracker import ObjectTracker

    tracker = ObjectTracker()

    # Manually populate tracker with 12 frames of manual unassisted handling
    c_pts = [make_track_point(f, f * 0.033, 1, "carton", (200, 200, 300, 300)) for f in range(12)]
    w_pts = [make_track_point(f, f * 0.033, 2, "person", (210, 200, 290, 320)) for f in range(12)]

    carton = make_tracked_object(1, "carton", c_pts)
    worker = make_tracked_object(2, "person", w_pts)

    tracker._tracked_objects = {1: carton, 2: worker}

    events_f11 = engine.evaluate_frame(tracker, current_frame_idx=11, timestamp=0.36, fps=30.0)
    equipment_events = [e for e in events_f11 if e.rule_name == "product_handled_without_required_equipment"]
    assert len(equipment_events) == 1

    # In frame 12 (within cooldown window), should NOT emit duplicate event
    c_pts.append(make_track_point(12, 0.40, 1, "carton", (200, 200, 300, 300)))
    w_pts.append(make_track_point(12, 0.40, 2, "person", (210, 200, 290, 320)))
    carton.trajectory = c_pts
    worker.trajectory = w_pts

    events_f12 = engine.evaluate_frame(tracker, current_frame_idx=12, timestamp=0.40, fps=30.0)
    equipment_events_f12 = [e for e in events_f12 if e.rule_name == "product_handled_without_required_equipment"]
    assert len(equipment_events_f12) == 0
