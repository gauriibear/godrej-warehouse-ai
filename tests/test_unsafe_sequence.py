"""
Unit and integration tests for Godrej Scenario #10: Unsafe Loading/Unloading Sequence.
Validates safe sequence suppression, out-of-order action detection, temporal window enforcement,
isolated action avoidance, independent track handling, metric recording, RiskEngine scoring,
EvidenceCollector compatibility, AI Assistant grounded query answering, and Godrej damage policy compliance.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pytest

from app.assistant.assistant import OperationsAssistant
from app.behaviour.engine import BehaviourConfig, BehaviourEngine, BehaviourEvent
from app.behaviour.sequence import UnsafeLoadingSequenceRule
from app.risk import DamageAssessmentStatus, EvidenceCollector, RiskConfig, RiskEngine, RiskLevel
from app.tracking.tracker import ObjectTracker, TrackedObject, TrackPoint


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
# 1. Temporal & Sequence Tests for UnsafeLoadingSequenceRule
# ─────────────────────────────────────────────────────────────────────────────

def test_correct_safe_sequence_no_event():
    """
    Verifies that a compliant safe sequence produces 0 events:
    1. Approach loading zone (in motion for >= 4 frames)
    2. Stabilized positioning (at rest / low speed for >= 5 frames)
    3. Subsequent handling / placement action.
    """
    rule = UnsafeLoadingSequenceRule(
        loading_zones=[(100, 100, 800, 600)],
        max_sequence_window_frames=45,
        min_approach_frames=4,
        approach_speed_threshold_px_s=60.0,
        min_stabilize_frames=5,
        stabilize_max_speed_px_s=35.0,
        unsafe_action_speed_px_s=180.0,
    )
    config = BehaviourConfig()

    # Step 1: Approach phase (frames 0..4, 5 frames, moving 10px per frame @ 30fps -> 300 px/s)
    points: List[TrackPoint] = []
    for f in range(5):
        x = 200 + f * 10
        points.append(make_track_point(f, f * 0.033, 1, "carton", (x, 300, x + 60, 360)))

    # Step 2: Stabilized positioning phase (frames 5..10, 6 frames, resting at x=250, speed 0)
    for f in range(5, 11):
        points.append(make_track_point(f, f * 0.033, 1, "carton", (250, 300, 310, 360)))

    # Step 3: Handling action (frames 11..13, rapid movement 15px per frame -> 450 px/s)
    for f in range(11, 14):
        x = 250 + (f - 10) * 15
        points.append(make_track_point(f, f * 0.033, 1, "carton", (x, 300, x + 60, 360)))

    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=13,
        timestamp=13 * 0.033,
        video_fps=30.0,
        config=config,
    )

    # Safe stabilization occurred before the action; no violation should trigger
    assert len(events) == 0


def test_unsafe_sequence_triggers_event():
    """
    Verifies that an unsafe out-of-order sequence generates a BehaviourEvent:
    1. Approach loading zone (in motion for >= 4 frames)
    2. Abrupt handling / release action occurs WITHOUT the prerequisite stabilized positioning.
    """
    rule = UnsafeLoadingSequenceRule(
        loading_zones=[(100, 100, 800, 600)],
        max_sequence_window_frames=45,
        min_approach_frames=4,
        approach_speed_threshold_px_s=60.0,
        min_stabilize_frames=5,
        stabilize_max_speed_px_s=35.0,
        unsafe_action_speed_px_s=180.0,
    )
    config = BehaviourConfig()

    # Step 1: Approach phase (frames 0..4, 5 frames, moving 8px per frame @ 30fps -> 240 px/s)
    points: List[TrackPoint] = []
    for f in range(5):
        x = 200 + f * 8
        points.append(make_track_point(f, f * 0.033, 1, "carton", (x, 300, x + 60, 360)))

    # Step 2: Immediate abrupt action (frames 5..7, moving 20px per frame -> 600 px/s, no stabilization)
    for f in range(5, 8):
        x = 240 + (f - 4) * 20
        points.append(make_track_point(f, f * 0.033, 1, "carton", (x, 300, x + 60, 360)))

    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=7,
        timestamp=7 * 0.033,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 1
    ev = events[0]
    assert ev.rule_name == "unsafe_loading_unloading_sequence"
    assert ev.track_id == 1
    assert "approach_loading_zone" in ev.metrics["sequence_steps"]
    assert "abrupt_handling_without_stabilization" in ev.metrics["sequence_steps"]
    assert ev.metrics["skipped_step"] == "stabilized_positioning"


def test_isolated_action_without_full_sequence_no_event():
    """
    Verifies that an isolated action without a preceding approach phase into the
    loading area does NOT trigger a loading sequence violation.
    """
    rule = UnsafeLoadingSequenceRule(
        loading_zones=[(100, 100, 800, 600)],
        max_sequence_window_frames=45,
        min_approach_frames=4,
        approach_speed_threshold_px_s=60.0,
        min_stabilize_frames=5,
        stabilize_max_speed_px_s=35.0,
        unsafe_action_speed_px_s=180.0,
    )
    config = BehaviourConfig()

    # Carton sits completely stationary at (400, 400) for 10 frames (no approach phase)
    points: List[TrackPoint] = []
    for f in range(10):
        points.append(make_track_point(f, f * 0.033, 1, "carton", (400, 400, 480, 480)))

    # Sudden speed spike on last 2 frames
    points.append(make_track_point(10, 10 * 0.033, 1, "carton", (430, 400, 510, 480)))
    points.append(make_track_point(11, 11 * 0.033, 1, "carton", (460, 400, 540, 480)))

    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=11,
        timestamp=11 * 0.033,
        video_fps=30.0,
        config=config,
    )

    # Lacks preceding approach phase in loading area -> Isolated action -> 0 events
    assert len(events) == 0


def test_sequence_occurring_outside_allowed_window_no_event():
    """
    Verifies that if the sequence duration exceeds max_sequence_window_frames
    (e.g. approach completed 60 frames ago, beyond window of 45), no event triggers.
    """
    rule = UnsafeLoadingSequenceRule(
        loading_zones=[(100, 100, 800, 600)],
        max_sequence_window_frames=30,  # Tight 30-frame window
        min_approach_frames=4,
        approach_speed_threshold_px_s=60.0,
        min_stabilize_frames=5,
        stabilize_max_speed_px_s=35.0,
        unsafe_action_speed_px_s=180.0,
    )
    config = BehaviourConfig()

    points: List[TrackPoint] = []
    # Frames 0..4: approach
    for f in range(5):
        x = 100 + f * 10
        points.append(make_track_point(f, f * 0.033, 1, "carton", (x, 300, x + 60, 360)))

    # Frames 5..50 (46 frames): sitting or drifting slowly
    for f in range(5, 51):
        points.append(make_track_point(f, f * 0.033, 1, "carton", (200, 300, 260, 360)))

    # Frame 51..53: sudden action
    for f in range(51, 54):
        x = 200 + (f - 50) * 15
        points.append(make_track_point(f, f * 0.033, 1, "carton", (x, 300, x + 60, 360)))

    carton = make_tracked_object(1, "carton", points)

    events = rule.evaluate(
        tracked_objects={1: carton},
        active_track_ids=[1],
        current_frame_idx=53,
        timestamp=53 * 0.033,
        video_fps=30.0,
        config=config,
    )

    # The approach occurred outside the 30-frame window; window expired -> 0 events
    assert len(events) == 0


def test_different_object_track_ids_handled_independently():
    """
    Verifies that multiple tracked objects are evaluated independently:
    - Carton #1: Unsafe sequence (triggers event)
    - Carton #2: Safe sequence with stabilization (produces 0 events)
    - Carton #3: Stationary resting carton (produces 0 events)
    """
    rule = UnsafeLoadingSequenceRule(
        loading_zones=[(100, 100, 900, 700)],
        max_sequence_window_frames=45,
        min_approach_frames=4,
        approach_speed_threshold_px_s=60.0,
        min_stabilize_frames=4,
        stabilize_max_speed_px_s=35.0,
        unsafe_action_speed_px_s=180.0,
    )
    config = BehaviourConfig()

    # Carton #1: Unsafe sequence (approach 4 frames + sudden action 2 frames)
    c1_pts = [make_track_point(f, f * 0.033, 1, "carton", (100 + f * 10, 200, 160 + f * 10, 260)) for f in range(5)]
    for f in range(5, 8):
        c1_pts.append(make_track_point(f, f * 0.033, 1, "carton", (150 + (f - 4) * 20, 200, 210 + (f - 4) * 20, 260)))
    c1 = make_tracked_object(1, "carton", c1_pts)

    # Carton #2: Safe sequence (approach 4 frames + stable 5 frames + action)
    c2_pts = [make_track_point(f, f * 0.033, 2, "carton", (500 + f * 8, 200, 560 + f * 8, 260)) for f in range(5)]
    for f in range(5, 10):
        c2_pts.append(make_track_point(f, f * 0.033, 2, "carton", (540, 200, 600, 260)))
    for f in range(10, 13):
        c2_pts.append(make_track_point(f, f * 0.033, 2, "carton", (540 + (f - 9) * 15, 200, 600 + (f - 9) * 15, 260)))
    c2 = make_tracked_object(2, "carton", c2_pts)

    # Carton #3: Resting carton alone
    c3_pts = [make_track_point(f, f * 0.033, 3, "carton", (300, 450, 370, 520)) for f in range(8)]
    c3 = make_tracked_object(3, "carton", c3_pts)

    all_objs = {1: c1, 2: c2, 3: c3}
    active_ids = [1, 2, 3]

    events = rule.evaluate(
        tracked_objects=all_objs,
        active_track_ids=active_ids,
        current_frame_idx=7,
        timestamp=7 * 0.033,
        video_fps=30.0,
        config=config,
    )

    # Only Carton #1 triggers
    assert len(events) == 1
    assert events[0].track_id == 1


def test_correct_sequence_steps_and_metrics_recorded():
    """
    Verifies that the generated BehaviourEvent accurately records:
    - sequence_steps: ['approach_loading_zone', 'abrupt_handling_without_stabilization']
    - skipped_step: 'stabilized_positioning'
    - temporal metrics: sequence_duration_frames, action_speed_px_s, window_frames
    - explanation and recommendation text.
    """
    rule = UnsafeLoadingSequenceRule(
        loading_zones=[(100, 100, 800, 600)],
        max_sequence_window_frames=45,
        min_approach_frames=4,
        approach_speed_threshold_px_s=60.0,
        min_stabilize_frames=5,
        stabilize_max_speed_px_s=35.0,
        unsafe_action_speed_px_s=180.0,
    )
    config = BehaviourConfig()

    points = [make_track_point(f, f * 0.033, 4, "carton", (200 + f * 10, 300, 260 + f * 10, 360)) for f in range(5)]
    for f in range(5, 8):
        points.append(make_track_point(f, f * 0.033, 4, "carton", (250 + (f - 4) * 22, 300, 310 + (f - 4) * 22, 360)))
    carton = make_tracked_object(4, "carton", points)

    events = rule.evaluate(
        tracked_objects={4: carton},
        active_track_ids=[4],
        current_frame_idx=7,
        timestamp=7 * 0.033,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 1
    ev = events[0]
    metrics = ev.metrics

    assert "approach_loading_zone" in metrics["sequence_steps"]
    assert "abrupt_handling_without_stabilization" in metrics["sequence_steps"]
    assert metrics["skipped_step"] == "stabilized_positioning"
    assert metrics["sequence_duration_frames"] > 0
    assert metrics["action_speed_px_s"] >= 180.0
    assert metrics["window_frames"] == 45

    # Check required text content
    assert "unsafe loading/unloading sequence was detected" in metrics["explanation"].lower()
    assert "material handling action occurred before the required safe positioning step" in metrics["explanation"].lower()
    assert "prescribed loading/unloading sequence" in metrics["recommendation"].lower()
    assert "safely positioned before continuing" in metrics["recommendation"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. RiskEngine & Evidence Collector Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_risk_engine_handles_unsafe_loading_sequence():
    """
    Verifies that RiskEngine scores unsafe_loading_unloading_sequence:
    - Base score is 66.0 (MEDIUM/HIGH tier)
    - Rapid out-of-order execution (<= 15 frames) adds penalty +10.0 pts -> reaches HIGH risk
    - Produces constructive sequence recommendation
    - Strictly sets damage_assessment to POTENTIAL_DAMAGE_RISK (never CONFIRMED_DAMAGE)
    """
    risk_engine = RiskEngine(config=RiskConfig())

    event = BehaviourEvent(
        event_id="EVT-UNSAFESEQ-10-F82",
        rule_name="unsafe_loading_unloading_sequence",
        track_id=10,
        display_label="Carton #10",
        secondary_track_id=18,
        secondary_label="Person #18",
        class_name="carton",
        frame_idx=82,
        timestamp_seconds=2.73,
        confidence=0.88,
        bbox=(320, 380, 430, 490),
        metrics={
            "sequence_duration_frames": 12,  # <= 15 frames -> rapid penalty +10 pts
            "sequence_steps": ["approach_loading_zone", "abrupt_handling_without_stabilization"],
            "skipped_step": "stabilized_positioning",
            "action_speed_px_s": 220.0,
            "action_type": "abrupt_release_or_movement",
            "max_stable_frames_observed": 1,
            "required_stable_frames": 5,
            "window_frames": 45,
        },
    )

    incident = risk_engine.evaluate_event(event)

    assert incident.rule_name == "unsafe_loading_unloading_sequence"
    assert incident.damage_assessment == DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK
    assert incident.damage_assessment != DamageAssessmentStatus.CONFIRMED_DAMAGE
    assert incident.risk_score >= 75.0  # 66 base + 10 penalty + 3 confidence = 79.0 (HIGH)
    assert incident.risk_level == RiskLevel.HIGH
    assert "sequence" in incident.recommendation.lower()
    assert "Rapid out-of-order execution" in incident.explanation


def test_evidence_collector_handles_sequence_keyframe(tmp_path):
    """Verifies that EvidenceCollector renders keyframe with UnsafeSeq badge."""
    import numpy as np

    collector = EvidenceCollector(output_dir=str(tmp_path))
    risk_engine = RiskEngine(config=RiskConfig())

    event = BehaviourEvent(
        event_id="EVT-UNSAFESEQ-10-F82",
        rule_name="unsafe_loading_unloading_sequence",
        track_id=10,
        display_label="Carton #10",
        class_name="carton",
        frame_idx=82,
        timestamp_seconds=2.73,
        confidence=0.88,
        bbox=(320, 380, 430, 490),
        metrics={"sequence_duration_frames": 14, "action_speed_px_s": 210.0},
    )
    incident = risk_engine.evaluate_event(event)

    dummy_frame = np.full((720, 1280, 3), 60, dtype=np.uint8)
    kf_path = collector.save_keyframe(incident, dummy_frame, annotate=True)

    assert Path(kf_path).exists()
    assert Path(kf_path).stat().st_size > 0


def test_assistant_can_answer_sequence_query():
    """
    Verifies that the grounded AI Operations Assistant correctly answers
    operator questions regarding unsafe loading/unloading sequence incidents.
    """
    risk_engine = RiskEngine(config=RiskConfig())

    event = BehaviourEvent(
        event_id="EVT-UNSAFESEQ-10-F82",
        rule_name="unsafe_loading_unloading_sequence",
        track_id=10,
        display_label="Carton #10",
        secondary_track_id=18,
        secondary_label="Person #18",
        class_name="carton",
        frame_idx=82,
        timestamp_seconds=2.73,
        confidence=0.88,
        bbox=(320, 380, 430, 490),
        metrics={
            "sequence_duration_frames": 12,
            "action_speed_px_s": 220.0,
            "explanation": "An unsafe loading/unloading sequence was detected because the material handling action occurred before the required safe positioning step.",
            "recommendation": "Follow the prescribed loading/unloading sequence and ensure the product is safely positioned before continuing the next handling step.",
        },
    )
    incident = risk_engine.evaluate_event(event)

    assistant = OperationsAssistant.from_incidents([incident])
    response = assistant.ask("Were there any unsafe loading sequence incidents?")

    assert len(response.matched_incidents) == 1
    assert response.matched_incidents[0].rule_name == "unsafe_loading_unloading_sequence"
    assert "INC-UNSAFESEQ-10-F82" in response.answer
    assert "prescribed loading/unloading sequence" in response.answer.lower()
    # Confirms Godrej Responsible AI policy disclaimer is present
    assert "Potential Damage Risk" in response.answer


def test_behaviour_engine_full_integration_with_sequence_rule():
    """
    Verifies that BehaviourEngine coordinates UnsafeLoadingSequenceRule:
    - Rule is loaded by default in engine.rules
    - Frame evaluation and cooldown deduplication operate seamlessly.
    """
    engine = BehaviourEngine()
    rule_names = [r.name for r in engine.rules]
    assert "unsafe_loading_unloading_sequence" in rule_names
    assert len(engine.rules) >= 9

    tracker = ObjectTracker()

    # Populate 5 approach frames + 2 abrupt action frames
    points = [make_track_point(f, f * 0.033, 1, "carton", (200 + f * 10, 300, 260 + f * 10, 360)) for f in range(5)]
    for f in range(5, 8):
        points.append(make_track_point(f, f * 0.033, 1, "carton", (250 + (f - 4) * 22, 300, 310 + (f - 4) * 22, 360)))
    carton = make_tracked_object(1, "carton", points)

    tracker._tracked_objects = {1: carton}

    events_f7 = engine.evaluate_frame(tracker, current_frame_idx=7, timestamp=0.23, fps=30.0)
    seq_events = [e for e in events_f7 if e.rule_name == "unsafe_loading_unloading_sequence"]
    assert len(seq_events) == 1

    # Cooldown suppression on frame 8
    points.append(make_track_point(8, 0.26, 1, "carton", (320, 300, 380, 360)))
    carton.trajectory = points
    events_f8 = engine.evaluate_frame(tracker, current_frame_idx=8, timestamp=0.26, fps=30.0)
    seq_events_f8 = [e for e in events_f8 if e.rule_name == "unsafe_loading_unloading_sequence"]
    assert len(seq_events_f8) == 0
