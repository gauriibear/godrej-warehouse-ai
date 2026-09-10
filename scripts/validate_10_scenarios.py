"""
Functional Validation Test Runner for all 10 Godrej Warehouse AI Behaviours.
Exercises the actual production BehaviourEngine and RiskEngine pipelines on:
 1. Product Dropped
 2. Product Thrown
 3. Product Dragged
 4. Product Pushed
 5. Product Rolled
 6. Unstable Stacking
 7. Pallet Overhang
 8. Product Outside Designated Area
 9. Product Handled Without Required Equipment
 10. Unsafe Loading/Unloading Sequence

Validates positive detection, negative (safe) suppression, risk scoring, explainability,
and strict adherence to the Godrej Responsible AI damage assessment policy.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.behaviour.engine import BehaviourConfig, BehaviourEngine, BehaviourEvent
from app.risk import DamageAssessmentStatus, IncidentRecord, RiskConfig, RiskEngine, RiskLevel
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


@dataclass
class ScenarioResult:
    scenario_num: int
    name: str
    expected_rule: str
    detected: bool
    event_id: str
    risk_level: str
    risk_score: float
    frame_idx: int
    negative_suppressed: bool
    damage_assessment: str
    explanation: str
    recommendation: str


def run_scenario_1_drop(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 1: Product Dropped"""
    # Positive case: unheld vertical descent and ground impact
    pts = [
        make_track_point(0, 0.000, 1, "carton", (100, 50, 160, 100)),
        make_track_point(1, 0.033, 1, "carton", (100, 100, 160, 160)),
        make_track_point(2, 0.067, 1, "carton", (100, 170, 160, 240)),
        make_track_point(3, 0.100, 1, "carton", (100, 230, 160, 300)),
        make_track_point(4, 0.133, 1, "carton", (100, 230, 160, 300)),
        make_track_point(5, 0.167, 1, "carton", (100, 230, 160, 300)),
        make_track_point(6, 0.200, 1, "carton", (100, 230, 160, 300)),
    ]
    carton = make_tracked_object(1, "carton", pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {1: carton}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=6, timestamp=0.200, fps=30.0)
    drop_events = [e for e in events if e.rule_name == "product_dropped"]
    detected = len(drop_events) >= 1
    ev = drop_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: worker holds carton during descent
    worker = make_tracked_object(2, "person", [make_track_point(6, 0.200, 2, "person", (80, 100, 190, 350))])
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {1: carton, 2: worker}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=6, timestamp=0.200, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "product_dropped"]) == 0

    return ScenarioResult(
        scenario_num=1,
        name="Product Dropped",
        expected_rule="product_dropped",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_2_throw(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 2: Product Thrown"""
    pts = [
        make_track_point(0, 0.000, 5, "carton", (100, 200, 160, 260)),
        make_track_point(1, 0.033, 5, "carton", (135, 185, 195, 245)),
        make_track_point(2, 0.067, 5, "carton", (170, 175, 230, 235)),
        make_track_point(3, 0.100, 5, "carton", (205, 180, 265, 240)),
        make_track_point(4, 0.133, 5, "carton", (245, 200, 305, 260)),
        make_track_point(5, 0.167, 5, "carton", (290, 230, 350, 290)),
    ]
    carton = make_tracked_object(5, "carton", pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {5: carton}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=5, timestamp=0.167, fps=30.0)
    throw_events = [e for e in events if e.rule_name == "product_thrown"]
    detected = len(throw_events) >= 1
    ev = throw_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: worker carrying package
    worker = make_tracked_object(2, "person", [make_track_point(5, 0.167, 2, "person", (270, 150, 360, 340))])
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {5: carton, 2: worker}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=5, timestamp=0.167, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "product_thrown"]) == 0

    return ScenarioResult(
        scenario_num=2,
        name="Product Thrown",
        expected_rule="product_thrown",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_3_drag(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 3: Product Dragged"""
    c_pts = []
    w_pts = []
    for i in range(12):
        t = i * 0.033
        cx = 100 + i * 15
        wx = 120 + i * 15
        c_pts.append(make_track_point(i, t, 3, "carton", (cx - 30, 410, cx + 30, 460)))
        w_pts.append(make_track_point(i, t, 2, "person", (wx - 30, 300, wx + 30, 500)))

    carton = make_tracked_object(3, "carton", c_pts)
    worker = make_tracked_object(2, "person", w_pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {3: carton, 2: worker}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=11, timestamp=11 * 0.033, fps=30.0)
    drag_events = [e for e in events if e.rule_name == "product_dragged"]
    detected = len(drag_events) >= 1
    ev = drag_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: resting carton with walking worker
    c_rest_pts = [make_track_point(i, i * 0.033, 3, "carton", (100, 410, 160, 460)) for i in range(12)]
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {3: make_tracked_object(3, "carton", c_rest_pts), 2: worker}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=11, timestamp=11 * 0.033, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "product_dragged"]) == 0

    return ScenarioResult(
        scenario_num=3,
        name="Product Dragged",
        expected_rule="product_dragged",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_4_push(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 4: Product Pushed"""
    c_pts = [
        make_track_point(0, 0.000, 4, "carton", (200, 400, 260, 460)),
        make_track_point(1, 0.033, 4, "carton", (200, 400, 260, 460)),
        make_track_point(2, 0.067, 4, "carton", (201, 400, 261, 460)),
        make_track_point(3, 0.100, 4, "carton", (230, 400, 290, 460)),  # Impulse spike
    ]
    w_pts = [make_track_point(3, 0.100, 2, "person", (160, 320, 220, 460))]
    carton = make_tracked_object(4, "carton", c_pts)
    worker = make_tracked_object(2, "person", w_pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {4: carton, 2: worker}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=3, timestamp=0.100, fps=30.0)
    push_events = [e for e in events if e.rule_name == "product_pushed"]
    detected = len(push_events) >= 1
    ev = push_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: slow movement without impulse
    c_slow_pts = [make_track_point(i, i * 0.033, 4, "carton", (200 + i * 2, 400, 260 + i * 2, 460)) for i in range(4)]
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {4: make_tracked_object(4, "carton", c_slow_pts), 2: worker}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=3, timestamp=0.100, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "product_pushed"]) == 0

    return ScenarioResult(
        scenario_num=4,
        name="Product Pushed",
        expected_rule="product_pushed",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_5_roll(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 5: Product Rolled"""
    pts = []
    for i in range(16):
        t = i * 0.033
        x = 100 + i * 8
        bbox = (x, 400, x + 120, 480) if (i // 4) % 2 == 0 else (x, 380, x + 70, 480)
        pts.append(make_track_point(i, t, 6, "carton", bbox))

    carton = make_tracked_object(6, "carton", pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {6: carton}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=15, timestamp=15 * 0.033, fps=30.0)
    roll_events = [e for e in events if e.rule_name == "product_rolled"]
    detected = len(roll_events) >= 1
    ev = roll_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: smooth translation without flipping AR
    pts_neg = [make_track_point(i, i * 0.033, 6, "carton", (100 + i * 8, 400, 200 + i * 8, 480)) for i in range(16)]
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {6: make_tracked_object(6, "carton", pts_neg)}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=15, timestamp=15 * 0.033, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "product_rolled"]) == 0

    return ScenarioResult(
        scenario_num=5,
        name="Product Rolled",
        expected_rule="product_rolled",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_6_stacking(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 6: Unstable Stacking"""
    b_pts = [make_track_point(i, i * 0.033, 10, "carton", (200, 400, 400, 500)) for i in range(12)]
    a_pts = [make_track_point(i, i * 0.033, 11, "carton", (290, 300, 490, 400)) for i in range(12)]
    carton_b = make_tracked_object(10, "carton", b_pts)
    carton_a = make_tracked_object(11, "carton", a_pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {10: carton_b, 11: carton_a}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=11, timestamp=11 * 0.033, fps=30.0)
    stack_events = [e for e in events if e.rule_name == "unstable_stacking"]
    detected = len(stack_events) >= 1
    ev = stack_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: squarely aligned stack
    a_aligned_pts = [make_track_point(i, i * 0.033, 11, "carton", (200, 300, 400, 400)) for i in range(12)]
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {10: carton_b, 11: make_tracked_object(11, "carton", a_aligned_pts)}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=11, timestamp=11 * 0.033, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "unstable_stacking"]) == 0

    return ScenarioResult(
        scenario_num=6,
        name="Unstable Stacking",
        expected_rule="unstable_stacking",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_7_overhang(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 7: Pallet Overhang"""
    p_pts = [make_track_point(i, i * 0.033, 20, "pallet", (200, 500, 600, 580)) for i in range(12)]
    c_pts = [make_track_point(i, i * 0.033, 21, "carton", (140, 420, 300, 500)) for i in range(12)]
    pallet = make_tracked_object(20, "pallet", p_pts)
    carton = make_tracked_object(21, "carton", c_pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {20: pallet, 21: carton}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=11, timestamp=11 * 0.033, fps=30.0)
    overhang_events = [e for e in events if e.rule_name == "pallet_overhang"]
    detected = len(overhang_events) >= 1
    ev = overhang_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: carton safely inside pallet perimeter
    c_safe_pts = [make_track_point(i, i * 0.033, 21, "carton", (250, 420, 450, 500)) for i in range(12)]
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {20: pallet, 21: make_tracked_object(21, "carton", c_safe_pts)}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=11, timestamp=11 * 0.033, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "pallet_overhang"]) == 0

    return ScenarioResult(
        scenario_num=7,
        name="Pallet Overhang",
        expected_rule="pallet_overhang",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_8_outside_area(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 8: Product Outside Designated Area"""
    # Zone is [(100, 150, 1180, 650)]; box at (1250, 300, 1350, 400) is outside
    pts = [make_track_point(i, i * 0.033, 31, "carton", (1250, 300, 1350, 400)) for i in range(12)]
    carton = make_tracked_object(31, "carton", pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {31: carton}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=11, timestamp=11 * 0.033, fps=30.0)
    outside_events = [e for e in events if e.rule_name == "product_outside_designated_area"]
    detected = len(outside_events) >= 1
    ev = outside_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: carton inside zone
    pts_in = [make_track_point(i, i * 0.033, 31, "carton", (300, 300, 400, 400)) for i in range(12)]
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {31: make_tracked_object(31, "carton", pts_in)}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=11, timestamp=11 * 0.033, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "product_outside_designated_area"]) == 0

    return ScenarioResult(
        scenario_num=8,
        name="Product Outside Designated Area",
        expected_rule="product_outside_designated_area",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_9_no_equipment(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 9: Product Handled Without Required Equipment"""
    c_pts = [make_track_point(i, i * 0.033, 41, "carton", (400, 400, 500, 500)) for i in range(15)]
    w_pts = [make_track_point(i, i * 0.033, 42, "person", (360, 400, 440, 520)) for i in range(15)]
    carton = make_tracked_object(41, "carton", c_pts)
    worker = make_tracked_object(42, "person", w_pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {41: carton, 42: worker}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=14, timestamp=14 * 0.033, fps=30.0)
    equip_events = [e for e in events if e.rule_name == "product_handled_without_required_equipment"]
    detected = len(equip_events) >= 1
    ev = equip_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: forklift assisting nearby (distance <= 250 px)
    fl_pts = [make_track_point(i, i * 0.033, 43, "equipment/forklift", (480, 400, 620, 520)) for i in range(15)]
    forklift = make_tracked_object(43, "equipment/forklift", fl_pts)
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {41: carton, 42: worker, 43: forklift}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=14, timestamp=14 * 0.033, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "product_handled_without_required_equipment"]) == 0

    return ScenarioResult(
        scenario_num=9,
        name="Product Handled Without Required Equipment",
        expected_rule="product_handled_without_required_equipment",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def run_scenario_10_unsafe_sequence(engine: BehaviourEngine, risk_engine: RiskEngine) -> ScenarioResult:
    """Scenario 10: Unsafe Loading/Unloading Sequence"""
    # Approach for 5 frames + abrupt action for 3 frames without stabilization
    pts = [make_track_point(f, f * 0.033, 51, "carton", (200 + f * 8, 300, 260 + f * 8, 360)) for f in range(5)]
    for f in range(5, 8):
        pts.append(make_track_point(f, f * 0.033, 51, "carton", (240 + (f - 4) * 20, 300, 300 + (f - 4) * 20, 360)))
    carton = make_tracked_object(51, "carton", pts)
    tracker = ObjectTracker()
    tracker._tracked_objects = {51: carton}
    engine.reset()
    events = engine.evaluate_frame(tracker, current_frame_idx=7, timestamp=7 * 0.033, fps=30.0)
    seq_events = [e for e in events if e.rule_name == "unsafe_loading_unloading_sequence"]
    detected = len(seq_events) >= 1
    ev = seq_events[0] if detected else None
    inc = risk_engine.evaluate_event(ev) if ev else None

    # Negative case: approach -> stabilized positioning for 6 frames -> placement
    pts_safe = [make_track_point(f, f * 0.033, 51, "carton", (200 + f * 10, 300, 260 + f * 10, 360)) for f in range(5)]
    for f in range(5, 11):
        pts_safe.append(make_track_point(f, f * 0.033, 51, "carton", (250, 300, 310, 360)))
    for f in range(11, 14):
        pts_safe.append(make_track_point(f, f * 0.033, 51, "carton", (250 + (f - 10) * 15, 300, 310 + (f - 10) * 15, 360)))
    tracker_neg = ObjectTracker()
    tracker_neg._tracked_objects = {51: make_tracked_object(51, "carton", pts_safe)}
    engine.reset()
    neg_events = engine.evaluate_frame(tracker_neg, current_frame_idx=13, timestamp=13 * 0.033, fps=30.0)
    neg_suppressed = len([e for e in neg_events if e.rule_name == "unsafe_loading_unloading_sequence"]) == 0

    return ScenarioResult(
        scenario_num=10,
        name="Unsafe Loading/Unloading Sequence",
        expected_rule="unsafe_loading_unloading_sequence",
        detected=detected,
        event_id=ev.event_id if ev else "NONE",
        risk_level=inc.risk_level.value if inc else "NONE",
        risk_score=inc.risk_score if inc else 0.0,
        frame_idx=ev.frame_idx if ev else -1,
        negative_suppressed=neg_suppressed,
        damage_assessment=inc.damage_assessment.value if inc else "NONE",
        explanation=inc.explanation if inc else "NONE",
        recommendation=inc.recommendation if inc else "NONE",
    )


def validate_all_10_scenarios() -> List[ScenarioResult]:
    engine = BehaviourEngine()
    risk_engine = RiskEngine()

    runners = [
        run_scenario_1_drop,
        run_scenario_2_throw,
        run_scenario_3_drag,
        run_scenario_4_push,
        run_scenario_5_roll,
        run_scenario_6_stacking,
        run_scenario_7_overhang,
        run_scenario_8_outside_area,
        run_scenario_9_no_equipment,
        run_scenario_10_unsafe_sequence,
    ]

    results: List[ScenarioResult] = []
    for runner in runners:
        res = runner(engine, risk_engine)
        results.append(res)

    return results


def print_validation_report(results: List[ScenarioResult]) -> None:
    print("\n" + "=" * 115)
    print(" GODREJ WAREHOUSE AI - 10 PREDEFINED SCENARIOS PRODUCTION FUNCTIONAL VALIDATION REPORT")
    print("=" * 115)
    header = (
        f"{'#':<3} | {'Scenario Name':<42} | {'Detected':<8} | {'Negative Supp':<13} | "
        f"{'Risk Level':<10} | {'Score':<6} | {'Damage Policy Status':<22}"
    )
    print(header)
    print("-" * 115)

    all_passed = True
    for r in results:
        det_str = "YES" if r.detected else "FAIL"
        neg_str = "YES (Pass)" if r.negative_suppressed else "FAIL"
        passed = r.detected and r.negative_suppressed and r.damage_assessment == "POTENTIAL_DAMAGE_RISK"
        if not passed:
            all_passed = False

        print(
            f"{r.scenario_num:<3} | {r.name:<42} | {det_str:<8} | {neg_str:<13} | "
            f"{r.risk_level:<10} | {r.risk_score:<6.1f} | {r.damage_assessment:<22}"
        )
        print(f"    - Event ID:       {r.event_id} (Frame #{r.frame_idx})")
        print(f"    - Hazard Reason:  {r.explanation[:95]}...")
        print(f"    - Recommendation: {r.recommendation[:95]}...")
        print("-" * 115)

    print(f"Summary: {sum(1 for r in results if r.detected and r.negative_suppressed)}/10 Scenarios Fully Functional")
    print(f"Status:  {'ALL 10 SCENARIOS VERIFIED FUNCTIONAL AND COMPLIANT' if all_passed else 'SOME SCENARIOS FAILED'}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    results = validate_all_10_scenarios()
    print_validation_report(results)
