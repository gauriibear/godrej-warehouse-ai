"""
Unit and integration tests for Stage 4: Temporal Behaviour Rules & Spatial Reasoning Engine.
Validates all 7 material handling violation rules, false-positive suppression,
kinematic heuristics, and engine event deduplication.
"""

from pathlib import Path
import pytest
from typing import List, Tuple

from app.behaviour.engine import BehaviourEngine, BehaviourConfig, BehaviourEvent
from app.behaviour.drop import DropRule
from app.behaviour.throw import ThrowRule
from app.behaviour.drag import DragRule
from app.behaviour.push import PushRule
from app.behaviour.roll import RollRule
from app.behaviour.stacking import StackingRule
from app.tracking.tracker import TrackedObject, TrackPoint, ObjectTracker
from app.detection.detector import Detection, DetectionResult


# ─────────────────────────── Test Helper Functions ──────────────────────────

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


# ───────────────────────── 1. Product Dropped Tests ─────────────────────────

def test_drop_rule_triggers_on_rapid_unheld_descent():
    """
    Simulates a carton falling vertically at high speed and arresting at floor level.
    Asserts product_dropped violation event is triggered.
    """
    rule = DropRule()
    config = BehaviourConfig()

    points = [
        make_track_point(0, 0.000, 1, "carton", (100, 50, 160, 100)),
        make_track_point(1, 0.033, 1, "carton", (100, 100, 160, 160)),
        make_track_point(2, 0.067, 1, "carton", (100, 170, 160, 240)),  # Descent: vy ~ 2100 px/s
        make_track_point(3, 0.100, 1, "carton", (100, 230, 160, 300)),
        make_track_point(4, 0.133, 1, "carton", (100, 230, 160, 300)),  # Stationary at impact
        make_track_point(5, 0.167, 1, "carton", (100, 230, 160, 300)),
        make_track_point(6, 0.200, 1, "carton", (100, 230, 160, 300)),
    ]
    carton = make_tracked_object(1, "carton", points)
    tracked_objects = {1: carton}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[1],
        current_frame_idx=6,
        timestamp=0.200,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 1
    assert events[0].rule_name == "product_dropped"
    assert events[0].track_id == 1
    assert "vertical_velocity_px_s" in events[0].metrics
    assert events[0].metrics["vertical_velocity_px_s"] >= config.drop_min_vertical_velocity_px_s


def test_drop_rule_suppressed_when_held_by_worker():
    """
    Simulates vertical downward motion but worker is holding the carton (IoU overlap).
    Asserts false-positive suppression works (0 events).
    """
    rule = DropRule()
    config = BehaviourConfig()

    carton_pts = [
        make_track_point(0, 0.000, 1, "carton", (100, 50, 160, 100)),
        make_track_point(1, 0.033, 1, "carton", (100, 110, 160, 160)),
        make_track_point(2, 0.067, 1, "carton", (100, 180, 160, 240)),
        make_track_point(3, 0.100, 1, "carton", (100, 240, 160, 300)),
        make_track_point(4, 0.133, 1, "carton", (100, 240, 160, 300)),
        make_track_point(5, 0.167, 1, "carton", (100, 240, 160, 300)),
        make_track_point(6, 0.200, 1, "carton", (100, 240, 160, 300)),
    ]
    carton = make_tracked_object(1, "carton", carton_pts)

    # Worker holding the carton (overlapping bounding box)
    worker_pts = [
        make_track_point(6, 0.200, 2, "person", (80, 100, 190, 350)),
    ]
    worker = make_tracked_object(2, "person", worker_pts)

    tracked_objects = {1: carton, 2: worker}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[1, 2],
        current_frame_idx=6,
        timestamp=0.200,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 0, "Expected drop to be suppressed when worker is holding the carton"


# ───────────────────────── 2. Product Thrown Tests ──────────────────────────

def test_throw_rule_triggers_on_ballistic_flight():
    """
    Simulates a carton flying through the air with high horizontal velocity and vertical arc.
    Asserts product_thrown violation is detected.
    """
    rule = ThrowRule()
    config = BehaviourConfig()

    points = [
        make_track_point(0, 0.000, 5, "carton", (100, 200, 160, 260)),
        make_track_point(1, 0.033, 5, "carton", (135, 185, 195, 245)),
        make_track_point(2, 0.067, 5, "carton", (170, 175, 230, 235)),  # High horizontal speed
        make_track_point(3, 0.100, 5, "carton", (205, 180, 265, 240)),  # Downward arc
        make_track_point(4, 0.133, 5, "carton", (245, 200, 305, 260)),
        make_track_point(5, 0.167, 5, "carton", (290, 230, 350, 290)),
    ]
    carton = make_tracked_object(5, "carton", points)
    tracked_objects = {5: carton}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[5],
        current_frame_idx=5,
        timestamp=0.167,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 1
    assert events[0].rule_name == "product_thrown"
    assert events[0].track_id == 5
    assert events[0].metrics["horizontal_velocity_px_s"] >= config.throw_min_horizontal_velocity_px_s


def test_throw_rule_suppressed_when_near_worker():
    """
    Simulates high horizontal speed but worker is carrying the package.
    Asserts suppression works.
    """
    rule = ThrowRule()
    config = BehaviourConfig()

    carton_pts = [
        make_track_point(0, 0.000, 5, "carton", (100, 200, 160, 260)),
        make_track_point(1, 0.033, 5, "carton", (135, 185, 195, 245)),
        make_track_point(2, 0.067, 5, "carton", (170, 175, 230, 235)),
        make_track_point(3, 0.100, 5, "carton", (205, 180, 265, 240)),
        make_track_point(4, 0.133, 5, "carton", (245, 200, 305, 260)),
        make_track_point(5, 0.167, 5, "carton", (290, 230, 350, 290)),
    ]
    carton = make_tracked_object(5, "carton", carton_pts)

    # Worker walking right alongside
    worker_pts = [
        make_track_point(5, 0.167, 2, "person", (270, 150, 360, 340)),
    ]
    worker = make_tracked_object(2, "person", worker_pts)

    tracked_objects = {5: carton, 2: worker}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[5, 2],
        current_frame_idx=5,
        timestamp=0.167,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 0


# ───────────────────────── 3. Product Dragged Tests ─────────────────────────

def test_drag_rule_triggers_on_floor_sliding():
    """
    Simulates a carton continuously sliding horizontally on floor alongside a walking worker.
    Asserts product_dragged violation is detected.
    """
    rule = DragRule()
    config = BehaviourConfig()

    carton_pts = []
    worker_pts = []

    for i in range(12):
        t = i * 0.033
        # Carton sliding horizontally along floor at y=450
        cx = 100 + i * 15  # speed = 15 / 0.033 = 454 px/s
        carton_pts.append(make_track_point(i, t, 3, "carton", (cx - 30, 410, cx + 30, 460)))

        # Worker walking adjacent at y=300..500
        wx = 120 + i * 15
        worker_pts.append(make_track_point(i, t, 2, "person", (wx - 30, 300, wx + 30, 500)))

    carton = make_tracked_object(3, "carton", carton_pts)
    worker = make_tracked_object(2, "person", worker_pts)
    tracked_objects = {3: carton, 2: worker}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[3, 2],
        current_frame_idx=11,
        timestamp=11 * 0.033,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 1
    assert events[0].rule_name == "product_dragged"
    assert events[0].track_id == 3
    assert events[0].secondary_track_id == 2


# ───────────────────────── 4. Product Pushed Tests ──────────────────────────

def test_push_rule_triggers_on_impulse_spike():
    """
    Simulates stationary carton receiving sudden impulse kick after worker foot proximity.
    Asserts product_pushed violation is detected.
    """
    rule = PushRule()
    config = BehaviourConfig()

    # Carton stationary for frames 0..2, then kicked at frame 3
    carton_pts = [
        make_track_point(0, 0.000, 4, "carton", (200, 400, 260, 460)),
        make_track_point(1, 0.033, 4, "carton", (200, 400, 260, 460)),
        make_track_point(2, 0.067, 4, "carton", (201, 400, 261, 460)),
        make_track_point(3, 0.100, 4, "carton", (230, 400, 290, 460)),  # Impulse: dx=29 px / 0.033s = 878 px/s
    ]
    carton = make_tracked_object(4, "carton", carton_pts)

    # Worker standing right behind carton, foot at bottom (x=195, y=450)
    worker_pts = [
        make_track_point(3, 0.100, 2, "person", (160, 320, 220, 460)),
    ]
    worker = make_tracked_object(2, "person", worker_pts)

    tracked_objects = {4: carton, 2: worker}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[4, 2],
        current_frame_idx=3,
        timestamp=0.100,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 1
    assert events[0].rule_name == "product_pushed"
    assert events[0].track_id == 4
    assert events[0].secondary_track_id == 2


# ───────────────────────── 5. Product Rolled Tests ──────────────────────────

def test_roll_rule_triggers_on_aspect_ratio_oscillation():
    """
    Simulates a translating box tumbling over its edges with alternating aspect ratios.
    Asserts product_rolled violation is detected.
    """
    rule = RollRule()
    config = BehaviourConfig()

    points = []
    # 16 frames moving horizontally, aspect ratio alternating wide (1.5) and tall (0.7)
    for i in range(16):
        t = i * 0.033
        x = 100 + i * 8  # speed = 8 / 0.033 = 242 px/s
        if (i // 4) % 2 == 0:
            # Wide: w=120, h=80 -> AR = 1.5
            bbox = (x, 400, x + 120, 480)
        else:
            # Tall: w=70, h=100 -> AR = 0.7
            bbox = (x, 380, x + 70, 480)
        points.append(make_track_point(i, t, 6, "carton", bbox))

    carton = make_tracked_object(6, "carton", points)
    tracked_objects = {6: carton}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[6],
        current_frame_idx=15,
        timestamp=15 * 0.033,
        video_fps=30.0,
        config=config,
    )

    assert len(events) == 1
    assert events[0].rule_name == "product_rolled"
    assert events[0].track_id == 6
    assert events[0].metrics["aspect_ratio_flips"] >= config.roll_min_aspect_ratio_flips


# ──────────────────────── 6. Stacking & Overhang Tests ──────────────────────

def test_unstable_stacking_detected():
    """
    Simulates two vertically stacked cartons with upper carton center offset by >35%.
    Asserts unstable_stacking violation is detected.
    """
    rule = StackingRule()
    config = BehaviourConfig()

    # Lower carton B at [200, 400, 400, 500] (width=200, center=300)
    # Upper carton A at [290, 300, 490, 400] (width=200, center=390)
    # Offset = |390 - 300| / 200 = 45% > 35%
    b_pts = [make_track_point(i, i * 0.033, 10, "carton", (200, 400, 400, 500)) for i in range(4)]
    a_pts = [make_track_point(i, i * 0.033, 11, "carton", (290, 300, 490, 400)) for i in range(4)]

    carton_b = make_tracked_object(10, "carton", b_pts)
    carton_a = make_tracked_object(11, "carton", a_pts)

    tracked_objects = {10: carton_b, 11: carton_a}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[10, 11],
        current_frame_idx=3,
        timestamp=3 * 0.033,
        video_fps=30.0,
        config=config,
    )

    unstable_events = [e for e in events if e.rule_name == "unstable_stacking"]
    assert len(unstable_events) == 1
    assert unstable_events[0].track_id == 11
    assert unstable_events[0].secondary_track_id == 10
    assert unstable_events[0].metrics["offset_ratio"] >= config.stacking_max_offset_ratio


def test_pallet_overhang_detected():
    """
    Simulates a carton resting on a pallet protruding beyond the left boundary by >15%.
    Asserts pallet_overhang violation is detected.
    """
    rule = StackingRule()
    config = BehaviourConfig()

    # Pallet P at [200, 500, 600, 580]
    # Carton A at [140, 420, 300, 500] (width=160, overhang_left = 200 - 140 = 60 px -> 60/160 = 37.5% > 15%)
    p_pts = [make_track_point(0, 0.0, 20, "pallet", (200, 500, 600, 580))]
    c_pts = [make_track_point(0, 0.0, 21, "carton", (140, 420, 300, 500))]

    pallet = make_tracked_object(20, "pallet", p_pts)
    carton = make_tracked_object(21, "carton", c_pts)

    tracked_objects = {20: pallet, 21: carton}

    events = rule.evaluate(
        tracked_objects=tracked_objects,
        active_track_ids=[20, 21],
        current_frame_idx=0,
        timestamp=0.0,
        video_fps=30.0,
        config=config,
    )

    overhang_events = [e for e in events if e.rule_name == "pallet_overhang"]
    assert len(overhang_events) == 1
    assert overhang_events[0].track_id == 21
    assert overhang_events[0].secondary_track_id == 20
    assert overhang_events[0].metrics["overhang_ratio"] >= config.pallet_overhang_max_ratio


# ─────────────────────── 7. Engine Deduplication Tests ──────────────────────

def test_engine_event_cooldown_deduplication():
    """
    Verifies that recurring violation conditions within the cooldown window (60 frames)
    are suppressed and produce exactly 1 confirmed event.
    """
    engine = BehaviourEngine(config=BehaviourConfig(cooldown_frames=60))

    # Mock tracker
    tracker = ObjectTracker()

    # Create an unstable stack fixture
    b_pts = [make_track_point(0, 0.0, 10, "carton", (200, 400, 400, 500)) for _ in range(4)]
    a_pts = [make_track_point(0, 0.0, 11, "carton", (290, 300, 490, 400)) for _ in range(4)]
    tracker._tracked_objects[10] = make_tracked_object(10, "carton", b_pts)
    tracker._tracked_objects[11] = make_tracked_object(11, "carton", a_pts)

    # Frame 0: First time triggering -> should produce 1 event
    events_f0 = engine.evaluate_frame(tracker, current_frame_idx=0, timestamp=0.0)
    assert len(events_f0) == 1
    assert events_f0[0].rule_name == "unstable_stacking"

    # Frame 10: Still within 60-frame cooldown -> should be suppressed (0 events)
    for obj in tracker._tracked_objects.values():
        obj.trajectory[-1].frame_idx = 10
    events_f10 = engine.evaluate_frame(tracker, current_frame_idx=10, timestamp=0.33)
    assert len(events_f10) == 0

    # Frame 65: Cooldown expired (>60 frames elapsed) -> can fire again
    for obj in tracker._tracked_objects.values():
        obj.trajectory[-1].frame_idx = 65
    events_f65 = engine.evaluate_frame(tracker, current_frame_idx=65, timestamp=2.16)
    assert len(events_f65) == 1
    assert events_f65[0].rule_name == "unstable_stacking"

    # Total recorded in history
    assert len(engine.all_events) == 2


def test_behaviour_engine_full_integration():
    """
    Verifies that BehaviourEngine coordinates all rules seamlessly
    and returns empty list when no violations occur.
    """
    engine = BehaviourEngine()
    tracker = ObjectTracker()

    # Single stationary worker
    worker_pts = [make_track_point(0, 0.0, 1, "person", (100, 100, 200, 300))]
    tracker._tracked_objects[1] = make_tracked_object(1, "person", worker_pts)

    events = engine.evaluate_frame(tracker, current_frame_idx=0, timestamp=0.0)
    assert events == []
    assert len(engine.rules) >= 6
