"""
Stage 5 Evidence-Generation Utility for Godrej Warehouse AI.
Generates deterministic synthetic material handling behaviour events, evaluates them
through the Risk Engine, and captures visual keyframe snapshots, video clips, and the
master JSON manifest using EvidenceCollector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.behaviour.engine import BehaviourEvent
from app.risk import (
    RiskEngine,
    RiskConfig,
    EvidenceCollector,
    RiskLevel,
    DamageAssessmentStatus,
    IncidentRecord,
)


def create_synthetic_cctv_frame(
    frame_idx: int,
    total_frames: int = 120,
    width: int = 1280,
    height: int = 720,
    box_positions: dict = None,
) -> np.ndarray:
    """
    Renders a synthetic warehouse camera frame with floor, shelving,
    camera timestamp overlay, and simulated carton boxes.
    """
    frame = np.full((height, width, 3), 42, dtype=np.uint8)

    # Floor (gray-brown concrete)
    floor_y = int(height * 0.65)
    cv2.rectangle(frame, (0, floor_y), (width, height), (55, 60, 65), -1)
    cv2.line(frame, (0, floor_y), (width, floor_y), (85, 90, 95), 2)

    # Storage rack structure (left)
    cv2.rectangle(frame, (40, int(height * 0.15)), (220, floor_y), (68, 72, 78), -1)
    for y_beam in [int(height * 0.32), int(height * 0.48)]:
        cv2.line(frame, (40, y_beam), (220, y_beam), (100, 105, 110), 3)

    # CCTV camera OSD (On-Screen Display)
    time_sec = frame_idx / 25.0
    cv2.putText(
        frame,
        f"CAM 04 - DISPATCH BAY 02 | 2026-09-09 {int(time_sec // 60):02d}:{int(time_sec % 60):02d}.{int((time_sec % 1) * 100):02d}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (190, 190, 190),
        2,
        cv2.LINE_AA,
    )

    # Draw simulated carton boxes
    if box_positions:
        for tid, box in box_positions.items():
            x1, y1, x2, y2 = box
            # Carton cardboard color
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 150, 60), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 115, 45), 2)
            cv2.putText(
                frame,
                f"T#{tid}",
                (x1 + 6, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return frame


def generate_synthetic_evidence(
    output_dir: str = "outputs/evidence",
    generate_clips: bool = True,
    verbose: bool = True,
) -> List[IncidentRecord]:
    """
    Generates deterministic behaviour events representing:
      1. Product Dropped -> CRITICAL risk (extreme freefall velocity)
      2. Product Dropped -> HIGH risk (standard freefall drop)
      3. Product Dragged -> MEDIUM risk (floor sliding)
      4. Unstable Stacking -> HIGH risk (misaligned center of mass)
      5. Unstable Stacking -> CRITICAL risk (severe offset >= 50%)
      6. Repeated Violations -> Escalated risk on identical entity
    Evaluates through existing RiskEngine and EvidenceCollector.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    risk_engine = RiskEngine(config=RiskConfig())
    evidence_collector = EvidenceCollector(
        output_dir=str(out_path),
        pre_event_frames=15,
        post_event_frames=15,
    )

    total_sequence_frames = 120
    fps = 25.0

    # Build sequence of synthetic frames in rolling buffer
    frames: List[Tuple[int, np.ndarray]] = []
    for f in range(total_sequence_frames):
        # Determine dynamic box positions for realism
        boxes = {
            1: (580, min(440, 180 + f * 9), 690, min(550, 290 + f * 9)),
            2: (250, min(430, 220 + f * 6), 360, min(540, 330 + f * 6)),
            3: (min(920, 650 + f * 3), 460, min(1030, 760 + f * 3), 560),
            4: (400, 260, 510, 370),
            5: (420, 240, 530, 350),
            7: (min(750, 480 + f * 4), 450, min(860, 590 + f * 4), 550),
        }
        frame_img = create_synthetic_cctv_frame(f, total_frames=total_sequence_frames, box_positions=boxes)
        evidence_collector.add_frame(f, frame_img)
        frames.append((f, frame_img))

    # ── Define Deterministic Scenarios ─────────────────────────────────────
    scenarios: List[BehaviourEvent] = [
        # 1. Product Dropped -> CRITICAL (v_y = 540 >= 500, drop_height = 160 >= 120)
        BehaviourEvent(
            event_id="EVT-DROP-1-F32",
            rule_name="product_dropped",
            track_id=1,
            display_label="Carton #1",
            class_name="carton",
            frame_idx=32,
            timestamp_seconds=round(32 / fps, 2),
            confidence=0.94,
            bbox=(580, 430, 690, 550),
            metrics={"vertical_velocity_px_s": 540.0, "drop_height_px": 160.0},
        ),
        # 2. Product Dropped -> HIGH (v_y = 380, drop_height = 75)
        BehaviourEvent(
            event_id="EVT-DROP-2-F45",
            rule_name="product_dropped",
            track_id=2,
            display_label="Carton #2",
            class_name="carton",
            frame_idx=45,
            timestamp_seconds=round(45 / fps, 2),
            confidence=0.80,
            bbox=(250, 420, 360, 540),
            metrics={"vertical_velocity_px_s": 380.0, "drop_height_px": 75.0},
        ),
        # 3. Product Dragged -> MEDIUM (v_x = 85 px/s, duration = 14 frames)
        BehaviourEvent(
            event_id="EVT-DRAG-3-F60",
            rule_name="product_dragged",
            track_id=3,
            display_label="Carton #3",
            secondary_track_id=10,
            secondary_label="Person #10",
            class_name="carton",
            frame_idx=60,
            timestamp_seconds=round(60 / fps, 2),
            confidence=0.82,
            bbox=(800, 460, 910, 560),
            metrics={"drag_duration_frames": 14, "horizontal_velocity_px_s": 85.0, "worker_dist_px": 110.0},
        ),
        # 4. Unstable Stacking -> HIGH (offset_ratio = 0.42)
        BehaviourEvent(
            event_id="EVT-STACK-4-F75",
            rule_name="unstable_stacking",
            track_id=4,
            display_label="Carton #4",
            class_name="carton",
            frame_idx=75,
            timestamp_seconds=round(75 / fps, 2),
            confidence=0.88,
            bbox=(400, 260, 510, 370),
            metrics={"offset_ratio": 0.42, "offset_px": 46.0, "stationary_frames": 12},
        ),
        # 5. Unstable Stacking -> CRITICAL (severe offset_ratio = 0.56 >= 0.50)
        BehaviourEvent(
            event_id="EVT-STACK-5-F88",
            rule_name="unstable_stacking",
            track_id=5,
            display_label="Carton #5",
            class_name="carton",
            frame_idx=88,
            timestamp_seconds=round(88 / fps, 2),
            confidence=0.91,
            bbox=(420, 240, 530, 350),
            metrics={"offset_ratio": 0.56, "offset_px": 62.0, "stationary_frames": 15},
        ),
        # 6. Repeated Violations on Same Entity (Carton #7):
        # 6a: First infraction (Dragged -> MEDIUM)
        BehaviourEvent(
            event_id="EVT-DRAG-7-F18",
            rule_name="product_dragged",
            track_id=7,
            display_label="Carton #7",
            class_name="carton",
            frame_idx=18,
            timestamp_seconds=round(18 / fps, 2),
            confidence=0.80,
            bbox=(480, 450, 590, 550),
            metrics={"drag_duration_frames": 12, "horizontal_velocity_px_s": 90.0},
        ),
        # 6b: Second infraction on Carton #7 (Pushed -> HIGH via repeat penalty)
        BehaviourEvent(
            event_id="EVT-PUSH-7-F52",
            rule_name="product_pushed",
            track_id=7,
            display_label="Carton #7",
            secondary_track_id=12,
            secondary_label="Person #12",
            class_name="carton",
            frame_idx=52,
            timestamp_seconds=round(52 / fps, 2),
            confidence=0.84,
            bbox=(540, 450, 650, 550),
            metrics={"impulse_spike_px_s": 290.0, "foot_distance_px": 55.0},
        ),
        # 6c: Third infraction on Carton #7 (Dropped -> CRITICAL via chronic repeat penalty)
        BehaviourEvent(
            event_id="EVT-DROP-7-F102",
            rule_name="product_dropped",
            track_id=7,
            display_label="Carton #7",
            class_name="carton",
            frame_idx=102,
            timestamp_seconds=round(102 / fps, 2),
            confidence=0.86,
            bbox=(640, 440, 750, 550),
            metrics={"vertical_velocity_px_s": 420.0, "drop_height_px": 95.0},
        ),
    ]

    generated_incidents: List[IncidentRecord] = []

    if verbose:
        print("=" * 75)
        print(" GODREJ WAREHOUSE AI - STAGE 5 SYNTHETIC EVIDENCE GENERATOR")
        print("=" * 75)
        print(f"[INFO] Target output directory: {out_path}")
        print(f"[INFO] Generating {len(scenarios)} deterministic behaviour incidents...")

    for event in scenarios:
        # Evaluate through RiskEngine
        incident = risk_engine.evaluate_event(event)
        generated_incidents.append(incident)

        # Retrieve matching frame
        target_frame = frames[min(event.frame_idx, len(frames) - 1)][1]

        # 1. Save annotated keyframe snapshot
        kf_path = evidence_collector.save_keyframe(incident, target_frame, annotate=True)

        # 2. Extract before/after video clip if requested
        clip_path = None
        if generate_clips:
            clip_path = evidence_collector.extract_clip(incident, fps=fps)

        if verbose:
            sec_info = f" (Worker: {incident.secondary_label})" if incident.secondary_label else ""
            print(
                f"  [{incident.risk_level.value:8s} | Score {incident.risk_score:4.1f}] "
                f"{incident.incident_id:22s} | {incident.human_readable_name:18s} | {incident.display_label}{sec_info}"
            )
            print(f"    - Assessment: {incident.damage_assessment.value}")
            print(f"    - Keyframe:   {Path(kf_path).name}")
            if clip_path:
                print(f"    - Video Clip: {Path(clip_path).name}")

    # 3. Save master incidents manifest
    manifest_path = evidence_collector.save_manifest(generated_incidents)

    if verbose:
        print("-" * 75)
        print(f"[SUCCESS] Saved master incidents manifest to: {manifest_path}")
        print(f"[SUCCESS] Generated {len(generated_incidents)} incidents ({sum(1 for i in generated_incidents if i.risk_level == RiskLevel.CRITICAL)} Critical, "
              f"{sum(1 for i in generated_incidents if i.risk_level == RiskLevel.HIGH)} High, "
              f"{sum(1 for i in generated_incidents if i.risk_level == RiskLevel.MEDIUM)} Medium).")
        print("=" * 75)

    return generated_incidents


def main():
    parser = argparse.ArgumentParser(
        description="Godrej Warehouse AI - Stage 5 Evidence Generation Utility"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/evidence",
        help="Directory to save generated evidence images, clips, and incidents.json (default: outputs/evidence)",
    )
    parser.add_argument(
        "--no-clips",
        action="store_true",
        default=False,
        help="Skip before/after MP4 clip generation (generate keyframes and JSON only)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress console log output",
    )

    args = parser.parse_args()
    generate_synthetic_evidence(
        output_dir=args.output_dir,
        generate_clips=not args.no_clips,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
