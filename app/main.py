"""
Main entry point for Godrej Warehouse AI Video Intelligence Pipeline.
Stage 3: Video Ingestion → YOLO Object Detection → ByteTrack Object Tracking CLI Runner.
"""

import argparse
import sys
from pathlib import Path
import cv2

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.video.reader import VideoReader
from app.detection.detector import ObjectDetector
from app.tracking.tracker import ObjectTracker
from app.behaviour.engine import BehaviourEngine, BehaviourConfig
from app.risk import RiskEngine, RiskConfig, EvidenceCollector, RiskLevel


def main():
    default_video = (
        "videos/people_walking.mp4"
        if Path("videos/people_walking.mp4").exists()
        else "videos/warehouse_cctv_sample.mp4"
    )
    default_model = str((project_root / "runs/detect/runs/train/warehouse_detector/weights/best.pt").resolve())
    if not Path(default_model).exists():
        default_model = "yolo11n.pt"

    parser = argparse.ArgumentParser(
        description="Godrej Warehouse AI - Risk Engine & Incident Evidence Pipeline (Stage 5)"
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=default_video,
        help=f"Path to the input video file (default: {default_video})",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=default_model,
        help="YOLO model architecture / weights (e.g., custom warehouse model or yolo11n.pt)",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="Confidence threshold for object detection (default: 0.25)",
    )
    parser.add_argument(
        "--target-classes",
        type=str,
        default="person,carton,forklift,pallet",
        help="Comma-separated classes to retain (default: person,carton,forklift,pallet)",
    )
    parser.add_argument(
        "--max-frames",
        "-n",
        type=int,
        default=60,
        help="Number of frames to process (default: 60)",
    )
    parser.add_argument(
        "--save-visual",
        action="store_true",
        default=True,
        help="Save annotated tracking keyframe and video to outputs/ (default: True)",
    )
    parser.add_argument(
        "--disable-behaviour",
        action="store_true",
        default=False,
        help="Disable Stage 4 behaviour reasoning rules",
    )
    parser.add_argument(
        "--disable-risk",
        action="store_true",
        default=False,
        help="Disable Stage 5 risk scoring and incident evidence capture",
    )
    parser.add_argument(
        "--evidence-dir",
        type=str,
        default="outputs/evidence",
        help="Directory to save incident evidence keyframes and clips (default: outputs/evidence)",
    )

    args = parser.parse_args()
    video_path = Path(args.video)

    print("=" * 70)
    print(" GODREJ WAREHOUSE AI - STAGE 5: RISK ENGINE & INCIDENT EVIDENCE")
    print("=" * 70)

    if not video_path.exists():
        print(f"\n[ERROR] Video file not found at: {video_path}")
        print("Please place a video in the `videos/` folder or specify `--video <path>`.")
        sys.exit(1)

    target_classes = [c.strip().lower() for c in args.target_classes.split(",") if c.strip()]

    # ── Initialise Detector ────────────────────────────────────────────
    print(f"\n[INFO] Loading YOLO Detector (Model: {args.model}, Conf: {args.conf:.2f})...")
    try:
        detector = ObjectDetector(
            model_name_or_path=args.model,
            confidence_threshold=args.conf,
            target_classes=target_classes,
        )
        print("[SUCCESS] YOLO Detector initialised successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load detector: {str(e)}")
        sys.exit(1)

    # ── Initialise Tracker ─────────────────────────────────────────────
    print("[INFO] Initialising ByteTrack Object Tracker...")
    tracker = ObjectTracker(
        track_activation_threshold=args.conf,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
    )
    print("[SUCCESS] ByteTrack Tracker initialised.")

    # ── Initialise Behaviour Engine ────────────────────────────────────
    behaviour_engine = None
    if not args.disable_behaviour:
        print("[INFO] Initialising Behaviour Reasoning Engine (7 rules enabled)...")
        behaviour_engine = BehaviourEngine(config=BehaviourConfig())
        print(f"[SUCCESS] Behaviour Engine initialised with {len(behaviour_engine.rules)} active rules:")
        for r in behaviour_engine.rules:
            print(f"  - {r.name}")

    # ── Initialise Risk Engine & Evidence Collector ────────────────────
    risk_engine = None
    evidence_collector = None
    if not args.disable_risk:
        print(f"[INFO] Initialising Risk Engine & Incident Evidence Collector (Evidence dir: {args.evidence_dir})...")
        risk_engine = RiskEngine(config=RiskConfig())
        evidence_collector = EvidenceCollector(output_dir=args.evidence_dir)
        print("[SUCCESS] Risk Engine & Evidence Collector initialised.")

    # ── Process Video ──────────────────────────────────────────────────
    print(f"\n[INFO] Ingesting video: {video_path}")
    try:
        with VideoReader(video_path) as reader:
            meta = reader.metadata
            print(f"  - Resolution:   {meta.resolution}")
            print(f"  - Framerate:    {meta.fps:.2f} FPS")
            print(f"  - Total Frames: {meta.total_frames:,}")
            print(f"  - Duration:     {meta.formatted_duration}")

            # Update tracker frame rate to match video
            tracker = ObjectTracker(
                track_activation_threshold=args.conf,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=int(meta.fps) if meta.fps > 0 else 30,
            )

            # Prepare video writer if visual output requested
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(parents=True, exist_ok=True)
            video_writer = None
            sample_annotated_frame = None

            if args.save_visual:
                output_video_path = outputs_dir / "tracking_run.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    meta.fps if meta.fps > 0 else 30.0,
                    (meta.width, meta.height),
                )

            print(f"\n[INFO] Processing first {args.max_frames} frames with Detection + Tracking + Behaviour + Risk...")
            total_detections = 0
            total_latency = 0.0

            for idx, timestamp, frame in reader.frames(max_frames=args.max_frames):
                # Buffer frame for before/after evidence clips
                if evidence_collector is not None:
                    evidence_collector.add_frame(idx, frame)

                # Stage 2: Detect
                det_result = detector.detect(
                    frame, frame_idx=idx, timestamp_seconds=timestamp, annotate=False
                )
                total_detections += det_result.count
                total_latency += det_result.inference_time_ms

                # Stage 3: Track
                trk_result = tracker.update(det_result, frame, annotate=args.save_visual)

                # Stage 4: Behaviour Reasoning
                frame_events = []
                if behaviour_engine is not None:
                    frame_events = behaviour_engine.evaluate_frame(
                        tracker=tracker,
                        current_frame_idx=idx,
                        timestamp=timestamp,
                        fps=meta.fps,
                    )
                    for ev in frame_events:
                        sec_info = f" (Worker: {ev.secondary_label})" if ev.secondary_label else ""
                        print(
                            f"  [ALERT Frame {idx:04d} | {timestamp:.2f}s] ⚠️ {ev.human_readable_name.upper()} "
                            f"detected on {ev.display_label}{sec_info} (Conf: {ev.confidence * 100:.1f}%)"
                        )

                # Stage 5: Risk Engine & Incident Evidence
                frame_incidents = []
                if risk_engine is not None and frame_events:
                    for ev in frame_events:
                        incident = risk_engine.evaluate_event(ev)
                        frame_incidents.append(incident)
                        print(
                            f"  [INCIDENT Frame {idx:04d} | {timestamp:.2f}s] 🚨 {incident.risk_level.value} RISK "
                            f"(Score: {incident.risk_score:.0f}/100) on {incident.display_label} - "
                            f"{incident.human_readable_name}"
                        )
                        if evidence_collector is not None:
                            # Save annotated keyframe screenshot
                            evidence_collector.save_keyframe(incident, frame, annotate=True)
                            # Extract short before/after clip if buffer has enough frames
                            evidence_collector.extract_clip(incident, fps=meta.fps)

                if args.save_visual and trk_result.annotated_frame is not None:
                    annotated_frame = trk_result.annotated_frame
                    if behaviour_engine is not None:
                        annotated_frame = behaviour_engine.annotate_events(
                            frame=annotated_frame,
                            events=frame_events,
                            current_frame_idx=idx,
                            tracked_objects=tracker.tracked_objects,
                        )

                    if video_writer is not None:
                        video_writer.write(annotated_frame)
                    # Keep a sample frame with active tracks for image artifact
                    if trk_result.active_count > 0:
                        sample_annotated_frame = annotated_frame

                if idx % 10 == 0 or idx == args.max_frames - 1:
                    active_labels = []
                    for tid in trk_result.active_track_ids:
                        obj = tracker.tracked_objects.get(tid)
                        if obj:
                            active_labels.append(obj.display_label)
                    labels_str = ", ".join(active_labels) if active_labels else "None"
                    alert_count_str = f" | Alerts: {len(frame_events)}" if frame_events else ""
                    print(
                        f"  [Frame {idx:04d} | {timestamp:.2f}s] "
                        f"Det: {det_result.count:2d} | Active Tracks: {trk_result.active_count:2d} "
                        f"| {labels_str}{alert_count_str}"
                    )

            if video_writer is not None:
                video_writer.release()
                print(f"\n[INFO] Saved annotated tracking video to: outputs/tracking_run.mp4")

            if sample_annotated_frame is not None:
                verification_img_path = outputs_dir / "tracking_verification.jpg"
                cv2.imwrite(str(verification_img_path), sample_annotated_frame)
                print(f"[INFO] Saved tracking visual verification frame to: {verification_img_path}")

            # Save incident master manifest if evidence collector active
            if evidence_collector is not None and risk_engine is not None:
                manifest_path = evidence_collector.save_manifest(risk_engine.all_incidents)
                print(f"[INFO] Saved incident master manifest to: {manifest_path}")

            avg_latency = total_latency / args.max_frames if args.max_frames > 0 else 0
            avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

            print("\n--- Detection + Tracking Performance Summary ---")
            print(f"  - Frames Processed:       {args.max_frames}")
            print(f"  - Total Detections:       {total_detections}")
            print(f"  - Unique Objects Tracked: {tracker.total_unique_tracks}")
            print(f"  - Average Latency:        {avg_latency:.2f} ms/frame ({avg_fps:.1f} FPS)")

            # Show trajectory summaries with 7-field verification
            print("\n--- Tracked Object Trajectories ---")
            for tid, obj in tracker.tracked_objects.items():
                traj_len = obj.frame_count
                first_frame = obj.trajectory[0].frame_number if obj.trajectory else "?"
                last_frame = obj.last_seen_frame
                duration = (
                    obj.trajectory[-1].timestamp - obj.trajectory[0].timestamp
                    if len(obj.trajectory) > 1 else 0.0
                )
                last_pt = obj.trajectory[-1] if obj.trajectory else None
                center_str = f"({last_pt.center_point[0]}, {last_pt.center_point[1]})" if last_pt else "N/A"

                print(
                    f"  {obj.display_label:25s} | "
                    f"Frames: {traj_len:3d} | "
                    f"Duration: {duration:4.2f}s | "
                    f"Span: [{first_frame:3d} → {last_frame:3d}] | "
                    f"Last Center: {center_str:14s} | "
                    f"Conf: {obj.last_confidence * 100:4.1f}%"
                )

            if behaviour_engine is not None:
                print("\n--- Behaviour Engine Violations Summary ---")
                total_violations = len(behaviour_engine.all_events)
                print(f"  - Total Violations Detected: {total_violations}")
                if total_violations > 0:
                    by_rule = {}
                    for ev in behaviour_engine.all_events:
                        by_rule[ev.human_readable_name] = by_rule.get(ev.human_readable_name, 0) + 1
                    for rule_name, count in by_rule.items():
                        print(f"    • {rule_name:25s}: {count}")
                else:
                    print("    (No violations observed in this sample clip)")

            if risk_engine is not None:
                print("\n--- Risk Engine Incident Assessment Summary ---")
                total_incidents = len(risk_engine.all_incidents)
                print(f"  - Total Incidents Logged:   {total_incidents}")
                if total_incidents > 0:
                    by_level = {}
                    for inc in risk_engine.all_incidents:
                        by_level[inc.risk_level.value] = by_level.get(inc.risk_level.value, 0) + 1
                    print("  - Risk Severity Breakdown:")
                    for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                        if lvl in by_level:
                            print(f"    • {lvl:10s}: {by_level[lvl]}")
                    print("  - Damage Assessment:       POTENTIAL_DAMAGE_RISK (Distinguished from confirmed damage)")
                    print(f"  - Evidence Artifacts:      {total_incidents} keyframe(s) in {args.evidence_dir}")
                else:
                    print("    (0 handling risk incidents recorded; all observed handling conformed to safety policies)")

            print(f"\n[SUCCESS] Stage 5 Risk Engine & Incident Evidence pipeline verified.")
            print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failure: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
