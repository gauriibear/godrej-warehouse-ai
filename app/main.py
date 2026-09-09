"""
Main entry point for Godrej Warehouse AI Video Intelligence Pipeline.
Stage 3: Video Ingestion → YOLO Object Detection → ByteTrack Object Tracking CLI Runner.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.video.reader import VideoReader
from app.detection.detector import ObjectDetector
from app.tracking.tracker import ObjectTracker


def main():
    parser = argparse.ArgumentParser(
        description="Godrej Warehouse AI - Detection + Tracking Pipeline (Stage 3)"
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default="videos/warehouse_cctv_sample.mp4",
        help="Path to the input video file (e.g., .mp4)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolo11n.pt",
        help="YOLO model architecture / weights (e.g., yolo11n.pt, yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="Confidence threshold for object detection (default: 0.25)",
    )
    parser.add_argument(
        "--max-frames",
        "-n",
        type=int,
        default=60,
        help="Number of frames to process (default: 60)",
    )

    args = parser.parse_args()
    video_path = Path(args.video)

    print("=" * 70)
    print(" GODREJ WAREHOUSE AI - STAGE 3: DETECTION + TRACKING PIPELINE")
    print("=" * 70)

    if not video_path.exists():
        print(f"\n[ERROR] Video file not found at: {video_path}")
        print("Please place a video in the `videos/` folder or specify `--video <path>`.")
        sys.exit(1)

    # ── Initialise Detector ────────────────────────────────────────────
    print(f"\n[INFO] Loading YOLO Detector (Model: {args.model}, Conf: {args.conf:.2f})...")
    try:
        detector = ObjectDetector(
            model_name_or_path=args.model,
            confidence_threshold=args.conf,
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
            tracker._byte_tracker = tracker._byte_tracker.__class__(
                track_activation_threshold=args.conf,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=int(meta.fps) if meta.fps > 0 else 30,
            )

            print(f"\n[INFO] Processing first {args.max_frames} frames with Detection + Tracking...")
            total_detections = 0
            total_latency = 0.0

            for idx, timestamp, frame in reader.frames(max_frames=args.max_frames):
                # Stage 2: Detect
                det_result = detector.detect(
                    frame, frame_idx=idx, timestamp_seconds=timestamp, annotate=False
                )
                total_detections += det_result.count
                total_latency += det_result.inference_time_ms

                # Stage 3: Track
                trk_result = tracker.update(det_result, frame, annotate=False)

                if idx % 10 == 0 or idx == args.max_frames - 1:
                    active_labels = []
                    for tid in trk_result.active_track_ids:
                        obj = tracker.tracked_objects.get(tid)
                        if obj:
                            active_labels.append(obj.display_label)
                    labels_str = ", ".join(active_labels) if active_labels else "None"
                    print(
                        f"  [Frame {idx:04d} | {timestamp:.2f}s] "
                        f"Det: {det_result.count} | Active Tracks: {trk_result.active_count} "
                        f"| {labels_str}"
                    )

            avg_latency = total_latency / args.max_frames if args.max_frames > 0 else 0
            avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

            print("\n--- Detection + Tracking Performance Summary ---")
            print(f"  - Frames Processed:      {args.max_frames}")
            print(f"  - Total Detections:      {total_detections}")
            print(f"  - Unique Objects Tracked: {tracker.total_unique_tracks}")
            print(f"  - Average Latency:       {avg_latency:.2f} ms/frame ({avg_fps:.1f} FPS)")

            # Show trajectory summaries
            print("\n--- Tracked Object Trajectories ---")
            for tid, obj in tracker.tracked_objects.items():
                traj_len = obj.frame_count
                first_frame = obj.trajectory[0].frame_idx if obj.trajectory else "?"
                last_frame = obj.last_seen_frame
                print(
                    f"  {obj.display_label:30s} | "
                    f"Frames: {traj_len:4d} | "
                    f"Span: [{first_frame} → {last_frame}]"
                )

            print(f"\n[SUCCESS] Stage 3 Detection + Tracking pipeline verified.")
            print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failure: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
