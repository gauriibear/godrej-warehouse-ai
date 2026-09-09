"""
Streamlit Web Dashboard for Godrej Warehouse AI Video Intelligence.
Stage 3: Object Detection & ByteTrack Object Tracking Module.
"""

import sys
from pathlib import Path
import tempfile
import time

# Add project root to sys.path for direct streamlit execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import pandas as pd
import streamlit as st
from app.video.reader import VideoReader, VideoMetadata
from app.detection.detector import ObjectDetector, DetectionResult, DEFAULT_WAREHOUSE_CLASS_MAPPING
from app.tracking.tracker import ObjectTracker, TrackingResult


# Page Configuration
st.set_page_config(
    page_title="Godrej Warehouse AI - Detection & Tracking",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .stage-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .track-badge {
        display: inline-block;
        background-color: #EDE9FE;
        color: #6D28D9;
        padding: 3px 8px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="stage-badge">STAGE 3: DETECTION + TRACKING</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">📦 Godrej Warehouse AI Video Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-class object detection with ByteTrack persistent ID tracking and trajectory history.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(model_name: str, confidence_thresh: float, iou_thresh: float) -> ObjectDetector:
    """Caches detector instance in Streamlit memory."""
    return ObjectDetector(
        model_name_or_path=model_name,
        confidence_threshold=confidence_thresh,
        iou_threshold=iou_thresh,
    )


# Sidebar Controls
st.sidebar.header("📁 Video Ingestion Source")

source_type = st.sidebar.radio(
    "Choose video input method:",
    ["Upload Video File (.mp4)", "Select from Sample Videos"],
)

video_path_to_process = None

if source_type == "Upload Video File (.mp4)":
    uploaded_file = st.sidebar.file_uploader(
        "Upload warehouse CCTV recording",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload an MP4 or standard video recording for ingestion.",
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path_to_process = tfile.name

else:
    videos_dir = Path("videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    sample_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
    
    if sample_files:
        selected_sample = st.sidebar.selectbox(
            "Select existing sample video:",
            [str(p) for p in sample_files],
            format_func=lambda p: Path(p).name,
        )
        if selected_sample:
            video_path_to_process = selected_sample
    else:
        st.sidebar.info("No sample videos found in `videos/` folder. Please upload a file above.")

# Detector Settings in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Detection Parameters")

model_choice = st.sidebar.selectbox(
    "YOLO Model Architecture",
    ["yolo11n.pt", "yolov8n.pt", "yolov8s.pt"],
    index=0,
    help="Select the pretrained YOLO object detection backbone.",
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Detections with confidence below this threshold are filtered out.",
)

iou_threshold = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
    help="IoU threshold for Non-Max Suppression.",
)

# Tracking Settings in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("🔗 Tracking Parameters")

lost_track_buffer = st.sidebar.slider(
    "Lost Track Buffer (frames)",
    min_value=5,
    max_value=120,
    value=30,
    step=5,
    help="How many frames to keep a lost track before deletion.",
)

match_threshold = st.sidebar.slider(
    "Matching IoU Threshold",
    min_value=0.3,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="IoU threshold for matching detections to existing tracks.",
)

# Load detector
detector = load_detector(model_choice, conf_threshold, iou_threshold)
detector.set_confidence_threshold(conf_threshold)

# Main Area
if video_path_to_process:
    try:
        reader = VideoReader(video_path_to_process)
        meta = reader.metadata

        # Top KPI Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resolution", meta.resolution)
        with col2:
            st.metric("Framerate", f"{meta.fps:.2f} FPS")
        with col3:
            st.metric("Total Frames", f"{meta.total_frames:,}")
        with col4:
            st.metric("Duration", meta.formatted_duration, f"{meta.duration_seconds:.1f}s")

        st.markdown("---")

        # Tabs for Detection, Tracking, and Inspection
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Frame-by-Frame Detection Inspector",
            "🔗 Video Tracking Runner",
            "📊 Trajectory Analysis",
            "🎬 Source Video & Info",
        ])

        # TAB 1: Single Frame Scrubbing & Inspection (unchanged from Stage 2)
        with tab1:
            st.subheader("Interactive Frame Detection Analysis")
            if meta.total_frames > 0:
                frame_idx = st.slider(
                    "Select frame to detect objects:",
                    min_value=0,
                    max_value=max(0, meta.total_frames - 1),
                    value=0,
                    step=1,
                    key="frame_scrubber",
                )

                frame_bgr = reader.get_frame_at(frame_idx)
                if frame_bgr is not None:
                    timestamp = frame_idx / (meta.fps if meta.fps > 0 else 30.0)

                    det_result = detector.detect(
                        frame=frame_bgr,
                        frame_idx=frame_idx,
                        timestamp_seconds=timestamp,
                        annotate=True,
                    )

                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.markdown("**Original Frame**")
                        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

                    with col_img2:
                        st.markdown(f"**Annotated Detection (Latency: {det_result.inference_time_ms:.1f} ms)**")
                        st.image(cv2.cvtColor(det_result.annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                    st.markdown("#### Detected Objects")
                    if det_result.count > 0:
                        badge_cols = st.columns(len(det_result.counts_by_class))
                        for idx_c, (cls_name, count) in enumerate(det_result.counts_by_class.items()):
                            with badge_cols[idx_c]:
                                st.info(f"**{cls_name}**: {count}")

                        records = []
                        for i_d, det in enumerate(det_result.detections, 1):
                            records.append({
                                "#": i_d,
                                "Category": det.class_name,
                                "Raw Model Class": det.raw_class_name,
                                "Confidence": f"{det.confidence * 100:.1f}%",
                                "Bounding Box [x1, y1, x2, y2]": f"[{det.x1}, {det.y1}, {det.x2}, {det.y2}]",
                                "Centroid (x, y)": f"({det.center[0]}, {det.center[1]})",
                                "Foot Point": f"({det.foot_point[0]}, {det.foot_point[1]})",
                                "Area (px²)": f"{det.area:,}",
                            })
                        st.dataframe(pd.DataFrame(records), use_container_width=True)
                    else:
                        st.warning(f"No objects detected in Frame #{frame_idx} above confidence {conf_threshold:.2f}.")
                else:
                    st.error(f"Could not read frame #{frame_idx}.")

        # TAB 2: Tracking Runner
        with tab2:
            st.subheader("Process Video with Detection + Tracking")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                max_process_frames = st.number_input(
                    "Frames to process:",
                    min_value=5,
                    max_value=max(10, meta.total_frames),
                    value=min(120, meta.total_frames),
                    step=10,
                    key="tracking_max_frames",
                )
            with col_opt2:
                frame_skip = st.selectbox(
                    "Frame sampling step:",
                    options=[1, 2, 3, 5],
                    index=0,
                    format_func=lambda x: f"Every {x} frame(s)",
                    key="tracking_frame_skip",
                )

            if st.button("▶️ Run Detection + Tracking", type="primary"):
                # Initialise tracker fresh for each run
                tracker = ObjectTracker(
                    track_activation_threshold=conf_threshold,
                    lost_track_buffer=lost_track_buffer,
                    minimum_matching_threshold=match_threshold,
                    frame_rate=int(meta.fps) if meta.fps > 0 else 30,
                )

                # Reset reader position
                reader.seek_frame(0)

                progress_bar = st.progress(0)
                status_text = st.empty()
                video_placeholder = st.empty()

                # Metrics containers
                metric_cols = st.columns(4)
                metric_frames = metric_cols[0].empty()
                metric_detections = metric_cols[1].empty()
                metric_active = metric_cols[2].empty()
                metric_unique = metric_cols[3].empty()

                active_tracks_placeholder = st.empty()

                processed_count = 0
                total_detections = 0
                total_inference_time = 0.0

                for idx, ts, frame in reader.frames(max_frames=max_process_frames):
                    if idx % frame_skip != 0:
                        continue

                    # Detect
                    det_res = detector.detect(frame, frame_idx=idx, timestamp_seconds=ts, annotate=False)
                    total_detections += det_res.count
                    total_inference_time += det_res.inference_time_ms

                    # Track
                    trk_res = tracker.update(det_res, frame, annotate=True)
                    processed_count += 1

                    # Live video frame display
                    if trk_res.annotated_frame is not None:
                        video_placeholder.image(
                            cv2.cvtColor(trk_res.annotated_frame, cv2.COLOR_BGR2RGB),
                            caption=f"Frame #{idx} | Time: {ts:.2f}s | Active Tracks: {trk_res.active_count}",
                            use_container_width=True,
                        )

                    # Update live metrics
                    metric_frames.metric("Frames", processed_count)
                    metric_detections.metric("Total Detections", total_detections)
                    metric_active.metric("Active Tracks", trk_res.active_count)
                    metric_unique.metric("Unique Objects", tracker.total_unique_tracks)

                    # Show active track labels
                    if trk_res.active_track_ids:
                        labels = []
                        for tid in trk_res.active_track_ids:
                            obj = tracker.tracked_objects.get(tid)
                            if obj:
                                labels.append(f'<span class="track-badge">{obj.display_label}</span>')
                        active_tracks_placeholder.markdown(
                            "**Active:** " + " ".join(labels),
                            unsafe_allow_html=True,
                        )

                    progress = min(1.0, (idx + 1) / max_process_frames)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {idx + 1}/{max_process_frames}...")

                avg_latency = total_inference_time / processed_count if processed_count > 0 else 0
                avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0

                status_text.success(
                    f"✅ Processed {processed_count} frames! "
                    f"Avg Latency: {avg_latency:.1f} ms ({avg_fps:.1f} FPS) | "
                    f"Unique Objects Tracked: {tracker.total_unique_tracks}"
                )

                # Store tracker in session state for trajectory analysis
                st.session_state["tracker_result"] = tracker

        # TAB 3: Trajectory Analysis
        with tab3:
            st.subheader("Trajectory History & Track Analysis")

            if "tracker_result" in st.session_state:
                trk = st.session_state["tracker_result"]
                all_objects = trk.tracked_objects

                if all_objects:
                    st.info(f"**{len(all_objects)} unique objects tracked** across the video sequence.")

                    # Summary table
                    summary_records = []
                    for tid, obj in all_objects.items():
                        first_frame = obj.trajectory[0].frame_idx if obj.trajectory else "-"
                        last_frame = obj.last_seen_frame
                        duration_frames = obj.frame_count
                        first_ts = obj.trajectory[0].timestamp_seconds if obj.trajectory else 0
                        last_ts = obj.trajectory[-1].timestamp_seconds if obj.trajectory else 0

                        summary_records.append({
                            "Track ID": obj.track_id,
                            "Label": obj.display_label,
                            "Class": obj.class_name,
                            "Frames Tracked": duration_frames,
                            "First Frame": first_frame,
                            "Last Frame": last_frame,
                            "Duration (s)": f"{last_ts - first_ts:.2f}",
                            "Avg Confidence": f"{sum(tp.confidence for tp in obj.trajectory) / len(obj.trajectory) * 100:.1f}%"
                            if obj.trajectory else "N/A",
                        })

                    st.dataframe(pd.DataFrame(summary_records), use_container_width=True)

                    # Detailed trajectory for selected track
                    st.markdown("---")
                    track_labels = {
                        obj.display_label: tid for tid, obj in all_objects.items()
                    }
                    selected_label = st.selectbox(
                        "Select a tracked object to inspect trajectory:",
                        options=list(track_labels.keys()),
                    )

                    if selected_label:
                        selected_tid = track_labels[selected_label]
                        selected_obj = all_objects[selected_tid]

                        st.markdown(f"#### {selected_obj.display_label} — Trajectory Detail")
                        st.markdown(f"**Class:** {selected_obj.class_name} | **Frames Observed:** {selected_obj.frame_count}")

                        traj_records = []
                        for tp in selected_obj.trajectory:
                            traj_records.append({
                                "Frame": tp.frame_idx,
                                "Time (s)": f"{tp.timestamp_seconds:.3f}",
                                "BBox [x1,y1,x2,y2]": f"[{tp.bbox[0]},{tp.bbox[1]},{tp.bbox[2]},{tp.bbox[3]}]",
                                "Center (x,y)": f"({tp.center[0]},{tp.center[1]})",
                                "Confidence": f"{tp.confidence * 100:.1f}%",
                            })

                        st.dataframe(
                            pd.DataFrame(traj_records),
                            use_container_width=True,
                            height=min(400, 35 * len(traj_records) + 50),
                        )

                        # Trajectory displacement chart
                        if len(selected_obj.trajectory) > 1:
                            import plotly.graph_objects as go

                            centers = selected_obj.center_history
                            xs = [c[0] for c in centers]
                            ys = [c[1] for c in centers]
                            frames = [tp.frame_idx for tp in selected_obj.trajectory]

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=xs,
                                y=ys,
                                mode="lines+markers",
                                marker=dict(
                                    size=5,
                                    color=frames,
                                    colorscale="Viridis",
                                    showscale=True,
                                    colorbar=dict(title="Frame"),
                                ),
                                line=dict(width=1, color="rgba(100,100,100,0.4)"),
                                text=[f"Frame {f}" for f in frames],
                                hovertemplate="X: %{x}<br>Y: %{y}<br>%{text}<extra></extra>",
                            ))
                            fig.update_layout(
                                title=f"Trajectory Path — {selected_obj.display_label}",
                                xaxis_title="X (px)",
                                yaxis_title="Y (px)",
                                yaxis=dict(autorange="reversed"),  # Image coords: Y grows downward
                                height=400,
                                template="plotly_white",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No tracked objects found. Run the Tracking Runner first.")
            else:
                st.info("👉 Run the **Video Tracking Runner** tab first to generate trajectory data.")

        # TAB 4: Source Video & Raw Metadata
        with tab4:
            st.subheader("Source Video Stream")
            st.video(video_path_to_process)
            st.markdown("#### Technical Stream Properties")
            st.json(meta.to_dict())

        reader.release()

    except Exception as e:
        st.error(f"Error during video processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Please upload an MP4 video or select a sample in the sidebar to start detection & tracking.")
    st.markdown(
        """
        ### Stage 3 Capabilities:
        - **YOLO Detection:** Decoupled `ObjectDetector` with YOLOv8 / YOLO11 support.
        - **ByteTrack Tracking:** Persistent object IDs across frames using `supervision.ByteTrack`.
        - **Trajectory History:** Full per-object trajectory with frame, timestamp, bbox, center, and confidence.
        - **Live Visualisation:** Bounding boxes with track ID labels (e.g., *Person #3*, *Carton #7*) and trajectory trails.
        - **Trajectory Analysis:** Interactive per-track inspection with displacement charts.
        """
    )
