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
from app.behaviour.engine import BehaviourEngine, BehaviourConfig, BehaviourEvent
from app.risk import (
    RiskEngine,
    RiskConfig,
    EvidenceCollector,
    RiskLevel,
    DamageAssessmentStatus,
    IncidentRecord,
)
from app.assistant import (
    OperationsAssistant,
    AssistantResponse,
    QueryIntent,
)


# Page Configuration
st.set_page_config(
    page_title="Godrej Warehouse AI - Risk Engine & Operations Assistant",
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
st.markdown('<div class="stage-badge">STAGE 5: RISK ENGINE & AI OPERATIONS ASSISTANT</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">📦 Godrej Warehouse AI Video Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-class object detection, ByteTrack tracking, behaviour reasoning, explainable risk scoring, visual evidence capture, and grounded AI operations assistant.</div>',
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

# Behaviour Reasoning Settings in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("⚠️ Behaviour & Risk Engine")

enable_behaviour = st.sidebar.checkbox(
    "Enable Behaviour Reasoning",
    value=True,
    help="Detect violations: drops, throws, dragging, kicking/pushing, rolling, unstable stacking, pallet overhang.",
)

cooldown_window = st.sidebar.slider(
    "Event Cooldown Window (frames)",
    min_value=15,
    max_value=180,
    value=60,
    step=15,
    help="Cooldown frames before the same track can re-trigger the same violation.",
)

enable_risk = st.sidebar.checkbox(
    "Enable Risk Engine & Evidence Capture",
    value=True,
    help="Classify incidents into LOW/MEDIUM/HIGH/CRITICAL and capture visual evidence keyframes & clips.",
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

        # Tabs for Detection, Tracking, Behaviour, and Inspection
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Frame-by-Frame Detection Inspector",
            "🔗 Video Tracking & Risk Runner",
            "📊 Trajectory Analysis",
            "🚨 Risk Engine & Incident Evidence",
            "🤖 AI Operations Assistant",
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

        # TAB 2: Tracking & Behaviour Runner
        with tab2:
            st.subheader("Process Video with Detection + Tracking + Behaviour Analysis")
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

            if st.button("▶️ Run Detection + Tracking + Behaviour + Risk", type="primary"):
                # Initialise tracker fresh for each run
                tracker = ObjectTracker(
                    track_activation_threshold=conf_threshold,
                    lost_track_buffer=lost_track_buffer,
                    minimum_matching_threshold=match_threshold,
                    frame_rate=int(meta.fps) if meta.fps > 0 else 30,
                )

                # Initialise behaviour engine if enabled
                behaviour_engine = None
                if enable_behaviour:
                    behaviour_config = BehaviourConfig(cooldown_frames=cooldown_window)
                    behaviour_engine = BehaviourEngine(config=behaviour_config)

                # Initialise risk engine & evidence collector if enabled
                risk_engine = None
                evidence_collector = None
                if enable_risk:
                    risk_engine = RiskEngine(config=RiskConfig())
                    evidence_collector = EvidenceCollector(output_dir="outputs/evidence")

                # Reset reader position
                reader.seek_frame(0)

                progress_bar = st.progress(0)
                status_text = st.empty()
                alert_banner = st.empty()
                video_placeholder = st.empty()

                # Metrics containers
                metric_cols = st.columns(5)
                metric_frames = metric_cols[0].empty()
                metric_detections = metric_cols[1].empty()
                metric_active = metric_cols[2].empty()
                metric_unique = metric_cols[3].empty()
                metric_violations = metric_cols[4].empty()

                active_tracks_placeholder = st.empty()

                processed_count = 0
                total_detections = 0
                total_inference_time = 0.0

                for idx, ts, frame in reader.frames(max_frames=max_process_frames):
                    if idx % frame_skip != 0:
                        continue

                    # Buffer frame for evidence extraction
                    if evidence_collector is not None:
                        evidence_collector.add_frame(idx, frame)

                    # Stage 2: Detect
                    det_res = detector.detect(frame, frame_idx=idx, timestamp_seconds=ts, annotate=False)
                    total_detections += det_res.count
                    total_inference_time += det_res.inference_time_ms

                    # Stage 3: Track
                    trk_res = tracker.update(det_res, frame, annotate=True)
                    processed_count += 1

                    # Stage 4: Behaviour Reasoning
                    frame_events = []
                    if behaviour_engine is not None:
                        frame_events = behaviour_engine.evaluate_frame(
                            tracker=tracker,
                            current_frame_idx=idx,
                            timestamp=ts,
                            fps=meta.fps,
                        )

                    # Stage 5: Risk Engine Evaluation & Evidence Capture
                    frame_incidents = []
                    if risk_engine is not None and frame_events:
                        for ev in frame_events:
                            incident = risk_engine.evaluate_event(ev)
                            frame_incidents.append(incident)
                            if evidence_collector is not None:
                                evidence_collector.save_keyframe(incident, frame, annotate=True)
                                evidence_collector.extract_clip(incident, fps=meta.fps)

                    # Live video frame display with tracking + behaviour alert overlays
                    if trk_res.annotated_frame is not None:
                        display_frame = trk_res.annotated_frame
                        if behaviour_engine is not None:
                            display_frame = behaviour_engine.annotate_events(
                                frame=display_frame,
                                events=frame_events,
                                current_frame_idx=idx,
                                tracked_objects=tracker.tracked_objects,
                            )

                        violation_count_now = len(behaviour_engine.all_events) if behaviour_engine else 0
                        video_placeholder.image(
                            cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                            caption=f"Frame #{idx} | Time: {ts:.2f}s | Active Tracks: {trk_res.active_count} | Violations: {violation_count_now}",
                            use_container_width=True,
                        )

                    # Display alert banner if incidents/violations occurred in this frame
                    if frame_incidents:
                        for inc in frame_incidents:
                            alert_banner.error(
                                f"🚨 **{inc.risk_level.value} RISK INCIDENT**: {inc.human_readable_name.upper()} on {inc.display_label} "
                                f"(Score: {inc.risk_score:.0f}/100 | Frame #{idx} @ {ts:.2f}s)"
                            )
                    elif frame_events:
                        for ev in frame_events:
                            alert_banner.warning(
                                f"⚠️ **VIOLATION DETECTED**: {ev.human_readable_name.upper()} on {ev.display_label} "
                                f"(Frame #{idx} @ {ts:.2f}s)"
                            )

                    # Update live metrics
                    metric_frames.metric("Frames", processed_count)
                    metric_detections.metric("Detections", total_detections)
                    metric_active.metric("Active Tracks", trk_res.active_count)
                    metric_unique.metric("Unique Objects", tracker.total_unique_tracks)
                    metric_violations.metric(
                        "Violations",
                        len(behaviour_engine.all_events) if behaviour_engine else 0,
                    )

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

                violation_total = len(behaviour_engine.all_events) if behaviour_engine else 0
                incident_total = len(risk_engine.all_incidents) if risk_engine else 0
                status_text.success(
                    f"✅ Processed {processed_count} frames! "
                    f"Avg Latency: {avg_latency:.1f} ms ({avg_fps:.1f} FPS) | "
                    f"Unique Objects: {tracker.total_unique_tracks} | "
                    f"Violations: {violation_total} | "
                    f"Risk Incidents: {incident_total}"
                )

                # Finalize video clips with buffered post-event frames
                if evidence_collector is not None and risk_engine is not None:
                    for inc in risk_engine.all_incidents:
                        evidence_collector.extract_clip(inc, fps=meta.fps)

                # Store tracker, behaviour engine, and risk engine in session state
                st.session_state["tracker_result"] = tracker
                st.session_state["behaviour_engine"] = behaviour_engine
                st.session_state["risk_engine"] = risk_engine
                st.session_state["evidence_collector"] = evidence_collector

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
                st.info("👉 Run the **Video Tracking & Behaviour Runner** tab first to generate trajectory data.")

        # TAB 4: Risk Engine & Incident Evidence
        with tab4:
            st.subheader("🚨 Risk Engine & Incident Evidence")

            # Godrej Policy Banner
            st.info(
                "ℹ️ **Godrej Damage Assessment Policy**: "
                "Observed Behaviour → Potential Damage Risk → Confirmed Damage. "
                "Detected incidents represent Potential Risk to packaging integrity and trigger damage prevention workflows; "
                "confirmed physical damage is only recorded upon physical verification."
            )

            risk_eng = st.session_state.get("risk_engine")
            ev_collector = st.session_state.get("evidence_collector")

            if risk_eng is not None:
                incidents = risk_eng.all_incidents

                if incidents:
                    # Top 4 Risk Tier KPI metrics
                    crit_count = sum(1 for i in incidents if i.risk_level == RiskLevel.CRITICAL)
                    high_count = sum(1 for i in incidents if i.risk_level == RiskLevel.HIGH)
                    med_count = sum(1 for i in incidents if i.risk_level == RiskLevel.MEDIUM)
                    low_count = sum(1 for i in incidents if i.risk_level == RiskLevel.LOW)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Critical Incidents", crit_count)
                    with c2:
                        st.metric("High Risk Incidents", high_count)
                    with c3:
                        st.metric("Medium Risk Incidents", med_count)
                    with c4:
                        st.metric("Low Risk Incidents", low_count)

                    st.markdown("---")
                    st.subheader("🔍 Incident Evidence Inspector")

                    # Dropdown selector
                    inc_labels = [
                        f"[{i.risk_level.value}] {i.incident_id} — {i.human_readable_name} ({i.display_label} @ {i.timestamp_seconds:.2f}s)"
                        for i in incidents
                    ]
                    selected_idx = st.selectbox(
                        "Select an incident to view visual keyframe evidence and hazard diagnostics:",
                        range(len(incidents)),
                        format_func=lambda idx: inc_labels[idx],
                    )
                    selected_inc = incidents[selected_idx]

                    col_ev1, col_ev2 = st.columns([1.2, 1.0])
                    with col_ev1:
                        st.markdown(f"#### Visual Keyframe Evidence ({selected_inc.incident_id})")
                        if selected_inc.evidence_image_path and Path(selected_inc.evidence_image_path).exists():
                            st.image(
                                selected_inc.evidence_image_path,
                                caption=f"Keyframe Snapshot | {selected_inc.display_label} | Frame #{selected_inc.frame_idx}",
                                use_container_width=True,
                            )
                        else:
                            st.warning("Keyframe image not available.")

                        st.markdown("#### Video Clip Snippet")
                        clip_p = Path(selected_inc.video_clip_path) if selected_inc.video_clip_path else None
                        if clip_p and clip_p.exists() and clip_p.stat().st_size > 0:
                            try:
                                with open(clip_p, "rb") as vf:
                                    video_bytes = vf.read()
                                st.video(video_bytes, format="video/mp4")
                            except Exception as e:
                                st.warning(f"Video clip replay unavailable: {e}")
                        else:
                            st.info("ℹ️ Video clip replay unavailable for this incident.")

                    with col_ev2:
                        st.markdown("#### Risk Assessment Profile")

                        # Risk Level Badge & Score
                        st.markdown(
                            f'<div style="background-color: {selected_inc.risk_level.color_hex}; color: white; '
                            f'padding: 8px 16px; border-radius: 8px; font-size: 1.1rem; font-weight: 700; display: inline-block;">'
                            f'{selected_inc.risk_level.value} RISK (Score: {selected_inc.risk_score:.0f}/100)</div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(selected_inc.risk_score / 100.0)

                        st.markdown(f"**Damage Status:** `{selected_inc.damage_assessment.value}`")
                        st.markdown(f"**Entity:** {selected_inc.display_label}" + (f" (Worker: {selected_inc.secondary_label})" if selected_inc.secondary_label else ""))
                        st.markdown(f"**Timestamp:** {selected_inc.timestamp_seconds:.2f}s (Frame #{selected_inc.frame_idx})")
                        st.markdown(f"**Visual Confidence:** {selected_inc.confidence * 100:.1f}%")

                        st.markdown("##### Hazard Explanation:")
                        st.info(selected_inc.explanation)

                        st.markdown("##### Damage Prevention Recommendation:")
                        st.success(selected_inc.recommendation)

                        st.markdown("##### Kinematic Metrics:")
                        st.json(selected_inc.metrics)

                    st.markdown("---")
                    st.subheader("📋 Complete Incident Manifest")
                    manifest_records = []
                    for inc in incidents:
                        manifest_records.append({
                            "Incident ID": inc.incident_id,
                            "Risk Level": inc.risk_level.value,
                            "Risk Score": f"{inc.risk_score:.0f}/100",
                            "Violation Rule": inc.human_readable_name,
                            "Target": inc.display_label,
                            "Frame #": inc.frame_idx,
                            "Time (s)": f"{inc.timestamp_seconds:.2f}",
                            "Assessment": inc.damage_assessment.value,
                            "Recommendation": inc.recommendation,
                        })
                    st.dataframe(pd.DataFrame(manifest_records), use_container_width=True)

                    # Export JSON Manifest
                    import json
                    manifest_json = json.dumps([i.to_dict() for i in incidents], indent=2)
                    st.download_button(
                        label="📥 Download Incidents Manifest (JSON)",
                        data=manifest_json,
                        file_name="warehouse_incidents_manifest.json",
                        mime="application/json",
                    )

                else:
                    st.success("✅ **No handling risk incidents detected** in the processed frames. All observed handling conformed to safety policies.")
            else:
                st.info("👉 Run the **Video Tracking & Risk Runner** tab first to evaluate handling risks and capture evidence.")

        # TAB 5: AI Operations Assistant
        with tab5:
            st.subheader("🤖 AI Operations Assistant")
            st.markdown(
                "Grounded operational query engine for warehouse supervisors. "
                "Answers queries about handling violations, risk classifications, repeat infractions, "
                "and prevention recommendations based strictly on verified incident telemetry."
            )
            st.info(
                "ℹ️ **Godrej Damage Assessment Policy**: "
                "Observed Behaviour → Potential Damage Risk → Confirmed Damage. "
                "All incidents represent Potential Damage Risk (Unconfirmed); physical damage requires physical QA inspection."
            )

            # Determine available incidents
            asst_incidents = []
            source_note = ""
            risk_eng = st.session_state.get("risk_engine")
            if risk_eng is not None and risk_eng.all_incidents:
                asst_incidents = risk_eng.all_incidents
                source_note = f"Active session ({len(asst_incidents)} incidents evaluated from video)"
            elif Path("outputs/evidence/incidents.json").exists():
                manifest_path = Path("outputs/evidence/incidents.json")
                temp_asst = OperationsAssistant.from_manifest(manifest_path)
                asst_incidents = temp_asst.incidents
                source_note = f"Loaded from `outputs/evidence/incidents.json` ({len(asst_incidents)} incidents)"

            if asst_incidents:
                assistant = OperationsAssistant.from_incidents(asst_incidents)
                st.caption(f"📊 **Data Source**: {source_note}")

                # Quick KPI Row
                q_crit = sum(1 for i in asst_incidents if i.risk_level == RiskLevel.CRITICAL)
                q_high = sum(1 for i in asst_incidents if i.risk_level == RiskLevel.HIGH)
                q_med = sum(1 for i in asst_incidents if i.risk_level == RiskLevel.MEDIUM)
                q_low = sum(1 for i in asst_incidents if i.risk_level == RiskLevel.LOW)

                kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
                kpi_c1.metric("Total Incidents", len(asst_incidents))
                kpi_c2.metric("Critical Risk", q_crit)
                kpi_c3.metric("High Risk", q_high)
                kpi_c4.metric("Medium Risk", q_med)
                kpi_c5.metric("Low Risk", q_low)

                st.markdown("#### ⚡ Quick Actions")
                btn_c1, btn_c2, btn_c3, btn_c4, btn_c5 = st.columns(5)

                selected_quick_prompt = None
                with btn_c1:
                    if st.button("📋 Executive Summary", key="btn_summary", use_container_width=True):
                        selected_quick_prompt = "Summarize the current incidents"
                with btn_c2:
                    if st.button("🚨 Top Highest Risks", key="btn_highest", use_container_width=True):
                        selected_quick_prompt = "What are the highest-risk incidents?"
                with btn_c3:
                    if st.button("📈 Behaviour Frequency", key="btn_frequency", use_container_width=True):
                        selected_quick_prompt = "Which behaviour occurs most frequently?"
                with btn_c4:
                    if st.button("🔁 Repeat Infractions", key="btn_repeat", use_container_width=True):
                        selected_quick_prompt = "Show repeat violations across all entities"
                with btn_c5:
                    if st.button("🛡️ Prevention Guide", key="btn_prevention", use_container_width=True):
                        selected_quick_prompt = "What prevention recommendations apply?"

                # Session chat history initialization
                if "assistant_chat_history" not in st.session_state:
                    st.session_state["assistant_chat_history"] = []

                # Handle quick prompt or chat input
                user_query = st.chat_input("Ask about incident causes, behaviour frequency, highest risks, Carton #7 history, or prevention...")
                active_query = user_query or selected_quick_prompt

                if active_query:
                    res = assistant.ask(active_query)
                    st.session_state["assistant_chat_history"].append({
                        "user": active_query,
                        "assistant": res.answer,
                        "matched_incidents": res.matched_incidents,
                    })

                # Display Chat Messages
                if st.session_state["assistant_chat_history"]:
                    for chat_item in st.session_state["assistant_chat_history"]:
                        with st.chat_message("user"):
                            st.markdown(chat_item["user"])
                        with st.chat_message("assistant"):
                            st.markdown(chat_item["assistant"])

                            # If matched incidents have evidence, show expandable preview
                            matched = chat_item.get("matched_incidents", [])
                            evidence_incidents = [m for m in matched if m.evidence_image_path and Path(m.evidence_image_path).exists()]
                            if evidence_incidents:
                                with st.expander(f"📷 View Visual Evidence for Referenced Incidents ({len(evidence_incidents)})"):
                                    ev_cols = st.columns(min(3, len(evidence_incidents)))
                                    for idx_ev, ev_inc in enumerate(evidence_incidents[:6]):
                                        with ev_cols[idx_ev % len(ev_cols)]:
                                            st.image(
                                                ev_inc.evidence_image_path,
                                                caption=f"{ev_inc.incident_id} | {ev_inc.display_label} ({ev_inc.risk_level.value})",
                                                use_container_width=True,
                                            )

                    if st.button("🗑️ Clear Chat History", key="clear_chat_btn"):
                        st.session_state["assistant_chat_history"] = []
                        st.rerun()

            else:
                st.warning(
                    "⚠️ **No incident records available to query.**\n\n"
                    "To use the assistant:\n"
                    "1. Run the **Video Tracking & Risk Runner** tab on a video, OR\n"
                    "2. Run `python scripts/generate_evidence.py` to generate synthetic warehouse incident data."
                )

        # TAB 6: Source Video & Raw Metadata
        with tab6:
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
        r"""
        ### Stage 5 Capabilities:
        - **YOLO Detection:** Multi-class detection with YOLOv8 / YOLO11 backbone.
        - **ByteTrack Tracking:** Persistent object IDs and trajectory histories across frames.
        - **Temporal Behaviour Rules:** 7 core rules (Drop, Throw, Drag, Push/Kick, Roll, Stacking, Pallet Overhang).
        - **Risk Engine:** Explainable, rule-based risk classification into **LOW**, **MEDIUM**, **HIGH**, and **CRITICAL** tiers.
        - **Metric Escalation:** Velocity, impulse, duration, overhang, and stacking offset multipliers.
        - **Chronic Repeat Penalty:** Multi-infraction escalation on identical entities to trigger quarantine review.
        - **Incident Evidence Capture:** Automatic annotated keyframe screenshots, before/after video clips, and JSON manifests.
        - **AI Operations Assistant:** Grounded, explainable supervisor query engine answering incident, risk, and repeat infraction questions.
        - **Responsible AI:** Strictly separates *Observed Behaviour* from *Potential Damage Risk* and *Confirmed Damage*.
        """
    )
