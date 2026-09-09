"""
Evidence Collector for Godrej Warehouse AI Risk Intelligence.
Stage 5: Frame ring buffer, keyframe snapshot generation with risk overlays,
short before/after video clip extraction, and incident manifest serialization.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.risk.models import IncidentRecord, RiskLevel


class EvidenceCollector:
    """
    Captures visual evidence for confirmed handling incidents.
    Maintains a rolling ring buffer of video frames to extract before/after
    clips and generates annotated keyframe screenshots.
    """

    def __init__(
        self,
        output_dir: str = "outputs/evidence",
        pre_event_frames: int = 30,
        post_event_frames: int = 30,
        max_buffer_len: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pre_event_frames = pre_event_frames
        self.post_event_frames = post_event_frames

        # Circular buffer: stores (frame_idx, frame_bgr)
        # Sized to hold both pre and post windows comfortably (at least 150 frames)
        buffer_len = max_buffer_len or max(150, (pre_event_frames + post_event_frames) * 4)
        self._frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=buffer_len)

        # Pending clip capture queue: incident_id -> {incident: IncidentRecord, target_frame_idx: int}
        self._pending_clips: Dict[str, Dict] = {}

    def reset(self) -> None:
        """Clears frame buffer and pending clips."""
        self._frame_buffer.clear()
        self._pending_clips.clear()

    def add_frame(self, frame_idx: int, frame: np.ndarray) -> None:
        """Adds a video frame to the rolling circular buffer."""
        # Store a copy to prevent in-place mutation
        self._frame_buffer.append((frame_idx, frame.copy()))

    def save_keyframe(
        self,
        incident: IncidentRecord,
        frame: np.ndarray,
        annotate: bool = True,
    ) -> str:
        """
        Renders and saves an annotated keyframe snapshot with risk banner,
        bounding box highlight, and diagnostic metrics.
        Returns the relative filepath of the saved image.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if annotate:
            # 1. Color mapping based on risk level
            color_bgr = (0, 0, 180)  # Default dark red
            if incident.risk_level == RiskLevel.LOW:
                color_bgr = (200, 100, 30)   # Blue
            elif incident.risk_level == RiskLevel.MEDIUM:
                color_bgr = (0, 140, 220)    # Amber
            elif incident.risk_level == RiskLevel.HIGH:
                color_bgr = (30, 30, 220)    # Red
            elif incident.risk_level == RiskLevel.CRITICAL:
                color_bgr = (15, 15, 140)    # Dark Crimson

            # 2. Top Risk & Violation Banner
            banner_h = 50
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), color_bgr, -1)
            cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)

            banner_text = (
                f"[{incident.risk_level.value} RISK | SCORE {incident.risk_score:.0f}] "
                f"{incident.human_readable_name.upper()} | {incident.display_label} "
                f"| Time: {incident.timestamp_seconds:.2f}s (Frame #{incident.frame_idx})"
            )
            cv2.putText(
                annotated,
                banner_text,
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 3. Highlight Offending Bounding Box
            x1, y1, x2, y2 = incident.bbox
            if x2 > x1 and y2 > y1:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3)

                # Diagnostic sub-tag beneath box
                tag_text = f"Risk: {incident.risk_score:.0f}/100"
                if "vertical_velocity_px_s" in incident.metrics:
                    tag_text += f" | {incident.metrics['vertical_velocity_px_s']:.0f} px/s"
                elif "impulse_spike_px_s" in incident.metrics:
                    tag_text += f" | Spike: {incident.metrics['impulse_spike_px_s']:.0f} px/s"
                elif "offset_ratio" in incident.metrics:
                    tag_text += f" | Offset: {incident.metrics['offset_ratio']*100:.0f}%"
                elif "overhang_ratio" in incident.metrics:
                    tag_text += f" | Overhang: {incident.metrics['overhang_ratio']*100:.0f}%"

                tag_y = min(h - 5, y2 + 20)
                cv2.rectangle(annotated, (x1, y2), (x1 + 180, tag_y + 4), color_bgr, -1)
                cv2.putText(
                    annotated,
                    tag_text,
                    (x1 + 4, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Write keyframe
        out_filename = f"{incident.incident_id}.jpg"
        out_path = self.output_dir / out_filename
        cv2.imwrite(str(out_path), annotated)
        incident.evidence_image_path = str(out_path)
        return str(out_path)

    def extract_clip(
        self,
        incident: IncidentRecord,
        fps: float = 30.0,
    ) -> Optional[str]:
        """
        Extracts a before/after video clip for the incident from the rolling frame buffer.
        Returns the saved video clip filepath if sufficient frames exist.
        """
        if not self._frame_buffer:
            return None

        event_frame = incident.frame_idx
        start_frame = event_frame - self.pre_event_frames
        end_frame = event_frame + self.post_event_frames

        # Filter buffered frames within window
        clip_frames = [
            frame for f_idx, frame in self._frame_buffer
            if start_frame <= f_idx <= end_frame
        ]

        if len(clip_frames) < 5:
            # Not enough frames buffered for a video clip
            return None

        out_filename = f"{incident.incident_id}.mp4"
        out_path = self.output_dir / out_filename

        h, w = clip_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, max(1.0, float(fps)), (w, h))

        for f in clip_frames:
            writer.write(f)
        writer.release()

        incident.video_clip_path = str(out_path)
        return str(out_path)

    def save_manifest(self, incidents: List[IncidentRecord]) -> str:
        """
        Serializes all session incidents to a master JSON manifest.
        Returns the saved manifest filepath.
        """
        manifest_path = self.output_dir / "incidents.json"
        data = [inc.to_dict() for inc in incidents]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(manifest_path)
