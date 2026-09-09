"""
Object Detection Abstraction Module for Godrej Warehouse AI.
Provides clean encapsulation over YOLO models, normalizing bounding boxes,
confidences, and category mappings for warehouse environments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import cv2
import numpy as np


# Standard COCO to Warehouse semantic mapping
DEFAULT_WAREHOUSE_CLASS_MAPPING: Dict[str, str] = {
    # People
    "person": "person",
    
    # Products / Packages / Containers (COCO approximations)
    "suitcase": "product/carton",
    "backpack": "product/carton",
    "handbag": "product/carton",
    "box": "product/carton",
    "book": "product/carton",
    "bottle": "product/carton",
    
    # Machinery & Transport Equipment
    "car": "equipment/machinery",
    "truck": "equipment/forklift",
    "bus": "equipment/machinery",
    "forklift": "equipment/forklift",
    
    # Storage & Pallet Structures
    "chair": "pallet/structure",
    "bench": "pallet/structure",
    "dining table": "pallet/rack",
    "pallet": "pallet",
}

# Color palette for clean visualization (BGR format)
CLASS_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "person": (255, 140, 0),             # Deep Sky Blue / Cyan (in BGR: (0, 140, 255) / (255, 140, 0))
    "product/carton": (50, 205, 50),     # Lime / Emerald Green
    "carton": (50, 205, 50),             # Lime / Emerald Green
    "equipment/forklift": (0, 165, 255), # Amber / Orange
    "forklift": (0, 165, 255),           # Amber / Orange
    "equipment/machinery": (0, 215, 255),# Gold
    "machinery": (0, 215, 255),          # Gold
    "pallet": (180, 105, 255),           # Purple / Pink
    "pallet/structure": (147, 112, 219), # Medium Purple
    "pallet/rack": (128, 128, 240),      # Slate Blue
    "default": (180, 180, 180),          # Neutral Gray
}


@dataclass
class Detection:
    """Represents a single detected object in a frame."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    raw_class_name: str

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        """Centroid coordinates (cx, cy)."""
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    @property
    def foot_point(self) -> Tuple[int, int]:
        """Bottom-center coordinate (ground contact point)."""
        return (self.x1 + self.width // 2, self.y2)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        return self.width / self.height if self.height > 0 else 1.0

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "raw_class_name": self.raw_class_name,
            "center": list(self.center),
            "foot_point": list(self.foot_point),
            "area": self.area,
        }


@dataclass
class DetectionResult:
    """Encapsulates the detection output for a single video frame."""
    frame_idx: int
    timestamp_seconds: float
    detections: List[Detection] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def counts_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        return [d for d in self.detections if d.class_name.lower() == class_name.lower()]


class ObjectDetector:
    """
    Detector abstraction wrapping Ultralytics YOLO inference.
    Shields the rest of the application from YOLO implementation specifics.
    """

    def __init__(
        self,
        model_name_or_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
        class_mapping: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initializes the YOLO Object Detector.

        Args:
            model_name_or_path: YOLO model path (e.g., 'yolo11n.pt', 'yolov8n.pt', or custom weights).
            confidence_threshold: Minimum confidence score (0.0 to 1.0) to retain a detection.
            iou_threshold: IoU threshold for Non-Max Suppression (NMS).
            target_classes: List of allowed class names (filtered by mapped or raw names).
            class_mapping: Mapping dict converting raw model class names to warehouse domain names.
            device: 'cpu', 'cuda', '0', etc. Defaults to auto-selection.
        """
        self.model_name_or_path = model_name_or_path
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.target_classes = set(c.lower() for c in target_classes) if target_classes else None
        self.class_mapping = class_mapping if class_mapping is not None else DEFAULT_WAREHOUSE_CLASS_MAPPING
        self.device = device

        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name_or_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{self.model_name_or_path}': {str(e)}")

    def set_confidence_threshold(self, threshold: float) -> None:
        """Updates the confidence threshold at runtime."""
        self.confidence_threshold = max(0.0, min(1.0, float(threshold)))

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp_seconds: float = 0.0,
        annotate: bool = True,
    ) -> DetectionResult:
        """
        Performs object detection on a single BGR frame.

        Args:
            frame: Input BGR image as a numpy array.
            frame_idx: 0-indexed sequential frame number.
            timestamp_seconds: Elapsed video timestamp.
            annotate: Whether to generate an annotated visualization frame.

        Returns:
            DetectionResult: Structured detection output.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(frame_idx=frame_idx, timestamp_seconds=timestamp_seconds)

        start_time = time.perf_counter()

        # Run YOLO inference
        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Detection] = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            names = result.names

            for box in boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                raw_name = names.get(cls_id, str(cls_id)).lower()

                # Map raw COCO name to warehouse domain concept
                mapped_name = self.class_mapping.get(raw_name, raw_name)

                # Filter target classes if specified
                if self.target_classes is not None:
                    if raw_name not in self.target_classes and mapped_name.lower() not in self.target_classes:
                        continue

                # Ensure valid bounding box coordinates
                h_img, w_img = frame.shape[:2]
                x1 = max(0, min(w_img - 1, x1))
                y1 = max(0, min(h_img - 1, y1))
                x2 = max(0, min(w_img, x2))
                y2 = max(0, min(h_img, y2))

                if x2 > x1 and y2 > y1:
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=mapped_name,
                            raw_class_name=raw_name,
                        )
                    )

        annotated = self.annotate_frame(frame, detections) if annotate else None

        return DetectionResult(
            frame_idx=frame_idx,
            timestamp_seconds=timestamp_seconds,
            detections=detections,
            annotated_frame=annotated,
            inference_time_ms=inference_time_ms,
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draws clean bounding boxes, class labels, and confidence tags on the frame.
        """
        annotated = frame.copy()

        for det in detections:
            color = CLASS_COLOR_PALETTE.get(det.class_name, CLASS_COLOR_PALETTE.get(det.raw_class_name, CLASS_COLOR_PALETTE["default"]))
            x1, y1, x2, y2 = det.bbox

            # Draw outer bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            # Draw bottom-center contact point
            cv2.circle(annotated, det.foot_point, 4, color, -1)

            # Format label: "class_name 89%"
            label = f"{det.class_name} {det.confidence * 100:.0f}%"

            # Compute label background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_y1 = max(0, y1 - text_height - 6)
            label_y2 = y1
            label_x2 = min(frame.shape[1], x1 + text_width + 8)

            # Draw label background tag
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (label_x2, label_y2),
                color,
                -1,
            )

            # Text color (white or dark depending on color brightness)
            text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 400 else (255, 255, 255)

            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA,
            )

        return annotated
