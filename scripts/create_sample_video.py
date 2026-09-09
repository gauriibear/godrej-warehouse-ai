"""
Utility script to generate a sample warehouse CCTV simulation video for quick testing.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_sample_video(output_path: str = "videos/warehouse_cctv_sample.mp4", duration_sec: int = 5, fps: int = 30):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    width, height = 1280, 720
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    
    for i in range(total_frames):
        # Draw warehouse floor & aisle
        frame = np.full((height, width, 3), 50, dtype=np.uint8)
        
        # Floor grid lines
        for y in range(200, height, 80):
            cv2.line(frame, (0, y), (width, y), (70, 70, 70), 1)
        
        # Staging Zone & Rack representation
        cv2.rectangle(frame, (100, 150), (450, 650), (40, 60, 40), -1)
        cv2.putText(frame, "ZONE A: PALLET STAGING", (110, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 2)
        
        # Aisle Lane
        cv2.rectangle(frame, (500, 150), (1200, 650), (40, 40, 60), -1)
        cv2.putText(frame, "AISLE 04 - FORKLIFT LANE", (510, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 2)
        
        # Moving worker representation (mock box)
        worker_x = 200 + int(np.sin(i / 15.0) * 80)
        worker_y = 350 + int((i / total_frames) * 150)
        cv2.rectangle(frame, (worker_x, worker_y), (worker_x + 50, worker_y + 110), (0, 165, 255), -1)
        cv2.putText(frame, "WORKER", (worker_x - 10, worker_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Stamped CCTV Header
        timestamp_str = f"2026-09-07 10:15:{i // fps:02d}.{(i % fps) * 33:03d} | CAM_04_SOUTH_BAY"
        cv2.rectangle(frame, (0, 0), (width, 45), (20, 20, 20), -1)
        cv2.putText(frame, timestamp_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Frame counter
        cv2.putText(frame, f"FRAME: {i:04d}/{total_frames}", (width - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        writer.write(frame)
        
    writer.release()
    print(f"Sample video created: {output_path} ({total_frames} frames, {width}x{height} @ {fps} FPS)")


if __name__ == "__main__":
    generate_sample_video()
