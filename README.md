# Godrej Warehouse AI Video Intelligence System

An AI-driven warehouse video intelligence system designed for detecting and preventing damaging handling behaviours using temporal reasoning, ByteTrack tracking, multi-factor risk categorization, automated evidence clipping, and a grounded supervisor assistant.

## System Architecture Pipeline

```text
Video Ingestion ➔ YOLO Detection ➔ ByteTrack Multi-Object Tracking ➔ Trajectories & Spatial Graph
➔ Temporal Behaviour Rules ➔ Risk Assessment Engine ➔ Incident Database ➔ Streamlit Dashboard ➔ AI Supervisor
```

## Monitored Handling Violations
1. Product Dropped
2. Product Thrown
3. Product Dragged
4. Improper Stacking
5. Unstable Stacking
6. Product Outside Pallet
7. Product Pushed
8. Product Rolled
9. Person Standing on Product
10. Incorrect Pallet/Product Positioning

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Streamlit Dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```
