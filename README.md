# 🚨 Godrej Warehouse AI — Video Intelligence System

> **AI-powered warehouse video intelligence for safer, damage-free operations.**

Godrej Warehouse AI is an end-to-end computer vision system designed to transform warehouse CCTV/video footage into **actionable safety and damage-prevention intelligence**.

Instead of simply recording incidents, the system analyzes warehouse activity to identify unsafe handling behaviour, assess potential damage risk, capture visual evidence, and provide explainable recommendations to warehouse supervisors.

---

## 🎯 Problem Statement

Traditional warehouse CCTV is mainly **retrospective** — supervisors review footage after an incident has already happened.

This project aims to move from:

> **"What happened?" → "What is happening, how risky is it, and what should we do?"**

The system uses AI-based video understanding to detect unsafe handling behaviours, evaluate their potential risk, preserve evidence, and support proactive intervention.

---

## 💡 Solution

The system converts warehouse video into operational intelligence through a complete AI pipeline:

```text
Warehouse Video
      ↓
YOLO Object Detection
      ↓
ByteTrack Multi-Object Tracking
      ↓
Trajectory & Spatial Analysis
      ↓
Temporal Behaviour Detection
      ↓
Risk Assessment Engine
      ↓
Incident & Evidence Generation
      ↓
Streamlit Operations Dashboard
      ↓
AI Operations Assistant
      ↓
Preventive Recommendation
```

The goal is not only to **detect unsafe behaviour**, but to help warehouse teams **understand, review and prevent potential damage**.

---

## 🧠 Core Capabilities

### 🎥 AI Video Understanding

The system processes warehouse video and detects relevant objects such as:

* People
* Products / cartons
* Forklifts
* Other objects detected by the vision model

The detected objects are continuously analyzed across video frames.

### 🎯 Multi-Object Tracking

The system uses **ByteTrack** to maintain persistent identities across frames.

Example:

```text
Person #12
Carton #10
Forklift #2
```

Tracking allows the system to understand how objects move and interact over time instead of treating every video frame independently.

### ⚠️ Unsafe Behaviour Detection

The temporal behaviour engine analyzes movement, trajectories and spatial relationships to identify unsafe handling patterns.

Current implemented behaviours include:

1. Product Dropped
2. Product Thrown
3. Product Dragged
4. Product Pushed
5. Product Rolled
6. Unstable Stacking
7. Pallet Overhang / Product Outside Pallet

The architecture is designed to support additional warehouse scenarios as development continues.

---

## 🚨 Risk Assessment Engine

Every detected behaviour can be converted into an explainable risk score.

### Risk Levels

| Risk Level  |  Score | Meaning                      |
| ----------- | -----: | ---------------------------- |
| 🟢 LOW      |   0–29 | Low concern                  |
| 🟡 MEDIUM   |  30–59 | Requires attention           |
| 🟠 HIGH     |  60–84 | Significant potential risk   |
| 🔴 CRITICAL | 85–100 | Immediate review recommended |

Risk assessment can consider factors such as:

* Behaviour type
* Movement intensity
* Drop/descent characteristics
* Object interaction
* Stacking configuration
* Pallet overhang
* Duration
* Repeated violations
* Behaviour-specific measurements

The scoring system is **rule-based and explainable**, allowing supervisors to understand why an incident received a particular risk level.

---

## 📸 Incident & Evidence Generation

When a significant event is detected, the system creates an incident record containing information such as:

* Incident ID
* Behaviour detected
* Risk level
* Risk score
* Timestamp
* Frame number
* Object / track ID
* Confidence
* Behaviour metrics
* Explanation
* Recommended action
* Evidence image
* Evidence video clip

Example:

```text
CRITICAL RISK — 93/100

Behaviour:
Product Dropped

Object:
Carton #1

Evidence:
Keyframe + video clip

Status:
Potential Damage Risk
```

### Important Safety Principle

The system distinguishes between:

```text
Observed Behaviour
        ↓
Potential Damage Risk
        ↓
Confirmed Damage
```

The AI **does not automatically claim that physical product damage has occurred**.

Physical damage requires human or physical verification.

---

# 🤖 AI Operations Assistant

The system includes a grounded **AI Operations Assistant** designed to help warehouse supervisors understand detected incidents.

Example questions:

```text
What are the highest-risk incidents?

Why was this incident classified as high risk?

What happened repeatedly to Carton #7?

Which behaviour occurred most frequently?

What action should the supervisor take?
```

The assistant can provide:

* Risk level
* Risk score
* Detected behaviour
* Relevant metrics
* Repeat-violation context
* Risk explanation
* Recommended preventive action
* Potential damage status

The assistant is grounded in available incident data and is designed to avoid inventing unsupported incidents or conclusions.

---

# 📊 Operations Dashboard

The Streamlit dashboard provides a centralized interface for warehouse intelligence.

### Dashboard capabilities

* 🎥 Video upload and processing
* 🔍 Object detection inspection
* 🎯 Multi-object tracking
* ⚠️ Behaviour violation detection
* 🚨 Risk incident monitoring
* 📸 Evidence inspection
* 🎬 Incident video clips
* 📋 Incident manifest
* 📈 Risk information
* 🤖 AI Operations Assistant

The intended workflow is:

```text
VIDEO
  ↓
DETECT
  ↓
IDENTIFY BEHAVIOUR
  ↓
ASSESS RISK
  ↓
CAPTURE EVIDENCE
  ↓
EXPLAIN INCIDENT
  ↓
RECOMMEND ACTION
```

---

# 🧪 Demonstration Evidence Generator

The project includes a deterministic evidence generator for controlled testing and demonstrations.

It can generate representative scenarios such as:

* Critical product drop
* High-risk product drop
* Product dragging
* Unstable stacking
* Critical stacking risk
* Repeated handling violations

Generated evidence is stored as:

```text
outputs/
└── evidence/
    ├── incidents.json
    ├── INC-*.jpg
    └── INC-*.mp4
```

This allows the complete:

```text
Behaviour → Risk → Evidence → Assistant
```

pipeline to be demonstrated even when a suitable real-world warehouse video is unavailable.

---

# 🔁 Repeated Behaviour Detection

Repeated unsafe behaviour can increase the severity of an incident.

For example:

```text
Carton #7

Product Dragged
      ↓
Product Pushed
      ↓
Product Dropped
```

The risk engine can apply escalation based on repeated violations, helping identify patterns that may require intervention rather than treating every event as an isolated incident.

---

# 🛠️ Technology Stack

| Component           | Technology                    |
| ------------------- | ----------------------------- |
| Programming         | Python                        |
| Object Detection    | YOLO                          |
| Object Tracking     | ByteTrack                     |
| Video Processing    | OpenCV                        |
| Behaviour Analysis  | Python Rule Engine            |
| Risk Assessment     | Explainable Rule-Based Engine |
| Evidence Generation | OpenCV / Video Processing     |
| Dashboard           | Streamlit                     |
| AI Assistant        | Grounded Operations Assistant |
| Testing             | Pytest                        |

---

# 📁 Project Structure

```text
godrej-warehouse-ai/
│
├── app/
│   ├── assistant/
│   │   ├── __init__.py
│   │   └── assistant.py
│   │
│   ├── behaviour/
│   │   ├── drag.py
│   │   ├── drop.py
│   │   ├── engine.py
│   │   ├── push.py
│   │   ├── roll.py
│   │   ├── stacking.py
│   │   └── throw.py
│   │
│   ├── detection/
│   │   └── detector.py
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── evidence.py
│   │   └── models.py
│   │
│   ├── tracking/
│   │   └── tracker.py
│   │
│   ├── video/
│   │   ├── reader.py
│   │   └── recorder.py
│   │
│   ├── dashboard.py
│   └── main.py
│
├── scripts/
│   └── generate_evidence.py
│
├── tests/
│   ├── test_assistant.py
│   ├── test_behaviour.py
│   ├── test_detector.py
│   ├── test_evidence_generator.py
│   ├── test_risk.py
│   ├── test_tracker.py
│   └── test_video_reader.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/gauriibear/godrej-warehouse-ai.git
cd godrej-warehouse-ai
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

The Streamlit dashboard will open in your browser.

---

# ▶️ Run the Video Pipeline Directly

Example:

```bash
python app/main.py --video videos/people_walking.mp4 --conf 0.25 --max-frames 40
```

The pipeline performs:

```text
YOLO Detection
      ↓
ByteTrack Tracking
      ↓
Behaviour Analysis
      ↓
Risk Assessment
      ↓
Incident / Evidence Generation
```

---

# 🎬 Generate Demonstration Evidence

Run:

```bash
python scripts/generate_evidence.py --output-dir outputs/evidence
```

This generates deterministic incident records and supporting evidence for testing and demonstrations.

---

# 🧪 Testing

The project contains automated tests for:

* Object detection
* Object tracking
* Behaviour detection
* Temporal reasoning
* Risk scoring
* Incident evidence
* Evidence generation
* AI Operations Assistant
* Video processing

Current test status:

```text
67 / 67 tests passing
```

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

---

# 🔬 Example Incident

```text
Incident ID:
INC-DROP-1-F32

Behaviour:
Product Dropped

Object:
Carton #1

Risk Level:
CRITICAL

Risk Score:
93 / 100

Evidence:
Keyframe + video clip

Status:
Potential Damage Risk

Recommendation:
Inspect the product before further handling or dispatch.
```

---

# 🛡️ Responsible AI

The system follows a human-in-the-loop approach.

Key principles include:

* 🔍 Explainable risk scoring
* 👤 Human review of incidents
* ⚠️ Potential damage is not automatically treated as confirmed damage
* 📸 Evidence-based decision making
* 🚫 No automatic punitive decisions
* 🔐 Data minimization
* 🛡️ Secure handling of warehouse video data
* ⚖️ False-positive awareness

The AI is intended to **assist warehouse operations**, not replace human judgement.

---

# 🚧 Future Development

Planned improvements include:

* 🎨 Advanced professional operations dashboard
* 📊 Historical warehouse analytics
* 🗺️ Warehouse zone intelligence
* 🔔 Real-time alert notifications
* 🔟 Expansion to additional predefined warehouse scenarios
* 🧠 Advanced prevention and repeated-pattern insights
* 👷 Human review workflow
* 🎬 Dedicated demonstration mode
* 📈 Historical incident trends
* 🔐 Role-based access and production security

---

# 🎯 Project Vision

The ultimate goal is to create a **Field Intelligence Assistant for Warehouse Operations** that helps supervisors identify unsafe handling early, understand why an event is risky, review visual evidence, and take preventive action before product damage occurs.

The system moves warehouse monitoring from:

```text
RETROSPECTIVE CCTV
"What happened?"
        ↓
AI VIDEO INTELLIGENCE
"What is happening?"
        ↓
RISK INTELLIGENCE
"How serious is it?"
        ↓
OPERATIONS ASSISTANT
"What should we do?"
        ↓
PREVENTION
"How do we stop it happening again?"
```

---

# 👥 Project

## Godrej Warehouse AI — AI Video Intelligence for Warehouse Handling

**Objective:**
Build an AI-powered video intelligence system that supports safer, damage-free warehouse operations through automated behaviour detection, explainable risk assessment, visual evidence, and proactive operational assistance.

---

## ⭐ Key Takeaway

> **Detect early. Understand risk. Capture evidence. Assist decisions. Prevent damage.**

```
```
