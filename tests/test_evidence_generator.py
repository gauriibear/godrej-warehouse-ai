"""
Unit and integration tests for Stage 5 Evidence-Generation Utility (scripts/generate_evidence.py).
Validates deterministic scenario generation, risk coverage, repeat escalation,
file artifact generation on disk, and damage policy compliance.
"""

from pathlib import Path
import json
import subprocess
import sys
import pytest

from app.risk.models import DamageAssessmentStatus, IncidentRecord, RiskLevel
from scripts.generate_evidence import generate_synthetic_evidence


def test_generate_synthetic_evidence_output_types(tmp_path):
    """Verifies that generator returns a list of valid IncidentRecord dataclasses."""
    incidents = generate_synthetic_evidence(output_dir=str(tmp_path), generate_clips=False, verbose=False)
    assert isinstance(incidents, list)
    assert len(incidents) >= 6
    for inc in incidents:
        assert isinstance(inc, IncidentRecord)
        assert inc.incident_id.startswith("INC-")
        assert 0.0 <= inc.risk_score <= 100.0


def test_coverage_of_required_behaviours(tmp_path):
    """Verifies that generated scenarios cover all required handling behaviours."""
    incidents = generate_synthetic_evidence(output_dir=str(tmp_path), generate_clips=False, verbose=False)
    rules_present = {inc.rule_name for inc in incidents}

    assert "product_dropped" in rules_present
    assert "product_dragged" in rules_present
    assert "unstable_stacking" in rules_present
    assert "product_pushed" in rules_present


def test_coverage_of_risk_levels(tmp_path):
    """Verifies that generated dataset includes MEDIUM, HIGH, and CRITICAL risk levels."""
    incidents = generate_synthetic_evidence(output_dir=str(tmp_path), generate_clips=False, verbose=False)
    levels_present = {inc.risk_level for inc in incidents}

    assert RiskLevel.MEDIUM in levels_present
    assert RiskLevel.HIGH in levels_present
    assert RiskLevel.CRITICAL in levels_present


def test_repeat_violation_escalates_score(tmp_path):
    """Verifies that repeated violations on the same track escalate risk score to CRITICAL."""
    incidents = generate_synthetic_evidence(output_dir=str(tmp_path), generate_clips=False, verbose=False)
    track_7_incidents = [i for i in incidents if i.track_id == 7]

    assert len(track_7_incidents) == 3
    # Scores must strictly increase due to repeat infraction penalties
    assert track_7_incidents[1].risk_score > track_7_incidents[0].risk_score
    assert track_7_incidents[2].risk_score > track_7_incidents[1].risk_score
    # 3rd infraction reaches CRITICAL
    assert track_7_incidents[2].risk_level == RiskLevel.CRITICAL
    assert "repeat" in track_7_incidents[2].explanation.lower()


def test_evidence_files_created_on_disk(tmp_path):
    """Verifies keyframe snapshots, video clips, and manifest are created on disk."""
    incidents = generate_synthetic_evidence(output_dir=str(tmp_path), generate_clips=True, verbose=False)

    manifest_file = tmp_path / "incidents.json"
    assert manifest_file.exists()
    assert manifest_file.stat().st_size > 0

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == len(incidents)

    for inc in incidents:
        # Check keyframe snapshot
        assert inc.evidence_image_path is not None
        assert Path(inc.evidence_image_path).exists()
        assert Path(inc.evidence_image_path).stat().st_size > 0

        # Check video clip
        assert inc.video_clip_path is not None
        assert Path(inc.video_clip_path).exists()
        assert Path(inc.video_clip_path).stat().st_size > 0


def test_damage_assessment_status_is_potential_risk(tmp_path):
    """
    Verifies strict compliance with Godrej Damage Policy:
    Observed Behaviour -> Potential Damage Risk -> Confirmed Damage.
    Every incident must be POTENTIAL_DAMAGE_RISK and never CONFIRMED_DAMAGE.
    """
    incidents = generate_synthetic_evidence(output_dir=str(tmp_path), generate_clips=False, verbose=False)
    for inc in incidents:
        assert inc.damage_assessment == DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK
        assert inc.damage_assessment != DamageAssessmentStatus.CONFIRMED_DAMAGE
        assert "Potential Damage Risk" in inc.to_dict()["damage_assessment_label"]


def test_cli_invocation(tmp_path):
    """Verifies that the generator can be invoked cleanly from the CLI."""
    cli_out = tmp_path / "cli_evidence"
    cmd = [
        sys.executable,
        "scripts/generate_evidence.py",
        "--output-dir",
        str(cli_out),
        "--no-clips",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "STAGE 5 SYNTHETIC EVIDENCE GENERATOR" in result.stdout
    assert (cli_out / "incidents.json").exists()
