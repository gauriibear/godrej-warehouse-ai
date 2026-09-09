"""
Unit and integration tests for AI Operations Assistant (app/assistant/assistant.py).
Validates grounded intent parsing, incident manifest loading, repeat tracking,
prevention guidance, and strict Godrej damage policy compliance.
"""

from pathlib import Path
import pytest

from app.assistant.assistant import OperationsAssistant, AssistantResponse, QueryIntent
from app.risk.models import DamageAssessmentStatus, IncidentRecord, RiskConfig, RiskLevel
from scripts.generate_evidence import generate_synthetic_evidence


@pytest.fixture
def sample_incidents(tmp_path):
    """Provides deterministic synthetic incidents for testing."""
    return generate_synthetic_evidence(
        output_dir=str(tmp_path),
        generate_clips=False,
        verbose=False,
    )


@pytest.fixture
def assistant(sample_incidents):
    """Provides initialized OperationsAssistant with sample incidents."""
    return OperationsAssistant.from_incidents(sample_incidents)


def test_assistant_initialization_empty():
    """Verifies assistant handles empty incidents list gracefully."""
    empty_asst = OperationsAssistant(incidents=[])
    res = empty_asst.ask("Summarize incidents")
    assert isinstance(res, AssistantResponse)
    assert "no incident records are currently loaded" in res.answer.lower()
    assert res.matched_incidents == []
    assert res.data["total_incidents"] == 0


def test_assistant_load_from_manifest(tmp_path, sample_incidents):
    """Verifies loading from JSON manifest file on disk."""
    manifest_file = tmp_path / "incidents.json"
    assert manifest_file.exists()

    asst = OperationsAssistant.from_manifest(manifest_file)
    assert len(asst.incidents) == len(sample_incidents)
    assert len(asst.incidents) >= 6

    # Test indexing
    first_id = asst.incidents[0].incident_id
    assert first_id.lower() in asst._by_id


def test_assistant_load_from_incident_records(sample_incidents):
    """Verifies direct in-memory instantiation."""
    asst = OperationsAssistant.from_incidents(sample_incidents)
    assert len(asst.incidents) == len(sample_incidents)


def test_query_summary(assistant):
    """Verifies summary intent, count calculation, and tier breakdown."""
    res = assistant.ask("Summarize the current incidents")
    assert res.intent == QueryIntent.SUMMARY
    assert "Warehouse Handling Operations Summary" in res.answer
    assert "CRITICAL Risk" in res.answer
    assert "HIGH Risk" in res.answer
    assert "MEDIUM Risk" in res.answer
    assert len(res.matched_incidents) == len(assistant.incidents)
    assert res.data["total"] == len(assistant.incidents)


def test_query_highest_risk(assistant):
    """Verifies highest risk query returns sorted top-N incidents."""
    res = assistant.ask("What are the highest-risk incidents?")
    assert res.intent == QueryIntent.HIGHEST_RISK
    assert len(res.matched_incidents) > 0

    # Ensure scores are sorted descending
    scores = [i.risk_score for i in res.matched_incidents]
    assert scores == sorted(scores, reverse=True)
    assert "Top" in res.answer
    assert "CRITICAL" in res.answer


def test_query_incident_diagnosis(assistant):
    """Verifies diagnosis query for a specific incident ID with kinematic metrics."""
    target_inc = assistant.incidents[0]
    res = assistant.ask(f"Why was {target_inc.incident_id} classified as critical?")

    assert res.intent == QueryIntent.DIAGNOSIS
    assert target_inc.incident_id in res.answer
    assert target_inc.human_readable_name in res.answer
    assert f"{target_inc.risk_score:.0f}/100" in res.answer
    assert len(res.matched_incidents) == 1
    assert res.matched_incidents[0].incident_id == target_inc.incident_id


def test_query_incident_diagnosis_not_found(assistant):
    """Verifies response when queried incident ID does not exist."""
    res = assistant.ask("Why was INC-NONEXISTENT-999 classified as critical?")
    assert res.intent == QueryIntent.DIAGNOSIS
    assert "was not found" in res.answer
    assert len(res.matched_incidents) == 0


def test_query_repeat_history_specific_track(assistant):
    """Verifies repeat history query for specific track (Carton #7 has multiple infractions)."""
    res = assistant.ask("What happened repeatedly to Carton #7?")
    assert res.intent == QueryIntent.REPEAT_HISTORY
    assert "Carton #7" in res.answer or "Track #7" in res.answer
    assert len(res.matched_incidents) >= 2

    # Verify chronic warning is present for 3+ infractions
    if len(res.matched_incidents) >= 3:
        assert "CHRONIC OFFENDER" in res.answer


def test_query_repeat_history_general(assistant):
    """Verifies general repeat violation overview."""
    res = assistant.ask("Show repeat violations across all entities")
    assert res.intent == QueryIntent.REPEAT_HISTORY
    assert "Repeat Violation Summary" in res.answer or "No repeat handling violations" in res.answer


def test_query_prevention_recommendation(assistant):
    """Verifies prevention recommendations for specific rule and general guidelines."""
    res_drop = assistant.ask("What prevention recommendation applies to drops?")
    assert res_drop.intent == QueryIntent.RECOMMENDATIONS
    assert "Product Dropped" in res_drop.answer or "dropping" in res_drop.answer.lower()
    assert "two-handed lifting" in res_drop.answer.lower() or "integrity" in res_drop.answer.lower()

    res_general = assistant.ask("What prevention recommendations apply?")
    assert res_general.intent == QueryIntent.RECOMMENDATIONS
    assert "Warehouse Handling Prevention Guidelines" in res_general.answer


def test_query_filter_by_rule(assistant):
    """Verifies filtering incidents by behaviour rule."""
    res = assistant.ask("Show all dropped cartons")
    assert res.intent == QueryIntent.FILTER_RULES
    assert all(i.rule_name == "product_dropped" for i in res.matched_incidents)


def test_query_filter_by_risk_level(assistant):
    """Verifies filtering incidents by risk level."""
    res = assistant.ask("Show all critical risk incidents")
    assert res.intent == QueryIntent.FILTER_RISK
    assert all(i.risk_level == RiskLevel.CRITICAL for i in res.matched_incidents)


def test_query_fallback_help(assistant):
    """Verifies helpful fallback guide when intent is unrecognized."""
    res = assistant.ask("Can you bake a chocolate cake?")
    assert res.intent == QueryIntent.FALLBACK
    assert "Grounded Query Guide" in res.answer
    assert "Summarize current incidents" in res.answer


def test_godrej_damage_policy_compliance(assistant):
    """Verifies that all assistant responses uphold Godrej damage distinction."""
    test_queries = [
        "Summarize incidents",
        "What are the highest-risk incidents?",
        f"Explain {assistant.incidents[0].incident_id}",
        "What happened to Carton #7?",
        "Show all critical",
    ]

    for q in test_queries:
        res = assistant.ask(q)
        # Must contain Godrej damage policy terminology
        assert "Potential Damage Risk" in res.answer
        # Must NEVER state that damage was confirmed
        assert "confirmed physical damage" not in res.answer.lower() or "is not confirmed" in res.answer.lower()
