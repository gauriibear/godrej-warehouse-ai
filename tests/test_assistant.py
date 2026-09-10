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


def test_query_behaviour_frequency_intent_classification(assistant):
    """
    Verifies that all specified frequency and trend query phrasings
    are accurately routed to QueryIntent.BEHAVIOUR_FREQUENCY.
    """
    queries = [
        "Which behaviour occurs most frequently?",
        "What is the most common violation?",
        "Which handling behaviour happens the most?",
        "Show behaviour frequency",
        "What behaviour is recurring most often?",
        "Which violation occurs most often?",
    ]
    for q in queries:
        res = assistant.ask(q)
        assert res.intent == QueryIntent.BEHAVIOUR_FREQUENCY, f"Failed for query: '{q}', got {res.intent}"
        assert "Behaviour frequency:" in res.answer
        assert "Total incidents analyzed:" in res.answer
        assert res.data["total_incidents"] > 0
        assert len(res.data["frequency_ranking"]) > 0


def test_query_behaviour_frequency_counts_and_ranking(assistant, sample_incidents):
    """
    Verifies that behaviour frequency counts match actual incident data,
    the top behaviour is correctly identified, and the operational interpretation is present.
    """
    res = assistant.ask("Which behaviour occurs most frequently?")
    assert res.intent == QueryIntent.BEHAVIOUR_FREQUENCY

    # Manually compute expected counts from sample_incidents
    expected_counts = {}
    for inc in sample_incidents:
        name = inc.human_readable_name
        expected_counts[name] = expected_counts.get(name, 0) + 1

    assert res.data["total_incidents"] == len(sample_incidents)
    ranking = res.data["frequency_ranking"]

    # Verify counts in ranking match expected
    for item in ranking:
        name = item["behaviour"]
        assert item["count"] == expected_counts[name]

    # Verify ranks are ordered descending by count
    counts = [item["count"] for item in ranking]
    assert counts == sorted(counts, reverse=True)

    # Verify top behaviour and interpretation
    top_behaviour = ranking[0]["behaviour"]
    top_count = ranking[0]["count"]
    assert top_count == res.data["highest_count"]
    assert top_behaviour in res.answer
    assert f"with {top_count} incident" in res.answer
    assert "most recurring observed handling issue" in res.answer


def test_query_behaviour_frequency_deterministic_ties():
    """
    Verifies that behaviour frequency handles ties deterministically
    by sorting alphabetically by behaviour name.
    """
    from app.risk.models import IncidentRecord, RiskLevel, DamageAssessmentStatus

    incidents = [
        IncidentRecord(
            incident_id="INC-1",
            event_id="EVT-1",
            rule_name="product_dropped",
            human_readable_name="Product Dropped",
            track_id=1,
            display_label="Carton #1",
            frame_idx=10,
            timestamp_seconds=1.0,
            risk_level=RiskLevel.HIGH,
            risk_score=75.0,
            confidence=0.9,
            damage_assessment=DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK,
            explanation="Drop test",
            recommendation="Handle carefully",
            metrics={},
        ),
        IncidentRecord(
            incident_id="INC-2",
            event_id="EVT-2",
            rule_name="product_dragged",
            human_readable_name="Product Dragged",
            track_id=2,
            display_label="Carton #2",
            frame_idx=20,
            timestamp_seconds=2.0,
            risk_level=RiskLevel.MEDIUM,
            risk_score=45.0,
            confidence=0.85,
            damage_assessment=DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK,
            explanation="Drag test",
            recommendation="Lift carton",
            metrics={},
        ),
    ]

    asst = OperationsAssistant.from_incidents(incidents)
    res = asst.ask("Which behaviour occurs most frequently?")

    assert res.intent == QueryIntent.BEHAVIOUR_FREQUENCY
    assert res.data["total_incidents"] == 2
    assert res.data["highest_count"] == 1

    ranking = res.data["frequency_ranking"]
    assert len(ranking) == 2
    # Both have count 1, so tie-break alphabetically: "Product Dragged" < "Product Dropped"
    assert ranking[0]["behaviour"] == "Product Dragged"
    assert ranking[1]["behaviour"] == "Product Dropped"
    assert "are the most frequently recorded behaviours with 1 incident each" in res.answer


def test_query_behaviour_frequency_empty_incidents():
    """Verifies that an empty assistant returns the required clear message."""
    empty_asst = OperationsAssistant(incidents=[])
    res = empty_asst.ask("Which behaviour occurs most frequently?")
    assert res.intent == QueryIntent.BEHAVIOUR_FREQUENCY
    assert res.answer == "No recorded behaviour incidents are available for frequency analysis."
    assert res.data["total_incidents"] == 0
    assert res.data["frequency_ranking"] == []
