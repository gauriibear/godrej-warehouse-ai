"""
Pytest integration test executing functional validation across all 10 Godrej Warehouse AI behaviours.
Validates production BehaviourEngine and RiskEngine end-to-end for:
 1. Product Dropped
 2. Product Thrown
 3. Product Dragged
 4. Product Pushed
 5. Product Rolled
 6. Unstable Stacking
 7. Pallet Overhang
 8. Product Outside Designated Area
 9. Product Handled Without Required Equipment
 10. Unsafe Loading/Unloading Sequence
"""

import pytest
from scripts.validate_10_scenarios import validate_all_10_scenarios


def test_all_10_behaviours_functional_validation():
    """
    Executes all 10 behaviour scenarios through the actual BehaviourEngine and RiskEngine.
    Asserts 100% detection rate on positive cases, 100% false-positive suppression on negative cases,
    valid risk score [0, 100], and strict compliance with Godrej Damage Policy.
    """
    results = validate_all_10_scenarios()

    assert len(results) == 10

    for r in results:
        # 1. Positive case must be detected
        assert r.detected is True, f"Scenario {r.scenario_num} ({r.name}) was not detected"
        assert r.event_id.startswith("EVT-")
        assert r.frame_idx >= 0

        # 2. Negative / safe baseline must be suppressed (0 false alarms)
        assert r.negative_suppressed is True, f"Scenario {r.scenario_num} ({r.name}) triggered false positive on safe baseline"

        # 3. Risk Engine scoring must be calibrated and bounded
        assert 0.0 <= r.risk_score <= 100.0
        assert r.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        # 4. Godrej Damage Policy Compliance
        assert r.damage_assessment == "POTENTIAL_DAMAGE_RISK"
        assert r.damage_assessment != "CONFIRMED_DAMAGE"

        # 5. Explainability and constructive prevention guidance
        assert len(r.explanation) > 15
        assert len(r.recommendation) > 15
