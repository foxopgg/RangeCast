"""
Unit tests for RangeCast scoring.py
"""
import pytest
import numpy as np
import pandas as pd
from scoring import (
    interval_score,
    weighted_interval_score,
    calculate_coverage,
    assign_trust_badge,
    compute_scorecard_summary,
)


def test_interval_score_exact_coverage():
    # If y is inside [lower, upper], score is simply upper - lower (sharpness)
    y = 50.0
    lower = 40.0
    upper = 60.0
    alpha = 0.20
    score = interval_score(y, lower, upper, alpha)
    assert score == pytest.approx(20.0)


def test_interval_score_underprediction():
    # y < lower: penalty is (2/alpha) * (lower - y)
    y = 30.0
    lower = 40.0
    upper = 60.0
    alpha = 0.20
    # Sharpness = 20
    # Penalty = (2 / 0.20) * (40 - 30) = 10 * 10 = 100
    # Total = 120
    score = interval_score(y, lower, upper, alpha)
    assert score == pytest.approx(120.0)


def test_interval_score_overprediction():
    # y > upper: penalty is (2/alpha) * (y - upper)
    y = 75.0
    lower = 40.0
    upper = 60.0
    alpha = 0.50
    # Sharpness = 20
    # Penalty = (2 / 0.50) * (75 - 60) = 4 * 15 = 60
    # Total = 80
    score = interval_score(y, lower, upper, alpha)
    assert score == pytest.approx(80.0)


def test_weighted_interval_score_properties():
    # Perfect median and symmetric intervals
    y = 100.0
    q10, q25, q50, q75, q90 = 80.0, 90.0, 100.0, 110.0, 120.0
    wis = weighted_interval_score(y, q10, q25, q50, q75, q90)
    
    # med_error = 0
    # is_50 = 110 - 90 = 20
    # is_80 = 120 - 80 = 40
    # wis = (0.50 * 0 + 0.25 * 20 + 0.10 * 40) / 2.5 = (5 + 4) / 2.5 = 9 / 2.5 = 3.6
    assert wis == pytest.approx(3.6)


def test_calculate_coverage():
    y = np.array([10, 20, 30, 40, 50])
    l = np.array([5, 15, 25, 35, 60])   # 50 is outside (50 < 60)
    u = np.array([15, 25, 35, 45, 70])
    cov = calculate_coverage(y, l, u)
    assert cov == pytest.approx(0.80)


def test_assign_trust_badge():
    # High trust
    badge_high = assign_trust_badge(
        coverage_80=0.82,
        coverage_50=0.52,
        relative_wis=0.72,
        recent_shift_alert=False,
        low_volume_warning=False,
        eval_periods=10,
    )
    assert badge_high["tier"] == "HIGH"
    
    # Low trust due to alert
    badge_alert = assign_trust_badge(
        coverage_80=0.82,
        coverage_50=0.52,
        relative_wis=0.72,
        recent_shift_alert=True,
        low_volume_warning=False,
        eval_periods=10,
    )
    assert badge_alert["tier"] == "LOW"
    
    # Low trust due to undercoverage
    badge_low_cov = assign_trust_badge(
        coverage_80=0.45,
        coverage_50=0.25,
        relative_wis=0.80,
        recent_shift_alert=False,
        low_volume_warning=False,
        eval_periods=10,
    )
    assert badge_low_cov["tier"] == "LOW"
    print("All scoring tests passed successfully!")


if __name__ == "__main__":
    test_interval_score_exact_coverage()
    test_interval_score_underprediction()
    test_interval_score_overprediction()
    test_weighted_interval_score_properties()
    test_calculate_coverage()
    test_assign_trust_badge()
    print("All test functions executed without errors!")
