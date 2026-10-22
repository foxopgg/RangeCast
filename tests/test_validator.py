"""
Unit Tests: Groq Number Validator (tests/test_validator.py)
===========================================================
Tests the validate_response_numbers() function in groq_client.py.
Verifies mathematical correctness of the number extractor and
that valid / invalid Groq responses are classified correctly.

Run with:
    PYTHONPATH=. pytest tests/test_validator.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from groq_client import (
    _extract_numbers,
    _build_allowed_set,
    _is_number_allowed,
    validate_response_numbers,
)


# ---------------------------------------------------------------------------
# Tests for _extract_numbers
# ---------------------------------------------------------------------------
class TestExtractNumbers:
    def test_integer(self):
        assert _extract_numbers("There are 150 cases.") == [150.0]

    def test_float(self):
        assert _extract_numbers("Coverage is 82.5 percent.") == [82.5]

    def test_comma_separated(self):
        assert _extract_numbers("Order 1,234 kits.") == [1234.0]

    def test_percentage_stripped(self):
        nums = _extract_numbers("80% coverage achieved.")
        assert 80.0 in nums

    def test_multiple_numbers(self):
        nums = _extract_numbers("Range: 100-200, median 145.")
        assert 100.0 in nums
        assert 200.0 in nums
        assert 145.0 in nums

    def test_no_numbers(self):
        assert _extract_numbers("No numbers here at all.") == []

    def test_zero(self):
        assert _extract_numbers("Zero cases: 0.") == [0.0]

    def test_large_comma_number(self):
        nums = _extract_numbers("12,500 cylinders required.")
        assert 12500.0 in nums


# ---------------------------------------------------------------------------
# Tests for _build_allowed_set
# ---------------------------------------------------------------------------
class TestBuildAllowedSet:
    def test_flat_dict(self):
        ctx = {"q50": 100, "q90": 200, "coverage": 0.83}
        allowed = _build_allowed_set(ctx)
        assert 100.0 in allowed
        assert 200.0 in allowed
        assert 0.83 in allowed

    def test_nested_dict(self):
        ctx = {"forecast": {"q10": 50, "q90": 150}, "trust": "HIGH"}
        allowed = _build_allowed_set(ctx)
        assert 50.0 in allowed
        assert 150.0 in allowed

    def test_list_values(self):
        ctx = {"quantiles": [10, 25, 50, 75, 90]}
        allowed = _build_allowed_set(ctx)
        for v in [10.0, 25.0, 50.0, 75.0, 90.0]:
            assert v in allowed

    def test_string_values_excluded(self):
        ctx = {"district": "Chennai", "trust": "HIGH"}
        allowed = _build_allowed_set(ctx)
        # Strings should not appear as numbers
        assert len(allowed) == 0

    def test_mixed_types(self):
        ctx = {"beds": 42, "rate": 0.065, "name": "test", "flag": True}
        allowed = _build_allowed_set(ctx)
        assert 42.0 in allowed
        assert 0.065 in allowed


# ---------------------------------------------------------------------------
# Tests for _is_number_allowed
# ---------------------------------------------------------------------------
class TestIsNumberAllowed:
    def test_exact_match(self):
        assert _is_number_allowed(150.0, [150.0, 200.0])

    def test_rounding_within_tolerance(self):
        # 1001 is within 2% of 1000
        assert _is_number_allowed(1001.0, [1000.0])

    def test_rounding_outside_tolerance(self):
        # 1025 is more than 2% from 1000
        assert not _is_number_allowed(1025.0, [1000.0])

    def test_small_ordinal_always_allowed(self):
        # Small whole numbers 0-10 are always allowed
        for n in range(11):
            assert _is_number_allowed(float(n), [])

    def test_not_in_allowed(self):
        assert not _is_number_allowed(999.0, [100.0, 200.0, 300.0])

    def test_zero_handling(self):
        assert _is_number_allowed(0.0, [0.0])
        assert _is_number_allowed(0.3, [0.0])  # Within 0.5 of zero


# ---------------------------------------------------------------------------
# Tests for validate_response_numbers (integration)
# ---------------------------------------------------------------------------
class TestValidateResponseNumbers:
    BASE_CONTEXT = {
        "q10_week1": 120,
        "q50_week1": 250,
        "q90_week1": 380,
        "coverage_80_pct": 82.5,
        "relative_wis": 0.78,
        "peak_beds_required": 45,
        "trust_tier": "HIGH",
        "district": "Chennai",
        "disease": "COVID-19",
    }

    def test_valid_response_passes(self):
        response = (
            "Next week's range is 120 to 380 cases (median 250). "
            "Historical 82.5% coverage. Trust: HIGH. "
            "Peak beds: 45. rWIS: 0.78."
        )
        is_valid, bad = validate_response_numbers(response, self.BASE_CONTEXT)
        assert is_valid, f"Should pass, got bad numbers: {bad}"
        assert bad == []

    def test_hallucinated_number_fails(self):
        # 9999 is not in context
        response = "We project 9999 cases next week based on our analysis."
        is_valid, bad = validate_response_numbers(response, self.BASE_CONTEXT)
        assert not is_valid
        assert 9999.0 in bad

    def test_rounded_number_passes(self):
        # 251 is within 2% of 250
        response = "The median forecast is approximately 251 cases."
        is_valid, bad = validate_response_numbers(response, self.BASE_CONTEXT)
        assert is_valid, f"Rounding should be allowed, got bad: {bad}"

    def test_empty_response_passes(self):
        is_valid, bad = validate_response_numbers("No numeric data available.", self.BASE_CONTEXT)
        assert is_valid

    def test_comma_formatted_number_passes(self):
        # 82.5% formatted as text with comma elsewhere
        response = "The 80% range held true 82.5% of the time."
        is_valid, bad = validate_response_numbers(response, self.BASE_CONTEXT)
        assert is_valid, f"Should pass, got bad numbers: {bad}"

    def test_multiple_hallucinated_numbers_caught(self):
        response = "I project 5000 cases and 3000 ICU admissions."
        is_valid, bad = validate_response_numbers(response, self.BASE_CONTEXT)
        assert not is_valid
        assert len(bad) >= 1

    def test_ordinal_numbers_in_prose_allowed(self):
        # "1 week", "2 horizons" etc. should always be allowed
        response = "In 1 week we expect 250 cases across 2 zones."
        is_valid, bad = validate_response_numbers(response, self.BASE_CONTEXT)
        assert is_valid, f"Ordinals should be allowed, got bad: {bad}"

    def test_empty_context_blocks_all_numbers(self):
        # Empty context: only ordinals 0-10 should pass
        response = "We project 500 cases."
        is_valid, bad = validate_response_numbers(response, {})
        assert not is_valid
        assert 500.0 in bad

    def test_percentage_hallucination_caught(self):
        # 97.3% is not in context and not a standard interval level
        response = "Our accuracy is 97.3% this week."
        is_valid, bad = validate_response_numbers(response, self.BASE_CONTEXT)
        assert not is_valid
        assert 97.3 in bad

    def test_nested_context_numbers_allowed(self):
        ctx = {
            "district": "Madurai",
            "forecast": {"q10": 80, "q90": 300},
            "plan": {"beds": 120, "kits": 1800},
        }
        response = "Range: 80 to 300 cases. Beds needed: 120. Kits: 1800."
        is_valid, bad = validate_response_numbers(response, ctx)
        assert is_valid, f"Should pass, got bad: {bad}"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_very_large_number_not_in_context(self):
        ctx = {"cases": 50}
        is_valid, bad = validate_response_numbers("Order 1000000 cylinders!", ctx)
        assert not is_valid

    def test_float_precision(self):
        ctx = {"relative_wis": 0.782}
        # 0.782 → rounds to 0.78 in output — within 2%
        is_valid, bad = validate_response_numbers("rWIS is 0.78.", ctx)
        assert is_valid, f"0.78 should match 0.782 within tolerance, got bad: {bad}"

    def test_response_with_no_numbers_and_empty_context(self):
        is_valid, bad = validate_response_numbers("Data is unavailable.", {})
        assert is_valid

    def test_year_in_text(self):
        # 2021 is not in typical context — should be caught
        ctx = {"cases": 100}
        is_valid, bad = validate_response_numbers("During 2021, cases rose to 100.", ctx)
        # 2021 is > 10 and not in context → should fail
        assert not is_valid
