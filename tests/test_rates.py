"""Tests for engine/rates.py -- effective rate calculation.

Phase 2 acceptance: hourly distribution must visually match weights,
day-of-week multipliers must apply correctly.
"""

from __future__ import annotations

import pytest

from cdr_generator.engine.rates import effective_rate


class TestEffectiveRateNormalized:
    """Sum of hourly_weights should not affect the absolute result --
    only the *shape* matters because weights are normalized internally."""

    def test_uniform_weights_sum_to_one(self) -> None:
        """Weights [1]*24 and [2]*24 must produce the same rate."""
        hourly_w_1 = [1.0] * 24
        hourly_w_2 = [2.0] * 24
        dow_mult = [1.0] * 7

        r1 = effective_rate(
            base_lambda=3.5,
            hourly_weights=hourly_w_1,
            dow_multipliers=dow_mult,
            hour=10,
            dow=0,  # Monday
            time_step_seconds=60,
        )
        r2 = effective_rate(
            base_lambda=3.5,
            hourly_weights=hourly_w_2,
            dow_multipliers=dow_mult,
            hour=10,
            dow=0,
            time_step_seconds=60,
        )
        assert r1 == pytest.approx(r2, rel=1e-6), (
            "Scaling all weights by a constant must not change the effective rate"
        )

    def test_non_uniform_weights_normalized(self) -> None:
        """Multiplying all weights by 10x must not change the result."""
        hourly_w = [
            0.01,
            0.01,
            0.01,
            0.01,
            0.02,
            0.03,
            0.05,
            0.08,
            0.10,
            0.12,
            0.11,
            0.10,
            0.08,
            0.09,
            0.10,
            0.10,
            0.08,
            0.06,
            0.05,
            0.04,
            0.03,
            0.02,
            0.01,
            0.01,
        ]
        hourly_w_10x = [w * 10 for w in hourly_w]
        dow_mult = [1.0] * 7

        r1 = effective_rate(
            base_lambda=5.0,
            hourly_weights=hourly_w,
            dow_multipliers=dow_mult,
            hour=9,
            dow=2,
            time_step_seconds=60,
        )
        r2 = effective_rate(
            base_lambda=5.0,
            hourly_weights=hourly_w_10x,
            dow_multipliers=dow_mult,
            hour=9,
            dow=2,
            time_step_seconds=60,
        )
        assert r1 == pytest.approx(r2, rel=1e-6)


class TestHourlyWeightApplied:
    """Rate must be higher at peak hours than at off-peak hours."""

    def test_peak_vs_offpeak(self) -> None:
        """Hour with weight 0.12 must produce higher rate than hour with weight 0.01."""
        hourly_w = [
            0.01,
            0.01,
            0.01,
            0.01,
            0.02,
            0.03,
            0.05,
            0.08,
            0.10,
            0.12,
            0.11,
            0.10,
            0.08,
            0.09,
            0.10,
            0.10,
            0.08,
            0.06,
            0.05,
            0.04,
            0.03,
            0.02,
            0.01,
            0.01,
        ]
        dow_mult = [1.0] * 7

        rate_peak = effective_rate(
            base_lambda=3.5,
            hourly_weights=hourly_w,
            dow_multipliers=dow_mult,
            hour=9,  # weight 0.12
            dow=0,
            time_step_seconds=60,
        )
        rate_off = effective_rate(
            base_lambda=3.5,
            hourly_weights=hourly_w,
            dow_multipliers=dow_mult,
            hour=0,  # weight 0.01
            dow=0,
            time_step_seconds=60,
        )
        assert rate_peak > rate_off, (
            f"Peak-hour rate ({rate_peak}) must exceed off-peak rate ({rate_off})"
        )

    def test_proportional_to_weight(self) -> None:
        """Rate ratio between two hours should match their weight ratio."""
        hourly_w = [0.0] * 24
        hourly_w[6] = 1.0  # hour 6 gets weight 1
        hourly_w[18] = 3.0  # hour 18 gets weight 3
        dow_mult = [1.0] * 7

        r6 = effective_rate(
            base_lambda=10.0,
            hourly_weights=hourly_w,
            dow_multipliers=dow_mult,
            hour=6,
            dow=0,
            time_step_seconds=60,
        )
        r18 = effective_rate(
            base_lambda=10.0,
            hourly_weights=hourly_w,
            dow_multipliers=dow_mult,
            hour=18,
            dow=0,
            time_step_seconds=60,
        )
        assert r18 == pytest.approx(r6 * 3.0, rel=1e-6), (
            "Rate must be proportional to the hourly weight"
        )


class TestDowMultiplierApplied:
    """Rate must be lower on weekend days when multiplier < 1."""

    def test_weekend_lower_than_weekday(self) -> None:
        """Sunday (dow=6) with multiplier 0.3 vs Monday (dow=0) with 1.0."""
        hourly_w = [1.0] * 24
        dow_mult = [1.0, 1.0, 1.0, 1.0, 0.9, 0.4, 0.3]

        rate_mon = effective_rate(
            base_lambda=3.5,
            hourly_weights=hourly_w,
            dow_multipliers=dow_mult,
            hour=12,
            dow=0,  # Monday
            time_step_seconds=60,
        )
        rate_sun = effective_rate(
            base_lambda=3.5,
            hourly_weights=hourly_w,
            dow_multipliers=dow_mult,
            hour=12,
            dow=6,  # Sunday
            time_step_seconds=60,
        )
        assert rate_sun < rate_mon, (
            f"Sunday rate ({rate_sun}) must be lower than Monday rate ({rate_mon})"
        )
        assert rate_sun == pytest.approx(rate_mon * 0.3, rel=1e-6), (
            "Sunday rate must equal Monday rate * 0.3"
        )

    def test_multiplier_of_zero_gives_zero(self) -> None:
        """A dow multiplier of 0.0 must yield rate 0."""
        hourly_w = [1.0] * 24
        dow_mult = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        rate = effective_rate(
            base_lambda=3.5,
            hourly_weights=hourly_w,
            dow_multipliers=dow_mult,
            hour=12,
            dow=0,  # Monday, multiplier = 0.0
            time_step_seconds=60,
        )
        assert rate == 0.0
