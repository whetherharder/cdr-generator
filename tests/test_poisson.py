"""Tests for engine/poisson.py -- Poisson sampling.

Phase 2 acceptance: event counts must follow Poisson distribution
with correct lambda parameterization.
"""

from __future__ import annotations

import numpy as np
import pytest

from cdr_generator.engine.poisson import sample_count


class TestSampleCountNonNegative:
    """Poisson samples must always be >= 0."""

    def test_zero_rate_returns_zero(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = sample_count(rate=0.0, rng=rng)
            assert n == 0, "sample_count(rate=0) must always return 0"

    def test_positive_rate_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(1000):
            n = sample_count(rate=5.0, rng=rng)
            assert n >= 0, "sample_count must never return negative"

    def test_very_small_rate(self) -> None:
        """Extremely small lambda should mostly produce 0."""
        rng = np.random.default_rng(42)
        counts = [sample_count(rate=0.001, rng=rng) for _ in range(1000)]
        assert all(c >= 0 for c in counts)
        # Most should be 0 with such a tiny rate
        zero_ratio = counts.count(0) / len(counts)
        assert zero_ratio > 0.95, (
            f"With rate=0.001, expected >95% zeros but got {zero_ratio:.1%}"
        )

    def test_returns_integer(self) -> None:
        """sample_count must return an integer type."""
        rng = np.random.default_rng(42)
        result = sample_count(rate=3.5, rng=rng)
        assert isinstance(result, (int, np.integer)), (
            f"Expected integer, got {type(result).__name__}"
        )


class TestSampleCountDistribution:
    """With large N, the mean of samples should approximate the lambda parameter."""

    def test_mean_approximates_lambda(self) -> None:
        rng = np.random.default_rng(42)
        lam = 5.0
        n_samples = 10000
        counts = [sample_count(rate=lam, rng=rng) for _ in range(n_samples)]
        mean = np.mean(counts)
        # For Poisson, mean == lambda. With N=10000 the standard error is
        # sqrt(lambda / N) ~ 0.022, so 10% tolerance is very conservative.
        assert mean == pytest.approx(lam, rel=0.1), (
            f"Mean {mean:.3f} should approximate lambda {lam} (within 10%)"
        )

    def test_variance_approximates_lambda(self) -> None:
        """For Poisson, variance == lambda."""
        rng = np.random.default_rng(42)
        lam = 8.0
        n_samples = 10000
        counts = [sample_count(rate=lam, rng=rng) for _ in range(n_samples)]
        var = np.var(counts)
        assert var == pytest.approx(lam, rel=0.15), (
            f"Variance {var:.3f} should approximate lambda {lam} (within 15%)"
        )

    def test_different_seeds_different_results(self) -> None:
        """Different RNG seeds should produce different sequences."""
        counts_a = [sample_count(rate=3.0, rng=np.random.default_rng(1)) for _ in range(100)]
        counts_b = [sample_count(rate=3.0, rng=np.random.default_rng(2)) for _ in range(100)]
        # Not all identical (extremely unlikely to be identical by chance)
        assert counts_a != counts_b

    def test_same_seed_reproducible(self) -> None:
        """Same RNG seed should produce identical sequence."""
        counts_a = [sample_count(rate=3.0, rng=np.random.default_rng(42)) for _ in range(100)]
        counts_b = [sample_count(rate=3.0, rng=np.random.default_rng(42)) for _ in range(100)]
        assert counts_a == counts_b
