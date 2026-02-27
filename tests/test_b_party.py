"""Tests for engine/b_party.py -- B-party selection logic.

Phase 3 acceptance criteria:
- B-party: ~60% from contact book, ~15% external, ~25% random
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cdr_generator.assets.models import Subscriber


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_subscribers(n: int) -> list[Subscriber]:
    """Create n test subscribers."""
    subs = []
    for i in range(n):
        subs.append(
            Subscriber(
                imsi=f"25001{i:010d}",
                msisdn=f"+7916{i:07d}",
                imei=f"35332508{i:06d}0",
                profile_name="office_worker",
                home_cell_id=20011 + (i % 5),
                work_cell_id=20012 + (i % 5),
                serving_ne_id="msc-01",
            )
        )
    return subs


@pytest.fixture
def subscribers_100() -> list[Subscriber]:
    return _make_subscribers(100)


@pytest.fixture
def contact_book_config() -> dict:
    """Contact book / B-party selection config dict."""
    return {
        "avg_contacts": 15,
        "degree_distribution": {"type": "zipf", "params": {"a": 2.0, "min": 3, "max": 100}},
        "asymmetric": True,
        "intra_profile_bias": 1.5,
        "repeat_call_probability": 0.6,
        "external_call_ratio": 0.15,
    }


@pytest.fixture
def external_numbers_config() -> dict:
    """External numbers config dict."""
    return {
        "count": 1000,
        "prefixes": [
            {"prefix": "+7495", "weight": 0.6, "label": "moscow_landline"},
            {"prefix": "+7800", "weight": 0.4, "label": "toll_free"},
        ],
    }


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ===================================================================
# B-party selection: basic functionality
# ===================================================================

class TestBPartySelection:
    """select_b_party returns a valid BPartyResult."""

    def test_returns_b_party_result(
        self, subscribers_100, contact_book_config, external_numbers_config, rng
    ) -> None:
        from cdr_generator.assets.contact_book import build_contact_book
        from cdr_generator.assets.external_numbers import generate_external_numbers
        from cdr_generator.engine.b_party import BPartyResult, select_b_party

        book = build_contact_book(subscribers_100, contact_book_config, rng)
        ext_numbers = generate_external_numbers(external_numbers_config, rng)
        a_party = subscribers_100[0]

        result = select_b_party(
            a_party, book, subscribers_100, ext_numbers, contact_book_config, rng,
        )

        assert isinstance(result, BPartyResult)
        assert result.msisdn is not None
        assert len(result.msisdn) > 0
        assert result.source in ("contact_book", "external", "random")

    def test_b_party_not_same_as_a_party(
        self, subscribers_100, contact_book_config, external_numbers_config
    ) -> None:
        from cdr_generator.assets.contact_book import build_contact_book
        from cdr_generator.assets.external_numbers import generate_external_numbers
        from cdr_generator.engine.b_party import select_b_party

        rng = np.random.default_rng(42)
        book = build_contact_book(subscribers_100, contact_book_config, rng)
        ext_numbers = generate_external_numbers(external_numbers_config, rng)

        a_party = subscribers_100[0]

        same_count = 0
        total = 500
        for i in range(total):
            iter_rng = np.random.default_rng(i)
            result = select_b_party(
                a_party, book, subscribers_100, ext_numbers, contact_book_config, iter_rng,
            )
            if result.subscriber is not None and result.subscriber.imsi == a_party.imsi:
                same_count += 1

        assert same_count / total < 0.05, (
            f"B-party was same as A-party {same_count}/{total} times"
        )


# ===================================================================
# B-party distribution: ~60% contact, ~15% external, ~25% random
# ===================================================================

class TestBPartyDistribution:
    """Verify the statistical distribution of B-party selection sources."""

    def test_distribution_approximately_correct(
        self, subscribers_100, contact_book_config, external_numbers_config
    ) -> None:
        """Over many selections, source ratios should match expected distribution."""
        from cdr_generator.assets.contact_book import build_contact_book
        from cdr_generator.assets.external_numbers import generate_external_numbers
        from cdr_generator.engine.b_party import select_b_party

        rng = np.random.default_rng(77)
        book = build_contact_book(subscribers_100, contact_book_config, rng)
        ext_numbers = generate_external_numbers(external_numbers_config, rng)

        a_party = subscribers_100[0]

        sources = Counter()
        n_trials = 2000
        for i in range(n_trials):
            iter_rng = np.random.default_rng(i + 5000)
            result = select_b_party(
                a_party, book, subscribers_100, ext_numbers, contact_book_config, iter_rng,
            )
            sources[result.source] += 1

        contact_ratio = sources["contact_book"] / n_trials
        external_ratio = sources["external"] / n_trials
        random_ratio = sources["random"] / n_trials

        # Expected: ~60% contact, ~15% external, ~25% random
        # Tolerance: generous for statistical tests
        assert 0.35 < contact_ratio < 0.85, (
            f"Contact ratio {contact_ratio:.2%} outside [35%, 85%]. "
            f"Expected ~60%. Sources: {dict(sources)}"
        )
        assert 0.03 < external_ratio < 0.35, (
            f"External ratio {external_ratio:.2%} outside [3%, 35%]. "
            f"Expected ~15%. Sources: {dict(sources)}"
        )
        assert 0.05 < random_ratio < 0.50, (
            f"Random ratio {random_ratio:.2%} outside [5%, 50%]. "
            f"Expected ~25%. Sources: {dict(sources)}"
        )


# ===================================================================
# External number pool
# ===================================================================

class TestExternalNumberPool:
    """External number pool generation produces valid phone numbers."""

    def test_pool_has_correct_count(
        self, external_numbers_config, rng
    ) -> None:
        from cdr_generator.assets.external_numbers import generate_external_numbers

        pool = generate_external_numbers(external_numbers_config, rng)
        assert len(pool) == external_numbers_config["count"]

    def test_numbers_have_configured_prefixes(
        self, external_numbers_config, rng
    ) -> None:
        from cdr_generator.assets.external_numbers import generate_external_numbers

        pool = generate_external_numbers(external_numbers_config, rng)
        valid_prefixes = [p["prefix"] for p in external_numbers_config["prefixes"]]

        for ext_num in pool:
            assert any(ext_num.msisdn.startswith(p) for p in valid_prefixes), (
                f"External number {ext_num.msisdn} doesn't start with any configured prefix"
            )

    def test_pick_returns_valid_external_number(
        self, external_numbers_config, rng
    ) -> None:
        from cdr_generator.assets.external_numbers import generate_external_numbers

        pool = generate_external_numbers(external_numbers_config, rng)
        assert len(pool) > 0
        # Each element is an ExternalNumber with .msisdn
        assert hasattr(pool[0], "msisdn")
        assert pool[0].imsi == "external"

    def test_prefix_weights_affect_distribution(
        self, external_numbers_config, rng
    ) -> None:
        """Prefix weights should influence the distribution of generated numbers."""
        from cdr_generator.assets.external_numbers import generate_external_numbers

        pool = generate_external_numbers(external_numbers_config, rng)
        prefix_counts = Counter()
        for ext_num in pool:
            for p in external_numbers_config["prefixes"]:
                if ext_num.msisdn.startswith(p["prefix"]):
                    prefix_counts[p["prefix"]] += 1
                    break

        # With weights 0.6 and 0.4, +7495 should have more numbers
        assert prefix_counts["+7495"] > prefix_counts["+7800"], (
            f"Expected +7495 (weight=0.6) > +7800 (weight=0.4), "
            f"got {prefix_counts['+7495']} vs {prefix_counts['+7800']}"
        )

    def test_deterministic_generation(self, external_numbers_config) -> None:
        from cdr_generator.assets.external_numbers import generate_external_numbers

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        pool1 = generate_external_numbers(external_numbers_config, rng1)
        pool2 = generate_external_numbers(external_numbers_config, rng2)

        msisdns1 = [n.msisdn for n in pool1]
        msisdns2 = [n.msisdn for n in pool2]
        assert msisdns1 == msisdns2, (
            "External numbers should be deterministic with same seed"
        )
