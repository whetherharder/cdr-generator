"""Tests for assets/contact_book.py -- Zipf-distributed contact graph.

Phase 3 acceptance criteria:
- Contact book: zipf distribution, asymmetric, bias verified statistically
- Degree distribution follows Zipf (power-law) shape
- Asymmetric contacts: if A has B in contact book, B doesn't necessarily have A
- Intra-profile bias: subscribers prefer contacts from the same profile
"""

from __future__ import annotations

import numpy as np
import pytest

from cdr_generator.assets.models import Subscriber


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_subscribers(n: int, profiles: list[str] | None = None) -> list[Subscriber]:
    """Create n test subscribers with rotating profiles."""
    if profiles is None:
        profiles = ["office_worker", "heavy_traveler"]
    subs = []
    for i in range(n):
        subs.append(
            Subscriber(
                imsi=f"25001{i:010d}",
                msisdn=f"+7916{i:07d}",
                imei=f"35332508{i:06d}0",
                profile_name=profiles[i % len(profiles)],
                home_cell_id=20011 + (i % 5),
                work_cell_id=20012 + (i % 5),
                serving_ne_id="msc-01",
            )
        )
    return subs


@pytest.fixture
def subscribers_200() -> list[Subscriber]:
    return _make_subscribers(200)


@pytest.fixture
def subscribers_50() -> list[Subscriber]:
    return _make_subscribers(50)


@pytest.fixture
def contact_book_config() -> dict:
    """Standard contact book configuration dict."""
    return {
        "avg_contacts": 15,
        "degree_distribution": {
            "type": "zipf",
            "params": {"a": 2.0, "min": 3, "max": 100},
        },
        "asymmetric": True,
        "intra_profile_bias": 1.5,
        "repeat_call_probability": 0.6,
        "external_call_ratio": 0.15,
    }


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ===================================================================
# Contact book generation
# ===================================================================


class TestContactBookCreation:
    """Contact book builds a contact graph for all subscribers."""

    def test_returns_dict_mapping(
        self, subscribers_50, contact_book_config, rng
    ) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        book = build_contact_book(subscribers_50, contact_book_config, rng)
        assert isinstance(book, dict)
        assert len(book) == len(subscribers_50)

    def test_every_subscriber_has_contacts(
        self, subscribers_50, contact_book_config, rng
    ) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        book = build_contact_book(subscribers_50, contact_book_config, rng)
        for sub in subscribers_50:
            contacts = book[sub.imsi]
            assert len(contacts) >= 1, (
                f"Subscriber {sub.imsi} must have at least 1 contact"
            )

    def test_no_self_contact(self, subscribers_50, contact_book_config, rng) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        book = build_contact_book(subscribers_50, contact_book_config, rng)
        for sub in subscribers_50:
            assert sub.imsi not in book[sub.imsi], (
                f"Subscriber {sub.imsi} must not be in their own contact book"
            )

    def test_contacts_are_valid_subscribers(
        self, subscribers_50, contact_book_config, rng
    ) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        book = build_contact_book(subscribers_50, contact_book_config, rng)
        all_imsis = {s.imsi for s in subscribers_50}
        for sub in subscribers_50:
            for contact_imsi in book[sub.imsi]:
                assert contact_imsi in all_imsis, (
                    f"Contact {contact_imsi} of {sub.imsi} is not a valid subscriber"
                )


class TestContactBookDeterministic:
    def test_same_seed_same_result(self, subscribers_50, contact_book_config) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        book1 = build_contact_book(
            subscribers_50, contact_book_config, np.random.default_rng(99)
        )
        book2 = build_contact_book(
            subscribers_50, contact_book_config, np.random.default_rng(99)
        )

        for sub in subscribers_50:
            assert book1[sub.imsi] == book2[sub.imsi], (
                f"Contact book for {sub.imsi} differs between identical seeds"
            )


# ===================================================================
# Zipf degree distribution
# ===================================================================


class TestZipfDegreeDistribution:
    def test_degree_respects_min_max(
        self, subscribers_200, contact_book_config, rng
    ) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        book = build_contact_book(subscribers_200, contact_book_config, rng)
        min_d = contact_book_config["degree_distribution"]["params"]["min"]
        max_d = contact_book_config["degree_distribution"]["params"]["max"]

        for sub in subscribers_200:
            degree = len(book[sub.imsi])
            assert degree >= min_d, (
                f"Subscriber {sub.imsi} degree {degree} < min {min_d}"
            )
            effective_max = min(max_d, len(subscribers_200) - 1)
            assert degree <= effective_max, (
                f"Subscriber {sub.imsi} degree {degree} > effective_max {effective_max}"
            )

    def test_degree_distribution_has_variance(
        self, subscribers_200, contact_book_config, rng
    ) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        book = build_contact_book(subscribers_200, contact_book_config, rng)
        degrees = [len(book[s.imsi]) for s in subscribers_200]
        assert len(set(degrees)) > 1, (
            "Zipf distribution should produce multiple distinct degree values"
        )

    def test_most_subscribers_have_few_contacts(
        self, subscribers_200, contact_book_config, rng
    ) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        book = build_contact_book(subscribers_200, contact_book_config, rng)
        degrees = [len(book[s.imsi]) for s in subscribers_200]
        mean_degree = np.mean(degrees)
        below_mean = sum(1 for d in degrees if d < mean_degree)
        assert below_mean > len(degrees) * 0.4, (
            f"Expected majority below mean ({mean_degree:.1f}), got {below_mean}/{len(degrees)}"
        )


# ===================================================================
# Asymmetric contacts
# ===================================================================


class TestAsymmetricContacts:
    def test_asymmetry_exists(self, subscribers_200, contact_book_config, rng) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        contact_book_config["asymmetric"] = True
        book = build_contact_book(subscribers_200, contact_book_config, rng)

        asymmetric_count = 0
        for sub in subscribers_200:
            for contact_imsi in book[sub.imsi]:
                if sub.imsi not in book.get(contact_imsi, []):
                    asymmetric_count += 1

        assert asymmetric_count > 0, (
            "With asymmetric=True, at least some contacts should be one-directional"
        )

    def test_symmetric_mode_all_bidirectional(self, subscribers_50) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        config = {
            "degree_distribution": {
                "type": "zipf",
                "params": {"a": 2.0, "min": 3, "max": 100},
            },
            "asymmetric": False,
            "intra_profile_bias": 1.5,
        }
        book = build_contact_book(subscribers_50, config, np.random.default_rng(42))

        for sub in subscribers_50:
            for contact_imsi in book[sub.imsi]:
                assert sub.imsi in book.get(contact_imsi, []), (
                    f"Symmetric mode: {sub.imsi} has {contact_imsi} but not vice versa"
                )


# ===================================================================
# Intra-profile bias
# ===================================================================


class TestIntraProfileBias:
    def test_bias_increases_same_profile_contacts(self, subscribers_200) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        config = {
            "degree_distribution": {
                "type": "zipf",
                "params": {"a": 2.0, "min": 3, "max": 100},
            },
            "asymmetric": True,
            "intra_profile_bias": 2.0,
        }
        book = build_contact_book(subscribers_200, config, np.random.default_rng(42))

        profile_by_imsi = {s.imsi: s.profile_name for s in subscribers_200}
        same_profile_count = 0
        total_contacts = 0
        for sub in subscribers_200:
            for contact_imsi in book[sub.imsi]:
                total_contacts += 1
                if profile_by_imsi.get(contact_imsi) == sub.profile_name:
                    same_profile_count += 1

        if total_contacts == 0:
            pytest.skip("No contacts generated")

        same_profile_ratio = same_profile_count / total_contacts
        assert same_profile_ratio > 0.5, (
            f"Same-profile ratio {same_profile_ratio:.2%} should be > 50% with bias=2.0"
        )

    def test_no_bias_gives_roughly_equal_distribution(self, subscribers_200) -> None:
        from cdr_generator.assets.contact_book import build_contact_book

        config = {
            "degree_distribution": {
                "type": "zipf",
                "params": {"a": 2.0, "min": 3, "max": 100},
            },
            "asymmetric": True,
            "intra_profile_bias": 1.0,
        }
        book = build_contact_book(subscribers_200, config, np.random.default_rng(123))

        profile_by_imsi = {s.imsi: s.profile_name for s in subscribers_200}
        same_profile_count = 0
        total_contacts = 0
        for sub in subscribers_200:
            for contact_imsi in book[sub.imsi]:
                total_contacts += 1
                if profile_by_imsi.get(contact_imsi) == sub.profile_name:
                    same_profile_count += 1

        if total_contacts == 0:
            pytest.skip("No contacts generated")

        same_profile_ratio = same_profile_count / total_contacts
        assert 0.35 < same_profile_ratio < 0.65, (
            f"Same-profile ratio {same_profile_ratio:.2%} should be ~50% with bias=1.0"
        )
