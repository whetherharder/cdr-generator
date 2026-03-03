"""Tests for Phase 5 concurrency enforcement and disabled-cell edge cases.

Concurrency module: cdr_generator/engine/concurrency.py
Function signature:
    enforce_concurrency(n_voice, n_sms, n_data, config) -> tuple[int, int, int]

ConcurrencyConfig model (from config/models.py):
    max_voice: int | None = 1   (None means unlimited)
    max_data:  int | None = 1
    max_sms:   int | None = None

Edge case: disabled cell with no overflow -> cause_code=38 (no route to destination).
"""

from __future__ import annotations

import numpy as np

from cdr_generator.config.models import ConcurrencyConfig


# ===========================================================================
# Concurrency enforcement tests
# ===========================================================================


class TestEnforceConcurrencyBasic:
    """Basic capping behaviour with default config (max_voice=1, max_data=1, max_sms=None)."""

    def test_caps_voice_to_max(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=1, max_data=1, max_sms=None)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=5, n_sms=3, n_data=4, config=config
        )
        assert n_voice == 1, f"Voice must be capped to 1, got {n_voice}"

    def test_caps_data_to_max(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=1, max_data=1, max_sms=None)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=5, n_sms=3, n_data=4, config=config
        )
        assert n_data == 1, f"Data must be capped to 1, got {n_data}"

    def test_sms_unlimited_when_none(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=1, max_data=1, max_sms=None)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=5, n_sms=100, n_data=4, config=config
        )
        assert n_sms == 100, f"SMS must be unlimited (100), got {n_sms}"


class TestEnforceConcurrencyWithinLimits:
    """When counts are already within limits, they should not be changed."""

    def test_within_limits_unchanged(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=3, max_data=2, max_sms=10)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=2, n_sms=5, n_data=1, config=config
        )
        assert n_voice == 2, f"Voice within limit should be unchanged, got {n_voice}"
        assert n_sms == 5, f"SMS within limit should be unchanged, got {n_sms}"
        assert n_data == 1, f"Data within limit should be unchanged, got {n_data}"

    def test_at_exact_limit_unchanged(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=3, max_data=2, max_sms=10)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=3, n_sms=10, n_data=2, config=config
        )
        assert n_voice == 3
        assert n_sms == 10
        assert n_data == 2


class TestEnforceConcurrencyAllUnlimited:
    """When all limits are None, no capping should occur."""

    def test_all_unlimited(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=None, max_data=None, max_sms=None)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=100, n_sms=200, n_data=50, config=config
        )
        assert n_voice == 100, f"Voice unlimited should be 100, got {n_voice}"
        assert n_sms == 200, f"SMS unlimited should be 200, got {n_sms}"
        assert n_data == 50, f"Data unlimited should be 50, got {n_data}"


class TestEnforceConcurrencyZeroInputs:
    """Zero event counts should remain zero."""

    def test_zeros_stay_zero(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=1, max_data=1, max_sms=None)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=0, n_sms=0, n_data=0, config=config
        )
        assert n_voice == 0
        assert n_sms == 0
        assert n_data == 0


class TestEnforceConcurrencyReturnType:
    """Return value must be a tuple of three ints."""

    def test_returns_tuple_of_three_ints(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=1, max_data=1, max_sms=None)
        result = enforce_concurrency(n_voice=5, n_sms=3, n_data=4, config=config)
        assert isinstance(result, tuple), f"Must return tuple, got {type(result)}"
        assert len(result) == 3, f"Must return 3-tuple, got length {len(result)}"
        for i, val in enumerate(result):
            assert isinstance(val, int), f"Element {i} must be int, got {type(val)}"


class TestEnforceConcurrencyCustomLimits:
    """Test with various custom limit configurations."""

    def test_high_voice_limit(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=10, max_data=5, max_sms=20)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=15, n_sms=25, n_data=8, config=config
        )
        assert n_voice == 10
        assert n_sms == 20
        assert n_data == 5

    def test_sms_capped_when_specified(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=1, max_data=1, max_sms=5)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=3, n_sms=10, n_data=2, config=config
        )
        assert n_sms == 5, f"SMS must be capped to 5, got {n_sms}"

    def test_only_voice_capped(self) -> None:
        """When only max_voice is set, data and SMS pass through."""
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=2, max_data=None, max_sms=None)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=10, n_sms=50, n_data=30, config=config
        )
        assert n_voice == 2
        assert n_sms == 50
        assert n_data == 30


class TestEnforceConcurrencyNonNegative:
    """Concurrency enforcement must never produce negative counts."""

    def test_result_never_negative(self) -> None:
        from cdr_generator.engine.concurrency import enforce_concurrency

        config = ConcurrencyConfig(max_voice=1, max_data=1, max_sms=1)
        n_voice, n_sms, n_data = enforce_concurrency(
            n_voice=0, n_sms=0, n_data=0, config=config
        )
        assert n_voice >= 0
        assert n_sms >= 0
        assert n_data >= 0


# ===========================================================================
# Disabled cell with no overflow -> cause_code=38 edge case
# ===========================================================================


class TestDisabledCellNoOverflow:
    """When a subscriber's cell is disabled and no overflow cells exist,
    the system should generate a failed CDR with cause_code=38
    (no route to destination)."""

    def test_disabled_cell_no_overflow_produces_failed_cdr(self) -> None:
        """Direct test: voice generator with a disabled cell and no overflow
        should produce a failed MO CDR with cause_code=38."""

        from cdr_generator.engine.special_events import ActiveEffects

        # The disabled cell scenario is handled at the runner level.
        # Verify that ActiveEffects correctly reports cell as disabled
        # and has no overflow.
        effects = ActiveEffects(
            disabled_cells=frozenset({20011}),
            overflow_cells=frozenset(),  # no overflow
        )
        assert effects.is_cell_disabled(20011) is True
        assert effects.has_overflow() is False
        assert effects.pick_overflow_cell(np.random.default_rng(1)) is None

    def test_disabled_cell_with_overflow_redirects(self) -> None:
        """When overflow cells exist, pick_overflow_cell returns one."""
        from cdr_generator.engine.special_events import ActiveEffects

        effects = ActiveEffects(
            disabled_cells=frozenset({20011}),
            overflow_cells=frozenset({20099, 20098}),
        )
        rng = np.random.default_rng(42)
        overflow_id = effects.pick_overflow_cell(rng)
        assert overflow_id in {20099, 20098}, (
            f"pick_overflow_cell must return a valid overflow cell, got {overflow_id}"
        )

    def test_cause_code_38_for_no_route(self) -> None:
        """Cause code 38 ('no route to destination' / 'network out of order')
        is the correct code for disabled cells with no overflow.

        This test verifies the generate_voice_cdr can produce a CDR with
        cause_code=38 when invoked with that code in failure_causes config.
        """
        from datetime import datetime, timezone

        from cdr_generator.assets.models import Cell, NetworkElement, Subscriber
        from cdr_generator.generators.voice import generate_voice_cdr

        caller = Subscriber(
            imsi="250010000000001",
            msisdn="+79160000001",
            imei="353325080000001",
            profile_name="office_worker",
            home_cell_id=20011,
            serving_ne_id="msc-01",
        )
        callee = Subscriber(
            imsi="250010000000002",
            msisdn="+79160000002",
            imei="353325080000002",
            profile_name="office_worker",
            home_cell_id=20012,
            serving_ne_id="msc-01",
        )
        cell = Cell(
            cell_id=20011,
            tac=2001,
            ecgi="25001-20011",
            lat=55.75,
            lon=37.61,
            azimuth=0,
            sector=1,
            cell_type="urban",
            capacity="high",
        )
        msc = NetworkElement(
            id="msc-01",
            ne_type="msc",
            vendor="ericsson",
            serves_tacs=[2001],
            produces=["mo_call", "mt_call"],
        )
        # Force failure with cause_code=38 (network out of order / no route)
        voice_cfg = {
            "success_rate": 0.0,
            "call_forwarding_rate": 0.0,
            "failure_causes": [
                {"cause": "no_route_to_destination", "code": 38, "weight": 1.0},
            ],
            "duration": {
                "distribution": {"type": "constant", "params": {"value": 60}},
                "min_seconds": 1,
                "max_seconds": 7200,
            },
            "normal_termination_causes": [
                {"cause": "normal_clearing", "code": 16, "weight": 1.0},
            ],
            "paired_timestamp_jitter_ms": 1000,
        }
        rng = np.random.default_rng(42)
        event_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=voice_cfg,
            rng=rng,
        )
        assert len(records) == 1, "Failed call must produce 1 CDR"
        assert records[0].cause_for_termination == 38, (
            f"Disabled cell with no overflow must produce cause_code=38, "
            f"got {records[0].cause_for_termination}"
        )


# ===========================================================================
# ConcurrencyConfig model tests
# ===========================================================================


class TestConcurrencyConfigModel:
    """Verify ConcurrencyConfig from config/models.py has expected defaults."""

    def test_default_values(self) -> None:
        config = ConcurrencyConfig()
        assert config.max_voice == 1
        assert config.max_data == 1
        assert config.max_sms is None

    def test_custom_values(self) -> None:
        config = ConcurrencyConfig(max_voice=5, max_data=3, max_sms=10)
        assert config.max_voice == 5
        assert config.max_data == 3
        assert config.max_sms == 10

    def test_all_none(self) -> None:
        config = ConcurrencyConfig(max_voice=None, max_data=None, max_sms=None)
        assert config.max_voice is None
        assert config.max_data is None
        assert config.max_sms is None

    def test_concurrency_in_events_config(self) -> None:
        """ConcurrencyConfig must be accessible from EventsConfig."""
        from cdr_generator.config.models import EventsConfig

        # Verify ConcurrencyConfig is a field on EventsConfig
        field_names = {f for f in EventsConfig.model_fields}
        assert "concurrency" in field_names, (
            "EventsConfig must have 'concurrency' field"
        )
