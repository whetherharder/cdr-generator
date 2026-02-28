"""Tests for Phase 5 call forwarding in generate_voice_cdr.

Call forwarding behaviour:
- A configurable fraction (call_forwarding_rate) of *successful* calls are forwarded.
- A forwarded call produces 3 CDRs:
    1. MO (A -> B):   record_type="mo_call", caller=A, called=B
    2. MT forwarding:  record_type="mt_call", served_imsi=B, redirecting_number=C.msisdn
    3. MT final:       record_type="mt_call", served_imsi=C, called_number=C.msisdn
- All 3 CDRs share the same consolidation_id.
- Failed calls are never forwarded (still produce 1 CDR).
- Non-forwarded successful calls produce 2 CDRs as before.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def caller() -> Subscriber:
    """A-party subscriber initiating the call."""
    return Subscriber(
        imsi="250010000000001",
        msisdn="+79160000001",
        imei="353325080000001",
        profile_name="office_worker",
        home_cell_id=20011,
        work_cell_id=20012,
        serving_ne_id="msc-01",
    )


@pytest.fixture
def callee() -> Subscriber:
    """B-party subscriber (original call target)."""
    return Subscriber(
        imsi="250010000000002",
        msisdn="+79160000002",
        imei="353325080000002",
        profile_name="office_worker",
        home_cell_id=20012,
        work_cell_id=20011,
        serving_ne_id="msc-01",
    )


@pytest.fixture
def forward_target() -> Subscriber:
    """C-party subscriber (call forwarding destination)."""
    return Subscriber(
        imsi="250010000000003",
        msisdn="+79160000003",
        imei="353325080000003",
        profile_name="office_worker",
        home_cell_id=20013,
        work_cell_id=20014,
        serving_ne_id="msc-01",
    )


@pytest.fixture
def cell() -> Cell:
    return Cell(
        cell_id=20011,
        tac=2001,
        ecgi="25001-20011",
        lat=55.7558,
        lon=37.6173,
        azimuth=0,
        sector=1,
        cell_type="urban",
        capacity="high",
        neighbors=[20012, 20013],
    )


@pytest.fixture
def msc() -> NetworkElement:
    return NetworkElement(
        id="msc-01",
        ne_type="msc",
        vendor="ericsson",
        serves_tacs=[2001, 2002],
        produces=["mo_call", "mt_call"],
    )


@pytest.fixture
def event_time() -> datetime:
    return datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def base_voice_config() -> dict:
    """Voice config with 100% success rate, no forwarding by default."""
    return {
        "success_rate": 1.0,
        "call_forwarding_rate": 0.0,
        "failure_causes": [{"cause": "no_answer", "code": 19, "weight": 1.0}],
        "duration": {
            "distribution": {"type": "lognormal", "params": {"mu": 4.0, "sigma": 1.2}},
            "min_seconds": 1,
            "max_seconds": 7200,
        },
        "normal_termination_causes": [
            {"cause": "normal_clearing", "code": 16, "weight": 1.0},
        ],
        "paired_timestamp_jitter_ms": 1000,
    }


# ---------------------------------------------------------------------------
# Tests: Non-forwarded calls (rate=0) remain unchanged
# ---------------------------------------------------------------------------


class TestNonForwardedCallsUnchanged:
    """When call_forwarding_rate=0, behaviour is identical to Phase 2."""

    def test_successful_call_produces_two_cdrs(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 0.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        assert len(records) == 2, (
            f"Non-forwarded successful call must produce 2 CDRs, got {len(records)}"
        )

    def test_non_forwarded_has_no_redirecting_number(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 0.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        for r in records:
            assert r.redirecting_number is None, (
                f"Non-forwarded CDR ({r.record_type}) must not have redirecting_number"
            )

    def test_failed_call_not_forwarded(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        """Failed calls are never forwarded regardless of call_forwarding_rate."""
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["success_rate"] = 0.0
        base_voice_config["call_forwarding_rate"] = 1.0  # 100% rate, but call fails
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        assert len(records) == 1, (
            f"Failed call must produce 1 CDR even with forwarding_rate=1.0, got {len(records)}"
        )
        assert records[0].record_type == "mo_call"


# ---------------------------------------------------------------------------
# Tests: Forwarded calls (rate=1.0) produce 3 CDRs
# ---------------------------------------------------------------------------


class TestForwardedCallProducesThreeCdrs:
    """When call_forwarding_rate=1.0, all successful calls produce 3 CDRs."""

    def test_forwarded_call_returns_three_records(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        assert len(records) == 3, (
            f"Forwarded call must produce 3 CDRs (MO + MT-fwd + MT-final), got {len(records)}"
        )

    def test_forwarded_record_types(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        types = [r.record_type for r in records]
        assert types[0] == "mo_call", "First record must be MO"
        assert types[1] == "mt_call", "Second record must be MT (forwarding leg)"
        assert types[2] == "mt_call", "Third record must be MT (final leg)"


class TestForwardedCallConsolidationId:
    """All 3 CDRs from a forwarded call share the same consolidation_id."""

    def test_all_three_share_consolidation_id(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        assert len(records) == 3
        cids = {r.consolidation_id for r in records}
        assert len(cids) == 1, (
            f"All 3 forwarded CDRs must share one consolidation_id, found {cids}"
        )
        assert None not in cids, "consolidation_id must not be None"


class TestForwardedCallMoRecord:
    """The MO record of a forwarded call targets the original B-party."""

    def test_mo_caller_is_a_party(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        mo = records[0]
        assert mo.served_imsi == caller.imsi, "MO served_imsi must be A-party"
        assert mo.calling_number == caller.msisdn, "MO calling_number must be A-party"
        assert mo.called_number == callee.msisdn, "MO called_number must be B-party"


class TestForwardedCallMtForwardingRecord:
    """The MT forwarding record: served by B-party, has redirecting_number=C."""

    def test_mt_forwarding_served_by_callee(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        mt_fwd = records[1]
        assert mt_fwd.served_imsi == callee.imsi, (
            "MT forwarding record served_imsi must be B-party (callee)"
        )

    def test_mt_forwarding_has_redirecting_number(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        mt_fwd = records[1]
        assert mt_fwd.redirecting_number == forward_target.msisdn, (
            f"MT forwarding record redirecting_number must be C-party MSISDN "
            f"({forward_target.msisdn}), got {mt_fwd.redirecting_number}"
        )


class TestForwardedCallMtFinalRecord:
    """The MT final record: served by C-party (forward target)."""

    def test_mt_final_served_by_forward_target(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        mt_final = records[2]
        assert mt_final.served_imsi == forward_target.imsi, (
            "MT final record served_imsi must be C-party (forward_target)"
        )
        assert mt_final.served_msisdn == forward_target.msisdn, (
            "MT final record served_msisdn must be C-party MSISDN"
        )

    def test_mt_final_called_number_is_forward_target(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        mt_final = records[2]
        assert mt_final.called_number == forward_target.msisdn, (
            f"MT final called_number must be C-party MSISDN ({forward_target.msisdn}), "
            f"got {mt_final.called_number}"
        )

    def test_mt_final_has_no_redirecting_number(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        """Only the forwarding leg has redirecting_number, not the final leg."""
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        mt_final = records[2]
        assert mt_final.redirecting_number is None, (
            "MT final record must NOT have redirecting_number"
        )


class TestForwardedCallDuration:
    """All legs of a forwarded call should have consistent duration."""

    def test_all_legs_have_positive_duration(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        for r in records:
            assert r.duration_seconds is not None and r.duration_seconds > 0, (
                f"CDR ({r.record_type}) must have positive duration, got {r.duration_seconds}"
            )

    def test_mo_and_mt_final_same_duration(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        """MO and MT-final should share the same call duration."""
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        mo = records[0]
        mt_final = records[2]
        assert mo.duration_seconds == mt_final.duration_seconds, (
            f"MO and MT-final must share call duration: "
            f"MO={mo.duration_seconds}, MT-final={mt_final.duration_seconds}"
        )


class TestForwardedCallMandatoryFields:
    """All 3 CDRs from a forwarded call must have mandatory fields populated."""

    MANDATORY_FIELDS = [
        "record_type",
        "served_imsi",
        "served_msisdn",
        "event_timestamp",
        "serving_ne_id",
        "consolidation_id",
    ]

    def test_all_mandatory_fields_present(
        self,
        rng,
        caller,
        callee,
        forward_target,
        cell,
        msc,
        event_time,
        base_voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            forward_target=forward_target,
        )
        for i, rec in enumerate(records):
            for field in self.MANDATORY_FIELDS:
                assert getattr(rec, field) is not None, (
                    f"Forwarded CDR #{i} ({rec.record_type}) field '{field}' must not be None"
                )


# ---------------------------------------------------------------------------
# Tests: Statistical behaviour of forwarding rate
# ---------------------------------------------------------------------------


class TestForwardingRateStatistical:
    """Verify call_forwarding_rate is respected statistically."""

    def test_rate_zero_never_forwards(
        self, caller, callee, forward_target, cell, msc, event_time, base_voice_config
    ) -> None:
        """With rate=0.0, no calls should ever be forwarded."""
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 0.0
        rng = np.random.default_rng(99)
        n_trials = 200
        for _ in range(n_trials):
            records = generate_voice_cdr(
                caller=caller,
                callee=callee,
                event_time=event_time,
                cell=cell,
                msc=msc,
                voice_cfg=base_voice_config,
                rng=rng,
                forward_target=forward_target,
            )
            assert len(records) == 2, (
                f"With forwarding_rate=0.0, all successful calls must produce 2 CDRs, "
                f"got {len(records)}"
            )

    def test_rate_one_always_forwards(
        self, caller, callee, forward_target, cell, msc, event_time, base_voice_config
    ) -> None:
        """With rate=1.0, all successful calls should be forwarded (3 CDRs)."""
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        rng = np.random.default_rng(99)
        n_trials = 200
        for _ in range(n_trials):
            records = generate_voice_cdr(
                caller=caller,
                callee=callee,
                event_time=event_time,
                cell=cell,
                msc=msc,
                voice_cfg=base_voice_config,
                rng=rng,
                forward_target=forward_target,
            )
            assert len(records) == 3, (
                f"With forwarding_rate=1.0, all successful calls must produce 3 CDRs, "
                f"got {len(records)}"
            )

    def test_rate_point_five_statistical(
        self, caller, callee, forward_target, cell, msc, event_time, base_voice_config
    ) -> None:
        """With rate=0.5, roughly half of successful calls should be forwarded."""
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 0.5
        rng = np.random.default_rng(42)
        n_trials = 1000
        n_forwarded = 0
        for _ in range(n_trials):
            records = generate_voice_cdr(
                caller=caller,
                callee=callee,
                event_time=event_time,
                cell=cell,
                msc=msc,
                voice_cfg=base_voice_config,
                rng=rng,
                forward_target=forward_target,
            )
            if len(records) == 3:
                n_forwarded += 1

        ratio = n_forwarded / n_trials
        assert 0.4 < ratio < 0.6, (
            f"With forwarding_rate=0.5, expected ~50% forwarded, got {ratio:.1%} "
            f"({n_forwarded}/{n_trials})"
        )


# ---------------------------------------------------------------------------
# Tests: Backward compatibility -- forward_target=None
# ---------------------------------------------------------------------------


class TestForwardTargetNone:
    """When forward_target is not provided, forwarding cannot occur."""

    def test_no_forward_target_defaults_to_two_cdrs(
        self, rng, caller, callee, cell, msc, event_time, base_voice_config
    ) -> None:
        """Without forward_target, even rate=1.0 should produce 2 CDRs."""
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 1.0
        records = generate_voice_cdr(
            caller=caller,
            callee=callee,
            event_time=event_time,
            cell=cell,
            msc=msc,
            voice_cfg=base_voice_config,
            rng=rng,
            # forward_target is omitted (defaults to None)
        )
        assert len(records) == 2, (
            f"Without forward_target, call must produce 2 CDRs, got {len(records)}"
        )


# ---------------------------------------------------------------------------
# Tests: Deterministic seeding
# ---------------------------------------------------------------------------


class TestForwardingDeterminism:
    """Same seed must produce identical forwarding decisions."""

    def test_deterministic_with_same_seed(
        self, caller, callee, forward_target, cell, msc, event_time, base_voice_config
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        base_voice_config["call_forwarding_rate"] = 0.5

        results = []
        for _ in range(2):
            rng = np.random.default_rng(777)
            run_records = []
            for _ in range(50):
                records = generate_voice_cdr(
                    caller=caller,
                    callee=callee,
                    event_time=event_time,
                    cell=cell,
                    msc=msc,
                    voice_cfg=base_voice_config,
                    rng=rng,
                    forward_target=forward_target,
                )
                run_records.append(len(records))
            results.append(run_records)

        assert results[0] == results[1], (
            "Same seed must produce identical forwarding decisions"
        )
