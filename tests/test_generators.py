"""Tests for generators (voice, sms, data).

Phase 2 acceptance criteria tested:
- SGW+PGW paired by charging_id
- MO+MT paired by consolidation_id
- Partial records for long data sessions
- All mandatory fields present
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_subscriber() -> Subscriber:
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
def b_subscriber() -> Subscriber:
    """A second subscriber to serve as B-party."""
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
def sample_cell() -> Cell:
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
        neighbors=[20012],
    )


@pytest.fixture
def msc_ne() -> NetworkElement:
    return NetworkElement(
        id="msc-01",
        ne_type="msc",
        vendor="ericsson",
        serves_tacs=[2001, 2002],
        produces=["mo_call", "mt_call"],
    )


@pytest.fixture
def sgw_ne() -> NetworkElement:
    return NetworkElement(
        id="sgw-01",
        ne_type="sgw",
        vendor="ericsson",
        serves_tacs=[2001, 2002],
        produces=["sgw_data"],
    )


@pytest.fixture
def pgw_ne() -> NetworkElement:
    return NetworkElement(
        id="pgw-01",
        ne_type="pgw",
        vendor="nokia",
        serves_apns=["internet", "mms", "ims"],
        produces=["pgw_data"],
    )


@pytest.fixture
def smsc_ne() -> NetworkElement:
    return NetworkElement(
        id="smsc-01",
        ne_type="smsc",
        vendor="ericsson",
        produces=["mo_sms", "mt_sms"],
    )


@pytest.fixture
def event_time() -> datetime:
    return datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def voice_config() -> dict:
    """Minimal voice event config dict for generator functions."""
    return {
        "success_rate": 1.0,
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


@pytest.fixture
def data_config() -> dict:
    """Minimal data event config dict for generator functions."""
    return {
        "duration": {
            "distribution": {"type": "lognormal", "params": {"mu": 6.0, "sigma": 1.5}},
            "min_seconds": 5,
            "max_seconds": 86400,
        },
        "volume_uplink": {
            "distribution": {"type": "lognormal", "params": {"mu": 12.0, "sigma": 2.5}},
            "min_bytes": 100,
        },
        "volume_downlink": {
            "distribution": {"type": "lognormal", "params": {"mu": 14.0, "sigma": 2.5}},
            "min_bytes": 100,
        },
        "apn_weights": {"internet": 1.0},
        "qos_distribution": [{"qci": 9, "weight": 1.0, "label": "default"}],
        "partial_records": {
            "enabled": True,
            "max_record_duration_seconds": 3600,
            "max_record_volume_bytes": 104857600,
        },
        "termination_causes": [{"cause": "normal_release", "weight": 1.0}],
    }


@pytest.fixture
def sms_config() -> dict:
    """Minimal SMS event config dict for generator functions."""
    return {
        "delivery_success_rate": 1.0,
        "delivery_delay": {
            "distribution": {"type": "lognormal", "params": {"mu": 1.0, "sigma": 1.5}},
            "min_seconds": 0.5,
            "max_seconds": 86400,
        },
        "failure_causes": [{"cause": "absent_subscriber", "weight": 1.0}],
    }


# ===================================================================
# Voice generator tests
# ===================================================================


class TestVoiceMoMtPair:
    """generate_voice_cdr must return MO+MT pair on successful call."""

    def test_success_returns_two_cdrs(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        msc_ne,
        sample_cell,
        event_time,
        voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        records = generate_voice_cdr(
            caller=sample_subscriber,
            callee=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            msc=msc_ne,
            voice_cfg=voice_config,
            rng=rng,
        )
        assert len(records) == 2, (
            f"Successful voice call must produce 2 CDRs (MO+MT), got {len(records)}"
        )

    def test_mo_and_mt_record_types(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        msc_ne,
        sample_cell,
        event_time,
        voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        records = generate_voice_cdr(
            caller=sample_subscriber,
            callee=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            msc=msc_ne,
            voice_cfg=voice_config,
            rng=rng,
        )
        types = {r.record_type for r in records}
        assert "mo_call" in types, "Must include MO record"
        assert "mt_call" in types, "Must include MT record"


class TestVoiceConsolidationId:
    """MO and MT records from the same call share a consolidation_id."""

    def test_shared_consolidation_id(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        msc_ne,
        sample_cell,
        event_time,
        voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        records = generate_voice_cdr(
            caller=sample_subscriber,
            callee=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            msc=msc_ne,
            voice_cfg=voice_config,
            rng=rng,
        )
        assert len(records) == 2
        mo = [r for r in records if r.record_type == "mo_call"][0]
        mt = [r for r in records if r.record_type == "mt_call"][0]

        assert mo.consolidation_id is not None, "MO must have consolidation_id"
        assert mt.consolidation_id is not None, "MT must have consolidation_id"
        assert mo.consolidation_id == mt.consolidation_id, (
            "MO and MT must share the same consolidation_id"
        )


class TestVoiceFailure:
    """When success_rate=0, all calls fail -- only MO CDR is produced."""

    def test_failure_returns_one_cdr(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        msc_ne,
        sample_cell,
        event_time,
        voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        voice_config["success_rate"] = 0.0
        records = generate_voice_cdr(
            caller=sample_subscriber,
            callee=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            msc=msc_ne,
            voice_cfg=voice_config,
            rng=rng,
        )
        assert len(records) == 1, (
            f"Failed voice call must produce 1 CDR (MO only), got {len(records)}"
        )
        assert records[0].record_type == "mo_call"

    def test_failure_has_cause_code(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        msc_ne,
        sample_cell,
        event_time,
        voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        voice_config["success_rate"] = 0.0
        records = generate_voice_cdr(
            caller=sample_subscriber,
            callee=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            msc=msc_ne,
            voice_cfg=voice_config,
            rng=rng,
        )
        assert records[0].cause_for_termination is not None, (
            "Failed call must have a cause_for_termination code"
        )


# ===================================================================
# SMS generator tests
# ===================================================================


class TestSmsPair:
    """generate_sms_cdr must produce MO+MT pair on successful delivery."""

    def test_success_returns_two_cdrs(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        smsc_ne,
        sample_cell,
        event_time,
        sms_config,
    ) -> None:
        from cdr_generator.generators.sms import generate_sms_cdr

        records = generate_sms_cdr(
            sender=sample_subscriber,
            recipient=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            smsc=smsc_ne,
            sms_cfg=sms_config,
            rng=rng,
        )
        assert len(records) == 2, (
            f"Successful SMS must produce 2 CDRs (MO+MT), got {len(records)}"
        )
        types = {r.record_type for r in records}
        assert "mo_sms" in types
        assert "mt_sms" in types

    def test_failure_returns_one_cdr(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        smsc_ne,
        sample_cell,
        event_time,
        sms_config,
    ) -> None:
        from cdr_generator.generators.sms import generate_sms_cdr

        sms_config["delivery_success_rate"] = 0.0
        records = generate_sms_cdr(
            sender=sample_subscriber,
            recipient=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            smsc=smsc_ne,
            sms_cfg=sms_config,
            rng=rng,
        )
        assert len(records) == 1, "Failed SMS must produce 1 CDR (MO only)"
        assert records[0].record_type == "mo_sms"


# ===================================================================
# Data generator tests
# ===================================================================


class TestDataSgwPgwPair:
    """generate_data_cdr must produce SGW+PGW pair with matching charging_id."""

    def test_pair_returned(
        self,
        rng,
        sample_subscriber,
        sgw_ne,
        pgw_ne,
        sample_cell,
        event_time,
        data_config,
    ) -> None:
        from cdr_generator.generators.data import generate_data_cdr

        records = generate_data_cdr(
            subscriber=sample_subscriber,
            event_time=event_time,
            cell=sample_cell,
            sgw=sgw_ne,
            pgw=pgw_ne,
            data_cfg=data_config,
            rng=rng,
        )
        # At least one SGW and one PGW record
        sgw_records = [r for r in records if r.record_type == "sgw_data"]
        pgw_records = [r for r in records if r.record_type == "pgw_data"]
        assert len(sgw_records) >= 1, "Must have at least one SGW record"
        assert len(pgw_records) >= 1, "Must have at least one PGW record"

    def test_charging_id_match(
        self,
        rng,
        sample_subscriber,
        sgw_ne,
        pgw_ne,
        sample_cell,
        event_time,
        data_config,
    ) -> None:
        from cdr_generator.generators.data import generate_data_cdr

        records = generate_data_cdr(
            subscriber=sample_subscriber,
            event_time=event_time,
            cell=sample_cell,
            sgw=sgw_ne,
            pgw=pgw_ne,
            data_cfg=data_config,
            rng=rng,
        )
        sgw_records = [r for r in records if r.record_type == "sgw_data"]
        pgw_records = [r for r in records if r.record_type == "pgw_data"]

        sgw_ids = {r.charging_id for r in sgw_records}
        pgw_ids = {r.charging_id for r in pgw_records}
        assert sgw_ids == pgw_ids, (
            f"SGW charging_ids {sgw_ids} must match PGW charging_ids {pgw_ids}"
        )

    def test_charging_id_not_none(
        self,
        rng,
        sample_subscriber,
        sgw_ne,
        pgw_ne,
        sample_cell,
        event_time,
        data_config,
    ) -> None:
        from cdr_generator.generators.data import generate_data_cdr

        records = generate_data_cdr(
            subscriber=sample_subscriber,
            event_time=event_time,
            cell=sample_cell,
            sgw=sgw_ne,
            pgw=pgw_ne,
            data_cfg=data_config,
            rng=rng,
        )
        for r in records:
            assert r.charging_id is not None, (
                f"Data record {r.record_type} must have a charging_id"
            )


class TestDataPartialRecords:
    """Long data sessions must produce multiple partial records."""

    def test_long_session_creates_partials(
        self,
        rng,
        sample_subscriber,
        sgw_ne,
        pgw_ne,
        sample_cell,
        event_time,
        data_config,
    ) -> None:
        from cdr_generator.generators.data import generate_data_cdr

        # Force a very long session by setting very low max_record_duration
        data_config["partial_records"]["max_record_duration_seconds"] = 60
        # Override duration to force a long session (~600s)
        data_config["duration"] = {
            "distribution": {"type": "constant", "params": {"value": 600}},
            "min_seconds": 600,
            "max_seconds": 600,
        }

        records = generate_data_cdr(
            subscriber=sample_subscriber,
            event_time=event_time,
            cell=sample_cell,
            sgw=sgw_ne,
            pgw=pgw_ne,
            data_cfg=data_config,
            rng=rng,
        )
        # 600s session with 60s max -> should produce multiple partial records
        # (at least 10 SGW + 10 PGW, or equivalent)
        assert len(records) > 2, (
            f"Long session (600s, max 60s per record) must create >2 records, "
            f"got {len(records)}"
        )

    def test_partial_records_share_charging_id(
        self,
        rng,
        sample_subscriber,
        sgw_ne,
        pgw_ne,
        sample_cell,
        event_time,
        data_config,
    ) -> None:
        from cdr_generator.generators.data import generate_data_cdr

        data_config["partial_records"]["max_record_duration_seconds"] = 60
        data_config["duration"] = {
            "distribution": {"type": "constant", "params": {"value": 300}},
            "min_seconds": 300,
            "max_seconds": 300,
        }

        records = generate_data_cdr(
            subscriber=sample_subscriber,
            event_time=event_time,
            cell=sample_cell,
            sgw=sgw_ne,
            pgw=pgw_ne,
            data_cfg=data_config,
            rng=rng,
        )
        charging_ids = {r.charging_id for r in records}
        assert len(charging_ids) == 1, (
            f"All partial records must share one charging_id, found {charging_ids}"
        )


# ===================================================================
# Mandatory fields test
# ===================================================================


class TestAllFieldsPresent:
    """All CDR records must have mandatory fields populated (not None)."""

    MANDATORY_FIELDS = [
        "record_type",
        "served_imsi",
        "event_timestamp",
        "serving_ne_id",
        "served_msisdn",
    ]

    def test_voice_mandatory_fields(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        msc_ne,
        sample_cell,
        event_time,
        voice_config,
    ) -> None:
        from cdr_generator.generators.voice import generate_voice_cdr

        records = generate_voice_cdr(
            caller=sample_subscriber,
            callee=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            msc=msc_ne,
            voice_cfg=voice_config,
            rng=rng,
        )
        for rec in records:
            for field in self.MANDATORY_FIELDS:
                assert getattr(rec, field) is not None, (
                    f"Voice CDR ({rec.record_type}) field '{field}' must not be None"
                )

    def test_data_mandatory_fields(
        self,
        rng,
        sample_subscriber,
        sgw_ne,
        pgw_ne,
        sample_cell,
        event_time,
        data_config,
    ) -> None:
        from cdr_generator.generators.data import generate_data_cdr

        records = generate_data_cdr(
            subscriber=sample_subscriber,
            event_time=event_time,
            cell=sample_cell,
            sgw=sgw_ne,
            pgw=pgw_ne,
            data_cfg=data_config,
            rng=rng,
        )
        for rec in records:
            for field in self.MANDATORY_FIELDS:
                assert getattr(rec, field) is not None, (
                    f"Data CDR ({rec.record_type}) field '{field}' must not be None"
                )

    def test_sms_mandatory_fields(
        self,
        rng,
        sample_subscriber,
        b_subscriber,
        smsc_ne,
        sample_cell,
        event_time,
        sms_config,
    ) -> None:
        from cdr_generator.generators.sms import generate_sms_cdr

        records = generate_sms_cdr(
            sender=sample_subscriber,
            recipient=b_subscriber,
            event_time=event_time,
            cell=sample_cell,
            smsc=smsc_ne,
            sms_cfg=sms_config,
            rng=rng,
        )
        for rec in records:
            for field in self.MANDATORY_FIELDS:
                assert getattr(rec, field) is not None, (
                    f"SMS CDR ({rec.record_type}) field '{field}' must not be None"
                )
