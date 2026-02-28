"""Tests for the anomaly injection pipeline.

Tests all 5 anomaly stages and the AnomalyPipeline orchestrator:
  1. Orphaned records — remove one side of MO/MT pairs
  2. Missing fields — set target fields to None
  3. Corrupt values — replace with invalid data
  4. Timestamp anomalies — corrupt timestamps
  5. Duplicate records — clone with timestamp jitter

Also covers AnomalyStats, pipeline ordering, disabled stages,
and deterministic RNG behavior.
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from cdr_generator.config.models import (
    AnomaliesConfig,
    CorruptValuesConfig,
    DuplicateRecordsConfig,
    MissingFieldsConfig,
    OrphanedRecordsConfig,
    TimestampAnomaliesConfig,
)
from cdr_generator.engine.anomalies import (
    AnomalyPipeline,
    AnomalyStats,
    apply_corrupt_values,
    apply_duplicates,
    apply_missing_fields,
    apply_orphaned,
    apply_timestamp_anomalies,
)
from cdr_generator.models.cdr import CDRRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _ts(hour: int = 10, minute: int = 0, second: int = 0) -> datetime:
    """Helper to build a UTC datetime on 2025-01-15."""
    return datetime(2025, 1, 15, hour, minute, second, tzinfo=timezone.utc)


def _make_voice_pair(
    consolidation_id: str | None = None,
    ne_id: str = "msc-01",
    imsi_a: str = "250010000000001",
    imsi_b: str = "250010000000002",
    cell_id: int = 20011,
) -> list[CDRRecord]:
    """Create a matched MO+MT voice call pair."""
    cid = consolidation_id or str(uuid.uuid4())
    mo = CDRRecord(
        record_type="mo_call",
        served_imsi=imsi_a,
        served_msisdn="+79160000001",
        event_timestamp=_ts(10, 0, 0),
        answer_timestamp=_ts(10, 0, 5),
        release_timestamp=_ts(10, 2, 5),
        duration_seconds=120.0,
        cause_for_termination=16,
        first_cell_id=cell_id,
        last_cell_id=cell_id,
        serving_ne_id=ne_id,
        consolidation_id=cid,
        calling_number="+79160000001",
        called_number="+79160000002",
    )
    mt = CDRRecord(
        record_type="mt_call",
        served_imsi=imsi_b,
        served_msisdn="+79160000002",
        event_timestamp=_ts(10, 0, 1),
        answer_timestamp=_ts(10, 0, 6),
        release_timestamp=_ts(10, 2, 6),
        duration_seconds=120.0,
        cause_for_termination=16,
        first_cell_id=cell_id,
        last_cell_id=cell_id,
        serving_ne_id=ne_id,
        consolidation_id=cid,
        calling_number="+79160000001",
        called_number="+79160000002",
    )
    return [mo, mt]


def _make_sms_pair(consolidation_id: str | None = None) -> list[CDRRecord]:
    """Create a matched MO+MT SMS pair."""
    cid = consolidation_id or str(uuid.uuid4())
    mo = CDRRecord(
        record_type="mo_sms",
        served_imsi="250010000000001",
        served_msisdn="+79160000001",
        event_timestamp=_ts(11, 0, 0),
        serving_ne_id="msc-01",
        consolidation_id=cid,
        calling_number="+79160000001",
        called_number="+79160000002",
    )
    mt = CDRRecord(
        record_type="mt_sms",
        served_imsi="250010000000002",
        served_msisdn="+79160000002",
        event_timestamp=_ts(11, 0, 3),
        serving_ne_id="msc-01",
        consolidation_id=cid,
        calling_number="+79160000001",
        called_number="+79160000002",
    )
    return [mo, mt]


def _make_data_pair(charging_id: int = 100001) -> list[CDRRecord]:
    """Create a matched SGW+PGW data pair."""
    sgw = CDRRecord(
        record_type="sgw_data",
        served_imsi="250010000000001",
        served_msisdn="+79160000001",
        event_timestamp=_ts(12, 0, 0),
        duration_seconds=600.0,
        first_cell_id=20011,
        serving_ne_id="sgw-01",
        charging_id=charging_id,
        uplink_volume_bytes=50000,
        downlink_volume_bytes=200000,
        apn="internet",
    )
    pgw = CDRRecord(
        record_type="pgw_data",
        served_imsi="250010000000001",
        served_msisdn="+79160000001",
        event_timestamp=_ts(12, 0, 0),
        duration_seconds=600.0,
        first_cell_id=20011,
        serving_ne_id="pgw-01",
        charging_id=charging_id,
        uplink_volume_bytes=50000,
        downlink_volume_bytes=200000,
        apn="internet",
    )
    return [sgw, pgw]


def _make_single_record(**kwargs) -> CDRRecord:
    """Create a single CDR record with sensible defaults."""
    defaults = dict(
        record_type="mo_call",
        served_imsi="250010000000001",
        served_msisdn="+79160000001",
        event_timestamp=_ts(14, 0, 0),
        answer_timestamp=_ts(14, 0, 3),
        release_timestamp=_ts(14, 1, 3),
        duration_seconds=60.0,
        cause_for_termination=16,
        first_cell_id=20011,
        last_cell_id=20011,
        serving_ne_id="msc-01",
        consolidation_id=str(uuid.uuid4()),
    )
    defaults.update(kwargs)
    return CDRRecord(**defaults)


def _make_many_records(n: int) -> list[CDRRecord]:
    """Create n standalone CDR records (no pairing)."""
    return [_make_single_record(sequence_number=i) for i in range(n)]


def _make_many_pairs(n_pairs: int) -> list[CDRRecord]:
    """Create n voice MO/MT pairs (2*n_pairs records total)."""
    records = []
    for _ in range(n_pairs):
        records.extend(_make_voice_pair())
    return records


# ---------------------------------------------------------------------------
# AnomalyStats tests
# ---------------------------------------------------------------------------


class TestAnomalyStats:
    def test_defaults_are_zero(self):
        stats = AnomalyStats()
        assert stats.orphaned_count == 0
        assert stats.missing_fields_count == 0
        assert stats.corrupt_values_count == 0
        assert stats.timestamp_anomalies_count == 0
        assert stats.duplicate_count == 0

    def test_total_sums_all_fields(self):
        stats = AnomalyStats(
            orphaned_count=2,
            missing_fields_count=3,
            corrupt_values_count=1,
            timestamp_anomalies_count=4,
            duplicate_count=5,
        )
        assert stats.total == 15

    def test_total_zero_when_empty(self):
        assert AnomalyStats().total == 0


# ---------------------------------------------------------------------------
# Stage 1: Orphaned Records
# ---------------------------------------------------------------------------


class TestApplyOrphaned:
    """Tests for apply_orphaned — removes one side of MO/MT pairs."""

    def test_disabled_returns_unchanged(self, rng):
        config = OrphanedRecordsConfig(enabled=False, rate=0.02)
        records = _make_voice_pair()
        result, count = apply_orphaned(records, config, rng)
        assert count == 0
        assert len(result) == 2

    def test_zero_rate_removes_nothing(self, rng):
        config = OrphanedRecordsConfig(enabled=True, rate=0.0)
        records = _make_voice_pair()
        result, count = apply_orphaned(records, config, rng)
        assert count == 0
        assert len(result) == 2

    def test_rate_one_removes_from_every_pair(self, rng):
        config = OrphanedRecordsConfig(enabled=True, rate=1.0)
        pairs = _make_many_pairs(10)
        result, count = apply_orphaned(pairs, config, rng)
        # Each pair should have one record removed
        assert count == 10
        assert len(result) == 10  # 20 - 10

    def test_voice_pairs_identified_by_consolidation_id(self, rng):
        config = OrphanedRecordsConfig(enabled=True, rate=1.0)
        cid = str(uuid.uuid4())
        records = _make_voice_pair(consolidation_id=cid)
        result, count = apply_orphaned(records, config, rng)
        assert count == 1
        assert len(result) == 1
        # Remaining record still has the same consolidation_id
        assert result[0].consolidation_id == cid

    def test_sms_pairs_identified_by_consolidation_id(self, rng):
        config = OrphanedRecordsConfig(enabled=True, rate=1.0)
        records = _make_sms_pair()
        result, count = apply_orphaned(records, config, rng)
        assert count == 1
        assert len(result) == 1

    def test_data_pairs_identified_by_charging_id(self, rng):
        config = OrphanedRecordsConfig(enabled=True, rate=1.0)
        records = _make_data_pair(charging_id=999)
        result, count = apply_orphaned(records, config, rng)
        assert count == 1
        assert len(result) == 1

    def test_mixed_pairs(self, rng):
        """Mix of voice, sms, data pairs -- all should be affected at rate=1.0."""
        config = OrphanedRecordsConfig(enabled=True, rate=1.0)
        records = _make_voice_pair() + _make_sms_pair() + _make_data_pair()
        result, count = apply_orphaned(records, config, rng)
        # 3 pairs, each loses one record
        assert count == 3
        assert len(result) == 3

    def test_unpaired_records_not_affected(self, rng):
        """Records without a matching pair should not be removed."""
        config = OrphanedRecordsConfig(enabled=True, rate=1.0)
        # Single MO record with unique consolidation_id (no MT partner)
        solo = _make_single_record(record_type="mo_call")
        result, count = apply_orphaned([solo], config, rng)
        # No pair to orphan
        assert count == 0
        assert len(result) == 1

    def test_empty_input(self, rng):
        config = OrphanedRecordsConfig(enabled=True, rate=0.5)
        result, count = apply_orphaned([], config, rng)
        assert count == 0
        assert result == []

    def test_statistical_rate(self, rng):
        """With rate=0.02 and many pairs, ~2% should be orphaned."""
        config = OrphanedRecordsConfig(enabled=True, rate=0.02)
        pairs = _make_many_pairs(1000)
        result, count = apply_orphaned(pairs, config, rng)
        # 2% of 1000 pairs = ~20 removed, allow wide tolerance
        assert 5 <= count <= 50
        assert len(result) == 2000 - count

    def test_deterministic_with_same_seed(self):
        """Same seed + same input => same output."""
        config = OrphanedRecordsConfig(enabled=True, rate=0.5)
        pairs = _make_many_pairs(50)

        rng1 = np.random.default_rng(123)
        result1, count1 = apply_orphaned(copy.deepcopy(pairs), config, rng1)

        rng2 = np.random.default_rng(123)
        result2, count2 = apply_orphaned(copy.deepcopy(pairs), config, rng2)

        assert count1 == count2
        assert len(result1) == len(result2)


# ---------------------------------------------------------------------------
# Stage 2: Missing Fields
# ---------------------------------------------------------------------------


class TestApplyMissingFields:
    """Tests for apply_missing_fields — sets target fields to None."""

    @pytest.fixture
    def missing_config(self) -> MissingFieldsConfig:
        return MissingFieldsConfig(
            enabled=True,
            rate=0.01,
            target_fields=[
                "served_msisdn",
                "cell_id",
                "duration_seconds",
                "cause_for_termination",
            ],
        )

    def test_disabled_returns_unchanged(self, rng, missing_config):
        missing_config.enabled = False
        records = _make_many_records(10)
        result, count = apply_missing_fields(records, missing_config, rng)
        assert count == 0
        assert len(result) == 10

    def test_zero_rate_changes_nothing(self, rng):
        config = MissingFieldsConfig(
            enabled=True, rate=0.0, target_fields=["served_msisdn"]
        )
        records = _make_many_records(10)
        result, count = apply_missing_fields(records, config, rng)
        assert count == 0

    def test_rate_one_affects_all_records(self, rng, missing_config):
        missing_config.rate = 1.0
        records = _make_many_records(20)
        result, count = apply_missing_fields(records, missing_config, rng)
        assert count == 20
        assert len(result) == 20  # length unchanged

    def test_cell_id_maps_to_first_cell_id(self, rng):
        """Config field 'cell_id' should null out 'first_cell_id' on CDRRecord."""
        config = MissingFieldsConfig(enabled=True, rate=1.0, target_fields=["cell_id"])
        record = _make_single_record(first_cell_id=20011)
        result, count = apply_missing_fields([record], config, rng)
        assert count == 1
        assert result[0].first_cell_id is None

    def test_served_msisdn_nulled(self, rng):
        config = MissingFieldsConfig(
            enabled=True, rate=1.0, target_fields=["served_msisdn"]
        )
        record = _make_single_record(served_msisdn="+79160000001")
        result, count = apply_missing_fields([record], config, rng)
        assert count == 1
        assert result[0].served_msisdn is None

    def test_duration_seconds_nulled(self, rng):
        config = MissingFieldsConfig(
            enabled=True, rate=1.0, target_fields=["duration_seconds"]
        )
        record = _make_single_record(duration_seconds=60.0)
        result, count = apply_missing_fields([record], config, rng)
        assert count == 1
        assert result[0].duration_seconds is None

    def test_cause_for_termination_nulled(self, rng):
        config = MissingFieldsConfig(
            enabled=True, rate=1.0, target_fields=["cause_for_termination"]
        )
        record = _make_single_record(cause_for_termination=16)
        result, count = apply_missing_fields([record], config, rng)
        assert count == 1
        assert result[0].cause_for_termination is None

    def test_one_field_per_record(self, rng, missing_config):
        """Each affected record should have exactly one field set to None."""
        missing_config.rate = 1.0
        records = _make_many_records(100)
        original = copy.deepcopy(records)
        result, count = apply_missing_fields(records, missing_config, rng)
        assert count == 100
        for orig, res in zip(original, result):
            nulled = 0
            # Check which target fields were nulled
            field_map = {
                "served_msisdn": "served_msisdn",
                "cell_id": "first_cell_id",
                "duration_seconds": "duration_seconds",
                "cause_for_termination": "cause_for_termination",
            }
            for cfg_name, attr_name in field_map.items():
                orig_val = getattr(orig, attr_name)
                res_val = getattr(res, attr_name)
                if orig_val is not None and res_val is None:
                    nulled += 1
            assert nulled == 1, f"Expected exactly 1 field nulled, got {nulled}"

    def test_list_length_unchanged(self, rng, missing_config):
        missing_config.rate = 0.5
        records = _make_many_records(50)
        result, count = apply_missing_fields(records, missing_config, rng)
        assert len(result) == 50

    def test_empty_input(self, rng, missing_config):
        result, count = apply_missing_fields([], missing_config, rng)
        assert count == 0
        assert result == []

    def test_empty_target_fields(self, rng):
        config = MissingFieldsConfig(enabled=True, rate=1.0, target_fields=[])
        records = _make_many_records(10)
        result, count = apply_missing_fields(records, config, rng)
        # No target fields to null, so nothing should happen
        assert count == 0


# ---------------------------------------------------------------------------
# Stage 3: Corrupt Values
# ---------------------------------------------------------------------------


class TestApplyCorruptValues:
    """Tests for apply_corrupt_values — replaces fields with invalid data."""

    @pytest.fixture
    def corrupt_config(self) -> CorruptValuesConfig:
        return CorruptValuesConfig(
            enabled=True,
            rate=0.002,
            types=["invalid_imsi", "impossible_cell_id", "extreme_duration"],
        )

    def test_disabled_returns_unchanged(self, rng, corrupt_config):
        corrupt_config.enabled = False
        records = _make_many_records(10)
        result, count = apply_corrupt_values(records, corrupt_config, rng)
        assert count == 0
        assert len(result) == 10

    def test_zero_rate_changes_nothing(self, rng):
        config = CorruptValuesConfig(enabled=True, rate=0.0, types=["invalid_imsi"])
        records = _make_many_records(10)
        result, count = apply_corrupt_values(records, config, rng)
        assert count == 0

    def test_rate_one_corrupts_all(self, rng, corrupt_config):
        corrupt_config.rate = 1.0
        records = _make_many_records(20)
        result, count = apply_corrupt_values(records, corrupt_config, rng)
        assert count == 20

    def test_invalid_imsi_corruption(self, rng):
        """Invalid IMSI should be non-numeric or wrong length."""
        config = CorruptValuesConfig(enabled=True, rate=1.0, types=["invalid_imsi"])
        record = _make_single_record(served_imsi="250010000000001")
        result, count = apply_corrupt_values([record], config, rng)
        assert count == 1
        corrupted_imsi = result[0].served_imsi
        # Must be different from original
        assert corrupted_imsi != "250010000000001"
        # Should be invalid: non-numeric or wrong length
        is_invalid = not corrupted_imsi.isdigit() or len(corrupted_imsi) != 15
        assert is_invalid, f"IMSI '{corrupted_imsi}' should be invalid"

    def test_impossible_cell_id_corruption(self, rng):
        """Impossible cell_id should be a value like 9999999."""
        config = CorruptValuesConfig(
            enabled=True, rate=1.0, types=["impossible_cell_id"]
        )
        record = _make_single_record(first_cell_id=20011)
        result, count = apply_corrupt_values([record], config, rng)
        assert count == 1
        # Cell ID should be set to an impossible value
        assert result[0].first_cell_id != 20011
        assert result[0].first_cell_id == 9999999

    def test_extreme_duration_corruption(self, rng):
        """Extreme duration should be set to 999999.0."""
        config = CorruptValuesConfig(enabled=True, rate=1.0, types=["extreme_duration"])
        record = _make_single_record(duration_seconds=60.0)
        result, count = apply_corrupt_values([record], config, rng)
        assert count == 1
        assert result[0].duration_seconds == 999999.0

    def test_list_length_unchanged(self, rng, corrupt_config):
        corrupt_config.rate = 0.5
        records = _make_many_records(50)
        result, count = apply_corrupt_values(records, corrupt_config, rng)
        assert len(result) == 50

    def test_empty_input(self, rng, corrupt_config):
        result, count = apply_corrupt_values([], corrupt_config, rng)
        assert count == 0
        assert result == []

    def test_empty_types_list(self, rng):
        config = CorruptValuesConfig(enabled=True, rate=1.0, types=[])
        records = _make_many_records(10)
        result, count = apply_corrupt_values(records, config, rng)
        # No corruption types configured, so nothing should happen
        assert count == 0

    def test_one_corruption_per_record(self, rng, corrupt_config):
        """Each affected record should have exactly one corruption type applied."""
        corrupt_config.rate = 1.0
        records = _make_many_records(100)
        original = copy.deepcopy(records)
        result, _ = apply_corrupt_values(records, corrupt_config, rng)
        for orig, res in zip(original, result):
            changes = 0
            if res.served_imsi != orig.served_imsi:
                changes += 1
            if res.first_cell_id != orig.first_cell_id:
                changes += 1
            if res.duration_seconds != orig.duration_seconds:
                changes += 1
            assert changes == 1, f"Expected 1 corruption, got {changes}"


# ---------------------------------------------------------------------------
# Stage 4: Timestamp Anomalies
# ---------------------------------------------------------------------------


class TestApplyTimestampAnomalies:
    """Tests for apply_timestamp_anomalies — corrupt timestamps."""

    @pytest.fixture
    def ts_config(self) -> TimestampAnomaliesConfig:
        return TimestampAnomaliesConfig(
            enabled=True,
            rate=0.003,
            types=["future_timestamp", "negative_duration", "midnight_rollover"],
        )

    def test_disabled_returns_unchanged(self, rng, ts_config):
        ts_config.enabled = False
        records = _make_many_records(10)
        result, count = apply_timestamp_anomalies(records, ts_config, rng)
        assert count == 0
        assert len(result) == 10

    def test_zero_rate_changes_nothing(self, rng):
        config = TimestampAnomaliesConfig(
            enabled=True, rate=0.0, types=["future_timestamp"]
        )
        records = _make_many_records(10)
        result, count = apply_timestamp_anomalies(records, config, rng)
        assert count == 0

    def test_rate_one_affects_all(self, rng, ts_config):
        ts_config.rate = 1.0
        records = _make_many_records(20)
        result, count = apply_timestamp_anomalies(records, ts_config, rng)
        assert count == 20

    def test_future_timestamp(self, rng):
        """Future timestamp should push event_timestamp 1-30 days ahead."""
        config = TimestampAnomaliesConfig(
            enabled=True, rate=1.0, types=["future_timestamp"]
        )
        original_ts = _ts(10, 0, 0)
        record = _make_single_record(event_timestamp=original_ts)
        result, count = apply_timestamp_anomalies([record], config, rng)
        assert count == 1
        assert result[0].event_timestamp > original_ts
        delta = result[0].event_timestamp - original_ts
        assert timedelta(days=1) <= delta <= timedelta(days=30)

    def test_negative_duration(self, rng):
        """Negative duration: either duration_seconds < 0, or release before answer."""
        config = TimestampAnomaliesConfig(
            enabled=True, rate=1.0, types=["negative_duration"]
        )
        record = _make_single_record(
            duration_seconds=60.0,
            answer_timestamp=_ts(10, 0, 0),
            release_timestamp=_ts(10, 1, 0),
        )
        result, count = apply_timestamp_anomalies([record], config, rng)
        assert count == 1
        rec = result[0]
        # Either duration is negative, or release is before answer
        has_negative_dur = rec.duration_seconds is not None and rec.duration_seconds < 0
        has_release_before_answer = (
            rec.answer_timestamp is not None
            and rec.release_timestamp is not None
            and rec.release_timestamp < rec.answer_timestamp
        )
        assert has_negative_dur or has_release_before_answer

    def test_midnight_rollover(self, rng):
        """Midnight rollover should create a cross-day record."""
        config = TimestampAnomaliesConfig(
            enabled=True, rate=1.0, types=["midnight_rollover"]
        )
        record = _make_single_record(
            event_timestamp=_ts(10, 0, 0),
            release_timestamp=_ts(10, 1, 0),
        )
        result, count = apply_timestamp_anomalies([record], config, rng)
        assert count == 1
        rec = result[0]
        # Event should be near midnight (23:xx) and release on the next day
        assert rec.event_timestamp.hour >= 23
        if rec.release_timestamp is not None:
            assert rec.release_timestamp.date() > rec.event_timestamp.date()

    def test_list_length_unchanged(self, rng, ts_config):
        ts_config.rate = 0.5
        records = _make_many_records(50)
        result, count = apply_timestamp_anomalies(records, ts_config, rng)
        assert len(result) == 50

    def test_empty_input(self, rng, ts_config):
        result, count = apply_timestamp_anomalies([], ts_config, rng)
        assert count == 0
        assert result == []

    def test_empty_types_list(self, rng):
        config = TimestampAnomaliesConfig(enabled=True, rate=1.0, types=[])
        records = _make_many_records(10)
        result, count = apply_timestamp_anomalies(records, config, rng)
        assert count == 0


# ---------------------------------------------------------------------------
# Stage 5: Duplicate Records
# ---------------------------------------------------------------------------


class TestApplyDuplicates:
    """Tests for apply_duplicates — clone records with timestamp jitter."""

    @pytest.fixture
    def dup_config(self) -> DuplicateRecordsConfig:
        return DuplicateRecordsConfig(enabled=True, rate=0.005, max_time_shift_ms=500)

    def test_disabled_returns_unchanged(self, rng, dup_config):
        dup_config.enabled = False
        records = _make_many_records(10)
        result, count = apply_duplicates(records, dup_config, rng)
        assert count == 0
        assert len(result) == 10

    def test_zero_rate_adds_nothing(self, rng):
        config = DuplicateRecordsConfig(enabled=True, rate=0.0, max_time_shift_ms=500)
        records = _make_many_records(10)
        result, count = apply_duplicates(records, config, rng)
        assert count == 0
        assert len(result) == 10

    def test_rate_one_duplicates_all(self, rng, dup_config):
        dup_config.rate = 1.0
        records = _make_many_records(20)
        result, count = apply_duplicates(records, dup_config, rng)
        assert count == 20
        assert len(result) == 40  # 20 originals + 20 duplicates

    def test_duplicate_has_jittered_timestamp(self, rng, dup_config):
        dup_config.rate = 1.0
        original_ts = _ts(10, 0, 0)
        record = _make_single_record(event_timestamp=original_ts)
        result, count = apply_duplicates([record], dup_config, rng)
        assert count == 1
        assert len(result) == 2
        # One should be the original, one the duplicate
        original = result[0]
        duplicate = result[1]
        # Duplicate timestamp should differ from original within max_time_shift_ms
        ts_diff = abs(
            (duplicate.event_timestamp - original.event_timestamp).total_seconds()
            * 1000
        )
        assert ts_diff <= dup_config.max_time_shift_ms

    def test_duplicate_preserves_fields(self, rng, dup_config):
        """Duplicate should have same field values as original (except timestamp jitter)."""
        dup_config.rate = 1.0
        record = _make_single_record(
            served_imsi="250010000000099",
            served_msisdn="+79160000099",
            duration_seconds=180.0,
            first_cell_id=20012,
        )
        result, count = apply_duplicates([record], dup_config, rng)
        assert count == 1
        dup = result[1]  # duplicate appended after original
        assert dup.served_imsi == record.served_imsi
        assert dup.served_msisdn == record.served_msisdn
        assert dup.duration_seconds == record.duration_seconds
        assert dup.first_cell_id == record.first_cell_id
        assert dup.record_type == record.record_type

    def test_list_grows_by_duplicate_count(self, rng, dup_config):
        dup_config.rate = 0.5
        records = _make_many_records(100)
        result, count = apply_duplicates(records, dup_config, rng)
        assert len(result) == 100 + count

    def test_empty_input(self, rng, dup_config):
        result, count = apply_duplicates([], dup_config, rng)
        assert count == 0
        assert result == []

    def test_deterministic_with_same_seed(self, dup_config):
        dup_config.rate = 0.5
        records = _make_many_records(50)

        rng1 = np.random.default_rng(99)
        result1, count1 = apply_duplicates(copy.deepcopy(records), dup_config, rng1)

        rng2 = np.random.default_rng(99)
        result2, count2 = apply_duplicates(copy.deepcopy(records), dup_config, rng2)

        assert count1 == count2
        assert len(result1) == len(result2)


# ---------------------------------------------------------------------------
# AnomalyPipeline tests
# ---------------------------------------------------------------------------


class TestAnomalyPipeline:
    """Tests for AnomalyPipeline — orchestrates all 5 stages in order."""

    @pytest.fixture
    def full_config(self) -> AnomaliesConfig:
        return AnomaliesConfig(
            enabled=True,
            orphaned_records=OrphanedRecordsConfig(enabled=True, rate=0.02),
            missing_fields=MissingFieldsConfig(
                enabled=True,
                rate=0.01,
                target_fields=[
                    "served_msisdn",
                    "cell_id",
                    "duration_seconds",
                    "cause_for_termination",
                ],
            ),
            corrupt_values=CorruptValuesConfig(
                enabled=True,
                rate=0.002,
                types=["invalid_imsi", "impossible_cell_id", "extreme_duration"],
            ),
            timestamp_anomalies=TimestampAnomaliesConfig(
                enabled=True,
                rate=0.003,
                types=["future_timestamp", "negative_duration", "midnight_rollover"],
            ),
            duplicate_records=DuplicateRecordsConfig(
                enabled=True, rate=0.005, max_time_shift_ms=500
            ),
        )

    def test_pipeline_returns_records_and_stats(self, rng, full_config):
        pipeline = AnomalyPipeline(full_config, rng)
        records = _make_many_pairs(50) + _make_many_records(50)
        result, stats = pipeline.apply(records)
        assert isinstance(result, list)
        assert isinstance(stats, AnomalyStats)

    def test_pipeline_applies_all_stages(self, rng):
        """With high rates, all stages should be reflected in stats."""
        config = AnomaliesConfig(
            enabled=True,
            orphaned_records=OrphanedRecordsConfig(enabled=True, rate=1.0),
            missing_fields=MissingFieldsConfig(
                enabled=True, rate=1.0, target_fields=["served_msisdn"]
            ),
            corrupt_values=CorruptValuesConfig(
                enabled=True, rate=1.0, types=["extreme_duration"]
            ),
            timestamp_anomalies=TimestampAnomaliesConfig(
                enabled=True, rate=1.0, types=["future_timestamp"]
            ),
            duplicate_records=DuplicateRecordsConfig(
                enabled=True, rate=1.0, max_time_shift_ms=500
            ),
        )
        pipeline = AnomalyPipeline(config, rng)
        records = _make_many_pairs(10)  # 20 records, 10 pairs
        result, stats = pipeline.apply(records)
        # Orphaned should have removed ~10 records
        assert stats.orphaned_count > 0
        # After orphaned, remaining records get missing, corrupt, ts, duplicates
        assert stats.missing_fields_count > 0
        assert stats.corrupt_values_count > 0
        assert stats.timestamp_anomalies_count > 0
        assert stats.duplicate_count > 0
        assert stats.total > 0

    def test_pipeline_disabled_returns_unchanged(self, rng):
        config = AnomaliesConfig(enabled=False)
        pipeline = AnomalyPipeline(config, rng)
        records = _make_many_records(20)
        result, stats = pipeline.apply(records)
        assert len(result) == 20
        assert stats.total == 0

    def test_all_stages_disabled(self, rng):
        config = AnomaliesConfig(
            enabled=True,
            orphaned_records=OrphanedRecordsConfig(enabled=False),
            missing_fields=MissingFieldsConfig(enabled=False),
            corrupt_values=CorruptValuesConfig(enabled=False),
            timestamp_anomalies=TimestampAnomaliesConfig(enabled=False),
            duplicate_records=DuplicateRecordsConfig(enabled=False),
        )
        pipeline = AnomalyPipeline(config, rng)
        records = _make_many_records(20)
        result, stats = pipeline.apply(records)
        assert len(result) == 20
        assert stats.total == 0

    def test_pipeline_order_orphaned_before_duplicates(self, rng):
        """Orphaned runs first (removes records), duplicates last (adds records).

        With rate=1 for both, orphaned halves the pairs, then duplicates doubles
        the remaining records. This verifies ordering.
        """
        config = AnomaliesConfig(
            enabled=True,
            orphaned_records=OrphanedRecordsConfig(enabled=True, rate=1.0),
            missing_fields=MissingFieldsConfig(enabled=False),
            corrupt_values=CorruptValuesConfig(enabled=False),
            timestamp_anomalies=TimestampAnomaliesConfig(enabled=False),
            duplicate_records=DuplicateRecordsConfig(
                enabled=True, rate=1.0, max_time_shift_ms=500
            ),
        )
        pipeline = AnomalyPipeline(config, rng)
        pairs = _make_many_pairs(10)  # 20 records, 10 pairs
        result, stats = pipeline.apply(pairs)
        # Orphaned removes 10 records (one from each pair) => 10 remain
        assert stats.orphaned_count == 10
        # Duplicates doubles remaining => 10 duplicates added
        assert stats.duplicate_count == 10
        assert len(result) == 20  # 10 surviving + 10 duplicates

    def test_empty_input(self, rng, full_config):
        pipeline = AnomalyPipeline(full_config, rng)
        result, stats = pipeline.apply([])
        assert result == []
        assert stats.total == 0

    def test_deterministic_pipeline(self, full_config):
        records = _make_many_pairs(30) + _make_many_records(30)

        rng1 = np.random.default_rng(777)
        pipeline1 = AnomalyPipeline(full_config, rng1)
        result1, stats1 = pipeline1.apply(copy.deepcopy(records))

        rng2 = np.random.default_rng(777)
        pipeline2 = AnomalyPipeline(full_config, rng2)
        result2, stats2 = pipeline2.apply(copy.deepcopy(records))

        assert stats1.total == stats2.total
        assert len(result1) == len(result2)

    def test_single_record_input(self, rng, full_config):
        """Pipeline handles a single record without errors."""
        pipeline = AnomalyPipeline(full_config, rng)
        records = [_make_single_record()]
        result, stats = pipeline.apply(records)
        assert isinstance(result, list)
        assert isinstance(stats, AnomalyStats)

    def test_only_orphaned_enabled(self, rng):
        """When only orphaned is enabled, other stats stay zero."""
        config = AnomaliesConfig(
            enabled=True,
            orphaned_records=OrphanedRecordsConfig(enabled=True, rate=1.0),
            missing_fields=MissingFieldsConfig(enabled=False),
            corrupt_values=CorruptValuesConfig(enabled=False),
            timestamp_anomalies=TimestampAnomaliesConfig(enabled=False),
            duplicate_records=DuplicateRecordsConfig(enabled=False),
        )
        pipeline = AnomalyPipeline(config, rng)
        records = _make_many_pairs(5)
        result, stats = pipeline.apply(records)
        assert stats.orphaned_count == 5
        assert stats.missing_fields_count == 0
        assert stats.corrupt_values_count == 0
        assert stats.timestamp_anomalies_count == 0
        assert stats.duplicate_count == 0
