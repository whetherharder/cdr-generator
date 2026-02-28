"""Anomaly injection pipeline for CDR records.

Applies a sequence of anomaly stages to generated CDR records before
they are written to output files.  Each stage independently decides
whether to apply an anomaly to each record based on its configured rate.

Pipeline order (per plan.md)::

    orphaned -> missing_fields -> corrupt_values -> timestamp_anomalies -> duplicates

This order is intentional:
- Orphaned first: removes MT records, so subsequent stages process fewer records.
- Missing/corrupt/timestamps: mutate individual records.
- Duplicates last: clones records (including already-anomalied ones), which
  is realistic -- a duplicate in a real network would carry the same data
  quality issues as the original.

Usage::

    pipeline = AnomalyPipeline(config.anomalies, rng)
    records, stats = pipeline.apply(records)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cdr_generator.config.models import (
        AnomaliesConfig,
        CorruptValuesConfig,
        DuplicateRecordsConfig,
        MissingFieldsConfig,
        OrphanedRecordsConfig,
        TimestampAnomaliesConfig,
    )
    from cdr_generator.models.cdr import CDRRecord


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Record types that form pairs via consolidation_id
_VOICE_MO = "mo_call"
_VOICE_MT = "mt_call"
_SMS_MO = "mo_sms"
_SMS_MT = "mt_sms"

# Record types that form pairs via charging_id
_DATA_SGW = "sgw_data"
_DATA_PGW = "pgw_data"

# All "originating" and "terminating" types for pair identification
_MO_TYPES = {_VOICE_MO, _SMS_MO, _DATA_SGW}
_MT_TYPES = {_VOICE_MT, _SMS_MT, _DATA_PGW}

# Field name mapping: config name -> CDRRecord attribute name
_FIELD_MAP = {
    "cell_id": "first_cell_id",
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class AnomalyStats:
    """Counts of anomalies applied by each pipeline stage.

    Attributes
    ----------
    orphaned_count:
        Number of records removed (one side of a pair dropped).
    missing_fields_count:
        Number of records that had fields set to None.
    corrupt_values_count:
        Number of records with corrupted field values.
    timestamp_anomalies_count:
        Number of records with corrupted timestamps.
    duplicate_count:
        Number of duplicate records added.
    """

    orphaned_count: int = 0
    missing_fields_count: int = 0
    corrupt_values_count: int = 0
    timestamp_anomalies_count: int = 0
    duplicate_count: int = 0

    @property
    def total(self) -> int:
        """Total number of anomalies applied across all stages."""
        return (
            self.orphaned_count
            + self.missing_fields_count
            + self.corrupt_values_count
            + self.timestamp_anomalies_count
            + self.duplicate_count
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AnomalyPipeline:
    """Composes all anomaly stages into a single pipeline.

    Parameters
    ----------
    config:
        Top-level anomalies configuration (``config.anomalies``).
    rng:
        Numpy random generator for deterministic anomaly application.
    """

    def __init__(self, config: AnomaliesConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

    def apply(self, records: list[CDRRecord]) -> tuple[list[CDRRecord], AnomalyStats]:
        """Apply all enabled anomaly stages in order.

        Parameters
        ----------
        records:
            List of CDR records to process.  Records may be mutated
            in place (missing fields, corrupt values, timestamps).

        Returns
        -------
        tuple[list[CDRRecord], AnomalyStats]
            The (possibly modified) record list and anomaly statistics.
            The list may be shorter (orphaned removed records) or longer
            (duplicates added records) than the input.
        """
        stats = AnomalyStats()

        if not self._config.enabled:
            return records, stats

        # Stage 1: Orphaned records
        records, stats.orphaned_count = apply_orphaned(
            records, self._config.orphaned_records, self._rng
        )

        # Stage 2: Missing fields
        records, stats.missing_fields_count = apply_missing_fields(
            records, self._config.missing_fields, self._rng
        )

        # Stage 3: Corrupt values
        records, stats.corrupt_values_count = apply_corrupt_values(
            records, self._config.corrupt_values, self._rng
        )

        # Stage 4: Timestamp anomalies
        records, stats.timestamp_anomalies_count = apply_timestamp_anomalies(
            records, self._config.timestamp_anomalies, self._rng
        )

        # Stage 5: Duplicates
        records, stats.duplicate_count = apply_duplicates(
            records, self._config.duplicate_records, self._rng
        )

        return records, stats


# ---------------------------------------------------------------------------
# Stage 1: Orphaned Records
# ---------------------------------------------------------------------------


def apply_orphaned(
    records: list[CDRRecord],
    config: OrphanedRecordsConfig,
    rng: np.random.Generator,
) -> tuple[list[CDRRecord], int]:
    """Remove one side of MO/MT or SGW/PGW pairs to create orphaned records.

    Pairs are identified by:
    - ``consolidation_id`` for voice (mo_call/mt_call) and SMS (mo_sms/mt_sms)
    - ``charging_id`` for data (sgw_data/pgw_data)

    For each pair selected (at ``config.rate``), either the MO or MT
    record is removed with equal probability.

    Parameters
    ----------
    records:
        CDR records to process.
    config:
        Orphaned records configuration (rate, enabled).
    rng:
        Numpy RNG.

    Returns
    -------
    tuple[list[CDRRecord], int]
        The filtered record list and count of removed records.
    """
    if not config.enabled or config.rate == 0.0 or not records:
        return records, 0

    # Build pair groups: key -> list of record indices
    # Voice/SMS pairs: keyed by (consolidation_id, pair_type)
    # Data pairs: keyed by (charging_id, "data")
    pairs: dict[tuple, list[int]] = {}

    for idx, rec in enumerate(records):
        rt = rec.record_type
        if rt in (_VOICE_MO, _VOICE_MT) and rec.consolidation_id is not None:
            key = (rec.consolidation_id, "voice")
        elif rt in (_SMS_MO, _SMS_MT) and rec.consolidation_id is not None:
            key = (rec.consolidation_id, "sms")
        elif rt in (_DATA_SGW, _DATA_PGW) and rec.charging_id is not None:
            key = (rec.charging_id, "data")
        else:
            continue
        pairs.setdefault(key, []).append(idx)

    # Identify actual pairs (groups with exactly 2 records)
    remove_indices: set[int] = set()
    for key, indices in pairs.items():
        if len(indices) != 2:
            continue
        # Decide whether to orphan this pair
        if rng.random() < config.rate:
            # Remove one side randomly (equal probability)
            victim = int(rng.integers(0, 2))
            remove_indices.add(indices[victim])

    count = len(remove_indices)
    if count == 0:
        return records, 0

    result = [rec for idx, rec in enumerate(records) if idx not in remove_indices]
    return result, count


# ---------------------------------------------------------------------------
# Stage 2: Missing Fields
# ---------------------------------------------------------------------------


def apply_missing_fields(
    records: list[CDRRecord],
    config: MissingFieldsConfig,
    rng: np.random.Generator,
) -> tuple[list[CDRRecord], int]:
    """Set target fields to None on selected records.

    For each record selected (at ``config.rate``), one field from
    ``config.target_fields`` is randomly chosen and set to ``None``.

    The target fields from the reference config are:
    ``served_msisdn``, ``cell_id``, ``duration_seconds``,
    ``cause_for_termination``.

    Note: ``cell_id`` in config maps to ``first_cell_id`` on CDRRecord.

    Parameters
    ----------
    records:
        CDR records to process (mutated in place).
    config:
        Missing fields configuration (rate, target_fields, enabled).
    rng:
        Numpy RNG.

    Returns
    -------
    tuple[list[CDRRecord], int]
        The same record list (unchanged length) and count of affected records.
    """
    if (
        not config.enabled
        or config.rate == 0.0
        or not records
        or not config.target_fields
    ):
        return records, 0

    # Pre-map config field names to CDRRecord attribute names
    attr_names = [_FIELD_MAP.get(f, f) for f in config.target_fields]
    n_fields = len(attr_names)

    count = 0
    for rec in records:
        if rng.random() < config.rate:
            # Pick a random field to null
            field_idx = int(rng.integers(0, n_fields))
            attr = attr_names[field_idx]
            setattr(rec, attr, None)
            count += 1

    return records, count


# ---------------------------------------------------------------------------
# Stage 3: Corrupt Values
# ---------------------------------------------------------------------------


def apply_corrupt_values(
    records: list[CDRRecord],
    config: CorruptValuesConfig,
    rng: np.random.Generator,
) -> tuple[list[CDRRecord], int]:
    """Replace field values with invalid/impossible data.

    Corruption types (from ``config.types``):
    - ``invalid_imsi``: Replace ``served_imsi`` with a malformed string
      (e.g., non-numeric, wrong length).
    - ``impossible_cell_id``: Replace ``first_cell_id`` with a cell_id
      that does not exist in the network (e.g., 9999999).
    - ``extreme_duration``: Set ``duration_seconds`` to an impossibly
      large value (e.g., 999999.0).

    For each record selected (at ``config.rate``), one corruption type
    is randomly chosen and applied.

    Parameters
    ----------
    records:
        CDR records to process (mutated in place).
    config:
        Corrupt values configuration (rate, types, enabled).
    rng:
        Numpy RNG.

    Returns
    -------
    tuple[list[CDRRecord], int]
        The same record list and count of corrupted records.
    """
    if not config.enabled or config.rate == 0.0 or not records or not config.types:
        return records, 0

    n_types = len(config.types)
    count = 0
    for rec in records:
        if rng.random() < config.rate:
            corruption = config.types[int(rng.integers(0, n_types))]
            if corruption == "invalid_imsi":
                rec.served_imsi = "INVALID_IMSI"
            elif corruption == "impossible_cell_id":
                rec.first_cell_id = 9999999
            elif corruption == "extreme_duration":
                rec.duration_seconds = 999999.0
            count += 1

    return records, count


# ---------------------------------------------------------------------------
# Stage 4: Timestamp Anomalies
# ---------------------------------------------------------------------------


def apply_timestamp_anomalies(
    records: list[CDRRecord],
    config: TimestampAnomaliesConfig,
    rng: np.random.Generator,
) -> tuple[list[CDRRecord], int]:
    """Corrupt timestamps on selected records.

    Anomaly types (from ``config.types``):
    - ``future_timestamp``: Set ``event_timestamp`` to a time in the
      future (e.g., +1 to +30 days from original).
    - ``negative_duration``: Set ``duration_seconds`` to a negative
      value, or set ``release_timestamp`` before ``answer_timestamp``.
    - ``midnight_rollover``: Shift ``event_timestamp`` to just before
      midnight so that ``release_timestamp`` falls on the next day,
      creating a cross-day record.

    For each record selected (at ``config.rate``), one anomaly type
    is randomly chosen and applied.

    Parameters
    ----------
    records:
        CDR records to process (mutated in place).
    config:
        Timestamp anomalies configuration (rate, types, enabled).
    rng:
        Numpy RNG.

    Returns
    -------
    tuple[list[CDRRecord], int]
        The same record list and count of affected records.
    """
    if not config.enabled or config.rate == 0.0 or not records or not config.types:
        return records, 0

    n_types = len(config.types)
    count = 0
    for rec in records:
        if rng.random() < config.rate:
            anomaly = config.types[int(rng.integers(0, n_types))]
            if anomaly == "future_timestamp":
                _apply_future_timestamp(rec, rng)
            elif anomaly == "negative_duration":
                _apply_negative_duration(rec, rng)
            elif anomaly == "midnight_rollover":
                _apply_midnight_rollover(rec, rng)
            count += 1

    return records, count


def _apply_future_timestamp(rec: CDRRecord, rng: np.random.Generator) -> None:
    """Shift event_timestamp 1-30 days into the future."""
    days = int(rng.integers(1, 31))  # 1 to 30 inclusive
    rec.event_timestamp = rec.event_timestamp + timedelta(days=days)


def _apply_negative_duration(rec: CDRRecord, rng: np.random.Generator) -> None:
    """Set duration_seconds negative, or set release_timestamp before answer_timestamp."""
    if rec.answer_timestamp is not None and rec.release_timestamp is not None:
        # Swap release and answer so release < answer
        rec.release_timestamp, rec.answer_timestamp = (
            rec.answer_timestamp,
            rec.release_timestamp,
        )
        # Also make duration negative for consistency
        if rec.duration_seconds is not None:
            rec.duration_seconds = -abs(rec.duration_seconds)
    elif rec.duration_seconds is not None:
        rec.duration_seconds = -abs(rec.duration_seconds)
    else:
        # Fallback: set duration to a negative value
        rec.duration_seconds = -60.0


def _apply_midnight_rollover(rec: CDRRecord, rng: np.random.Generator) -> None:
    """Shift event_timestamp to 23:59:xx so release crosses midnight."""
    ts = rec.event_timestamp
    # Calculate the duration so we can set the release properly
    duration = timedelta(seconds=60)  # default 1 minute
    if rec.duration_seconds is not None and rec.duration_seconds > 0:
        duration = timedelta(seconds=rec.duration_seconds)
    elif rec.release_timestamp is not None:
        duration = rec.release_timestamp - ts

    # Set event_timestamp to 23:59:xx on the same day
    seconds_into_last_minute = int(rng.integers(0, 50))  # 0-49 seconds into 23:59
    new_ts = ts.replace(
        hour=23, minute=59, second=seconds_into_last_minute, microsecond=0
    )
    rec.event_timestamp = new_ts

    # Set release_timestamp to cross midnight
    if rec.release_timestamp is not None:
        rec.release_timestamp = new_ts + duration
    if rec.answer_timestamp is not None:
        rec.answer_timestamp = new_ts + timedelta(seconds=3)


# ---------------------------------------------------------------------------
# Stage 5: Duplicate Records
# ---------------------------------------------------------------------------


def apply_duplicates(
    records: list[CDRRecord],
    config: DuplicateRecordsConfig,
    rng: np.random.Generator,
) -> tuple[list[CDRRecord], int]:
    """Clone selected records with jittered timestamps.

    For each record selected (at ``config.rate``), a shallow copy is
    created with ``event_timestamp`` shifted by a random amount within
    ``[-max_time_shift_ms, +max_time_shift_ms]``.  The duplicate is
    appended to the record list.

    Parameters
    ----------
    records:
        CDR records to process.  New records are appended.
    config:
        Duplicate records configuration (rate, max_time_shift_ms, enabled).
    rng:
        Numpy RNG.

    Returns
    -------
    tuple[list[CDRRecord], int]
        The extended record list and count of duplicates added.
    """
    if not config.enabled or config.rate == 0.0 or not records:
        return records, 0

    duplicates: list[CDRRecord] = []
    max_shift = config.max_time_shift_ms

    for rec in records:
        if rng.random() < config.rate:
            dup = copy.copy(rec)
            # Jitter timestamp by [-max_shift, +max_shift] milliseconds
            shift_ms = int(rng.integers(-max_shift, max_shift + 1))
            dup.event_timestamp = rec.event_timestamp + timedelta(milliseconds=shift_ms)
            duplicates.append(dup)

    count = len(duplicates)
    if count > 0:
        records = records + duplicates

    return records, count
