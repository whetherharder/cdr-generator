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

from dataclasses import dataclass, field
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
        ...

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
        ...


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
    ...


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
    ...


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
    ...


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
    ...


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
    ...
