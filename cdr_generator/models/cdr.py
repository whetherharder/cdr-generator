"""CDR record model and CSV serialization utilities."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any


# Ordered list of CSV column names — defines the header row.
CDR_FIELDS: list[str] = [
    "record_type",
    "sequence_number",
    "consolidation_id",
    "charging_id",
    "served_imsi",
    "served_msisdn",
    "served_imei",
    "calling_number",
    "called_number",
    "redirecting_number",
    "event_timestamp",
    "answer_timestamp",
    "release_timestamp",
    "duration_seconds",
    "cause_for_termination",
    "first_cell_id",
    "last_cell_id",
    "serving_ne_id",
    "record_opening_time",
    "record_closure_time",
    "uplink_volume_bytes",
    "downlink_volume_bytes",
    "apn",
    "qci",
    "rat_type",
    "vendor_extensions",
]


@dataclass
class CDRRecord:
    """A single CDR record that maps 1-to-1 with a CSV row.

    Only ``record_type``, ``served_imsi``, ``event_timestamp``, and
    ``serving_ne_id`` are mandatory.  Every other field defaults to ``None``
    and will be serialized as an empty string in CSV output.
    """

    # --- mandatory ---
    record_type: str
    served_imsi: str
    event_timestamp: datetime
    serving_ne_id: str

    # --- identity ---
    sequence_number: int | None = None
    served_msisdn: str | None = None
    served_imei: str | None = None

    # --- parties ---
    calling_number: str | None = None
    called_number: str | None = None
    redirecting_number: str | None = None

    # --- timestamps ---
    answer_timestamp: datetime | None = None
    release_timestamp: datetime | None = None

    # --- duration / cause ---
    duration_seconds: float | None = None
    cause_for_termination: int | None = None

    # --- location ---
    first_cell_id: int | None = None
    last_cell_id: int | None = None

    # --- data session ---
    record_opening_time: datetime | None = None
    record_closure_time: datetime | None = None
    uplink_volume_bytes: int | None = None
    downlink_volume_bytes: int | None = None
    charging_id: int | None = None
    apn: str | None = None
    qci: int | None = None

    # --- misc ---
    rat_type: str | None = None
    consolidation_id: str | None = None
    vendor_extensions: str | None = None


def _format_value(value: Any) -> str:
    """Format a single field value for CSV output."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if isinstance(value, float):
        # Avoid trailing zeros but keep precision
        return f"{value:g}"
    return str(value)


def to_csv_row(record: CDRRecord) -> list[str]:
    """Convert a CDRRecord to a list of string values in CDR_FIELDS order."""
    return [_format_value(getattr(record, field)) for field in CDR_FIELDS]
