"""Data session CDR generator -- SGW/PGW paired records with partial record support."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta

import numpy as np

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber
from cdr_generator.generators.voice import (
    _sample_distribution,
    _weighted_choice,
)
from cdr_generator.models.cdr import CDRRecord


def generate_data_cdr(
    subscriber: Subscriber,
    event_time: datetime,
    cell: Cell,
    sgw: NetworkElement,
    pgw: NetworkElement,
    data_cfg: dict,
    rng: np.random.Generator,
) -> list[CDRRecord]:
    """Generate data session CDR records (SGW + PGW pairs).

    Parameters
    ----------
    subscriber:
        Subscriber initiating the data session.
    event_time:
        Timestamp of the session start.
    cell:
        Cell where the subscriber is located.
    sgw:
        Serving Gateway network element.
    pgw:
        PDN Gateway network element.
    data_cfg:
        Data event configuration dict (from config.events.data).
    rng:
        Numpy random generator for deterministic sampling.

    Returns
    -------
    list[CDRRecord]
        SGW + PGW record pairs. Long sessions produce multiple partial
        records sharing the same charging_id.
    """
    charging_id = int(rng.integers(1, 2**31))

    # Sample session duration
    duration = _sample_data_duration(data_cfg, rng)

    # Sample total volumes
    uplink_bytes = _sample_volume(data_cfg["volume_uplink"], rng)
    downlink_bytes = _sample_volume(data_cfg["volume_downlink"], rng)

    # Pick APN
    apn = _pick_apn(data_cfg, rng)

    # Pick QoS/QCI
    qci = _pick_qci(data_cfg, rng)

    # Pick termination cause
    termination_cause = _pick_termination_cause(data_cfg, rng)

    # Determine partial records
    partial_cfg = data_cfg.get("partial_records", {})
    partial_enabled = partial_cfg.get("enabled", True)
    max_duration = partial_cfg.get("max_record_duration_seconds", 3600)
    max_volume = partial_cfg.get("max_record_volume_bytes", 104857600)

    if partial_enabled and duration > max_duration:
        return _generate_partial_records(
            subscriber=subscriber,
            event_time=event_time,
            cell=cell,
            sgw=sgw,
            pgw=pgw,
            charging_id=charging_id,
            total_duration=duration,
            total_uplink=uplink_bytes,
            total_downlink=downlink_bytes,
            max_duration=max_duration,
            max_volume=max_volume,
            apn=apn,
            qci=qci,
            termination_cause=termination_cause,
            rng=rng,
        )

    # Single SGW + PGW pair
    return _create_sgw_pgw_pair(
        subscriber=subscriber,
        event_time=event_time,
        cell=cell,
        sgw=sgw,
        pgw=pgw,
        charging_id=charging_id,
        duration=duration,
        uplink_bytes=uplink_bytes,
        downlink_bytes=downlink_bytes,
        apn=apn,
        qci=qci,
        termination_cause=termination_cause,
    )


def _generate_partial_records(
    subscriber: Subscriber,
    event_time: datetime,
    cell: Cell,
    sgw: NetworkElement,
    pgw: NetworkElement,
    charging_id: int,
    total_duration: float,
    total_uplink: int,
    total_downlink: int,
    max_duration: int,
    max_volume: int,
    apn: str,
    qci: int,
    termination_cause: str,
    rng: np.random.Generator,
) -> list[CDRRecord]:
    """Split a long session into multiple partial records."""
    num_parts = max(1, math.ceil(total_duration / max_duration))

    # Distribute volume proportionally across parts
    part_duration = total_duration / num_parts
    part_uplink = total_uplink // num_parts
    part_downlink = total_downlink // num_parts

    records: list[CDRRecord] = []
    current_time = event_time

    for i in range(num_parts):
        # Last part gets any remaining bytes
        if i == num_parts - 1:
            ul = total_uplink - part_uplink * (num_parts - 1)
            dl = total_downlink - part_downlink * (num_parts - 1)
        else:
            ul = part_uplink
            dl = part_downlink

        pair = _create_sgw_pgw_pair(
            subscriber=subscriber,
            event_time=current_time,
            cell=cell,
            sgw=sgw,
            pgw=pgw,
            charging_id=charging_id,
            duration=part_duration,
            uplink_bytes=ul,
            downlink_bytes=dl,
            apn=apn,
            qci=qci,
            termination_cause=termination_cause,
            sequence_number=i + 1,
        )
        records.extend(pair)
        current_time += timedelta(seconds=part_duration)

    return records


def _create_sgw_pgw_pair(
    subscriber: Subscriber,
    event_time: datetime,
    cell: Cell,
    sgw: NetworkElement,
    pgw: NetworkElement,
    charging_id: int,
    duration: float,
    uplink_bytes: int,
    downlink_bytes: int,
    apn: str,
    qci: int,
    termination_cause: str,
    sequence_number: int | None = None,
) -> list[CDRRecord]:
    """Create one SGW + one PGW CDR record pair."""
    opening_time = event_time
    closure_time = event_time + timedelta(seconds=duration)

    sgw_record = CDRRecord(
        record_type="sgw_data",
        served_imsi=subscriber.imsi,
        served_msisdn=subscriber.msisdn,
        served_imei=subscriber.imei,
        event_timestamp=event_time,
        record_opening_time=opening_time,
        record_closure_time=closure_time,
        duration_seconds=duration,
        uplink_volume_bytes=uplink_bytes,
        downlink_volume_bytes=downlink_bytes,
        charging_id=charging_id,
        apn=apn,
        qci=qci,
        first_cell_id=cell.cell_id,
        last_cell_id=cell.cell_id,
        serving_ne_id=sgw.id,
        cause_for_termination=None,
        sequence_number=sequence_number,
        rat_type="eutran",
    )

    pgw_record = CDRRecord(
        record_type="pgw_data",
        served_imsi=subscriber.imsi,
        served_msisdn=subscriber.msisdn,
        served_imei=subscriber.imei,
        event_timestamp=event_time,
        record_opening_time=opening_time,
        record_closure_time=closure_time,
        duration_seconds=duration,
        uplink_volume_bytes=uplink_bytes,
        downlink_volume_bytes=downlink_bytes,
        charging_id=charging_id,
        apn=apn,
        qci=qci,
        first_cell_id=cell.cell_id,
        last_cell_id=cell.cell_id,
        serving_ne_id=pgw.id,
        cause_for_termination=None,
        sequence_number=sequence_number,
        rat_type="eutran",
    )

    return [sgw_record, pgw_record]


def _sample_data_duration(data_cfg: dict, rng: np.random.Generator) -> float:
    """Sample data session duration from the configured distribution."""
    dur_cfg = data_cfg["duration"]
    dist = dur_cfg["distribution"]
    raw = _sample_distribution(dist, rng)
    min_s = dur_cfg.get("min_seconds", 5)
    max_s = dur_cfg.get("max_seconds", 86400)
    return float(max(min_s, min(raw, max_s)))


def _sample_volume(vol_cfg: dict, rng: np.random.Generator) -> int:
    """Sample a volume in bytes from the configured distribution."""
    dist = vol_cfg["distribution"]
    raw = _sample_distribution(dist, rng)
    min_bytes = vol_cfg.get("min_bytes", 100)
    return max(min_bytes, int(raw))


def _pick_apn(data_cfg: dict, rng: np.random.Generator) -> str:
    """Pick an APN from weighted distribution."""
    apn_weights = data_cfg.get("apn_weights", {"internet": 1.0})
    if not apn_weights:
        return "internet"

    apns = list(apn_weights.keys())
    weights = list(apn_weights.values())
    idx = _weighted_choice(weights, rng)
    return apns[idx]


def _pick_qci(data_cfg: dict, rng: np.random.Generator) -> int:
    """Pick a QCI from the QoS distribution."""
    qos_dist = data_cfg.get("qos_distribution", [])
    if not qos_dist:
        return 9  # default bearer

    weights = [q["weight"] for q in qos_dist]
    idx = _weighted_choice(weights, rng)
    return qos_dist[idx]["qci"]


def _pick_termination_cause(data_cfg: dict, rng: np.random.Generator) -> str:
    """Pick a data session termination cause from weighted list."""
    causes = data_cfg.get("termination_causes", [])
    if not causes:
        return "normal_release"

    weights = [c["weight"] for c in causes]
    idx = _weighted_choice(weights, rng)
    return causes[idx]["cause"]
