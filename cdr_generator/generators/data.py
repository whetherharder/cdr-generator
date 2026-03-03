"""Data session CDR generator -- SGW/PGW paired records with partial record support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber
from cdr_generator.generators.voice import (
    _sample_distribution,
    _weighted_choice,
)
from cdr_generator.models.cdr import CDRRecord

if TYPE_CHECKING:
    from cdr_generator.engine.rng_buffer import _RngBuffer


@dataclass(slots=True)
class _DataCfgCache:
    """Pre-computed static config for zero-dict-lookup hot path.

    Build once with ``build_data_cfg_cache(data_cfg)`` and pass as
    ``_cfg`` to ``generate_data_cdr`` to bypass all dict.get calls
    in the inner generation loop.
    """

    dur_mu: float
    dur_sigma: float
    min_dur: float
    max_dur: float
    ul_mu: float
    ul_sigma: float
    ul_min: int
    dl_mu: float
    dl_sigma: float
    dl_min: int
    ul_mult: float
    dl_mult: float
    apn_list: list
    apn_probs: tuple
    qci_list: list
    qci_probs: tuple
    term_codes: list
    term_probs: tuple
    partial_enabled: bool
    max_record_duration: float
    max_record_volume: int


def build_data_cfg_cache(data_cfg: dict) -> _DataCfgCache:
    """Pre-extract all static config values from data_cfg into a fast-access object."""
    dur_cfg = data_cfg.get("duration", {})
    dur_dist = dur_cfg.get("distribution", {})
    dur_params = dur_dist.get("params", {})

    vol_ul_cfg = data_cfg.get("volume_uplink", {})
    ul_dist = vol_ul_cfg.get("distribution", {})
    ul_params = ul_dist.get("params", {})

    vol_dl_cfg = data_cfg.get("volume_downlink", {})
    dl_dist = vol_dl_cfg.get("distribution", {})
    dl_params = dl_dist.get("params", {})

    # APN
    apn_weights = data_cfg.get("apn_weights", {"internet": 1.0}) or {"internet": 1.0}
    apn_list = list(apn_weights.keys())
    raw_apn_w = list(apn_weights.values())
    total_apn = sum(raw_apn_w) or 1.0
    apn_probs = tuple(w / total_apn for w in raw_apn_w)

    # QCI
    qos_dist = data_cfg.get("qos_distribution", [])
    if qos_dist:
        qci_list = [q["qci"] for q in qos_dist]
        raw_qci_w = [q["weight"] for q in qos_dist]
        total_qci = sum(raw_qci_w) or 1.0
        qci_probs = tuple(w / total_qci for w in raw_qci_w)
    else:
        qci_list = [9]
        qci_probs = (1.0,)

    # Termination causes
    causes = data_cfg.get("termination_causes", [])
    if causes:
        term_codes = [c.get("code") for c in causes]
        raw_term_w = [c["weight"] for c in causes]
        total_term = sum(raw_term_w) or 1.0
        term_probs = tuple(w / total_term for w in raw_term_w)
    else:
        term_codes = [None]
        term_probs = (1.0,)

    # Partial records
    partial_cfg = data_cfg.get("partial_records", {})

    return _DataCfgCache(
        dur_mu=float(dur_params.get("mu", 0.0)),
        dur_sigma=float(dur_params.get("sigma", 1.0)),
        min_dur=float(dur_cfg.get("min_seconds", 5)),
        max_dur=float(dur_cfg.get("max_seconds", 86400)),
        ul_mu=float(ul_params.get("mu", 0.0)),
        ul_sigma=float(ul_params.get("sigma", 1.0)),
        ul_min=int(vol_ul_cfg.get("min_bytes", 100)),
        dl_mu=float(dl_params.get("mu", 0.0)),
        dl_sigma=float(dl_params.get("sigma", 1.0)),
        dl_min=int(vol_dl_cfg.get("min_bytes", 100)),
        ul_mult=float(data_cfg.get("_volume_multiplier_uplink", 1.0)),
        dl_mult=float(data_cfg.get("_volume_multiplier_downlink", 1.0)),
        apn_list=apn_list,
        apn_probs=apn_probs,
        qci_list=qci_list,
        qci_probs=qci_probs,
        term_codes=term_codes,
        term_probs=term_probs,
        partial_enabled=bool(partial_cfg.get("enabled", True)),
        max_record_duration=float(partial_cfg.get("max_record_duration_seconds", 3600)),
        max_record_volume=int(partial_cfg.get("max_record_volume_bytes", 104857600)),
    )


def generate_data_cdr(
    subscriber: Subscriber,
    event_time: datetime,
    cell: Cell,
    sgw: NetworkElement,
    pgw: NetworkElement,
    data_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
    _cfg: _DataCfgCache | None = None,
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
    _buf:
        Optional pre-filled RNG buffer for high-throughput generation.
    _cfg:
        Optional pre-computed config cache (build with
        ``build_data_cfg_cache``).  When provided together with ``_buf``,
        all dict lookups and weight normalizations are bypassed.

    Returns
    -------
    list[CDRRecord]
        SGW + PGW record pairs. Long sessions produce multiple partial
        records sharing the same charging_id.
    """
    if _cfg is not None and _buf is not None:
        # FAST PATH: pre-computed values + buffer-backed RNG.
        # Zero dict.get calls, zero weight normalization per event.
        charging_id = _buf.get_int(1, 2**31)

        raw_dur = _buf.get_lognormal(_cfg.dur_mu, _cfg.dur_sigma)
        duration = float(max(_cfg.min_dur, min(raw_dur, _cfg.max_dur)))

        raw_ul = _buf.get_lognormal(_cfg.ul_mu, _cfg.ul_sigma)
        uplink_bytes = max(_cfg.ul_min, int(raw_ul))
        if _cfg.ul_mult != 1.0:
            uplink_bytes = max(1, int(uplink_bytes * _cfg.ul_mult))

        raw_dl = _buf.get_lognormal(_cfg.dl_mu, _cfg.dl_sigma)
        downlink_bytes = max(_cfg.dl_min, int(raw_dl))
        if _cfg.dl_mult != 1.0:
            downlink_bytes = max(1, int(downlink_bytes * _cfg.dl_mult))

        apn = _cfg.apn_list[_buf.get_choice(len(_cfg.apn_list), _cfg.apn_probs)]
        qci = _cfg.qci_list[_buf.get_choice(len(_cfg.qci_list), _cfg.qci_probs)]
        termination_cause = _cfg.term_codes[
            _buf.get_choice(len(_cfg.term_codes), _cfg.term_probs)
        ]

        if _cfg.partial_enabled and duration > _cfg.max_record_duration:
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
                max_duration=int(_cfg.max_record_duration),
                max_volume=_cfg.max_record_volume,
                apn=apn,
                qci=qci,
                termination_cause=termination_cause,
                rng=rng,
            )
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

    # SLOW PATH: dict-based config (backward-compatible).
    if _buf is not None:
        charging_id = _buf.get_int(1, 2**31)
    else:
        charging_id = int(rng.integers(1, 2**31))

    # Sample session duration
    duration = _sample_data_duration(data_cfg, rng, _buf=_buf)

    # Sample total volumes and apply profile volume multipliers
    uplink_bytes = _sample_volume(data_cfg["volume_uplink"], rng, _buf=_buf)
    downlink_bytes = _sample_volume(data_cfg["volume_downlink"], rng, _buf=_buf)

    ul_mult = data_cfg.get("_volume_multiplier_uplink", 1.0)
    dl_mult = data_cfg.get("_volume_multiplier_downlink", 1.0)
    if ul_mult != 1.0:
        uplink_bytes = max(1, int(uplink_bytes * ul_mult))
    if dl_mult != 1.0:
        downlink_bytes = max(1, int(downlink_bytes * dl_mult))

    # Pick APN
    apn = _pick_apn(data_cfg, rng, _buf=_buf)

    # Pick QoS/QCI
    qci = _pick_qci(data_cfg, rng, _buf=_buf)

    # Pick termination cause
    termination_cause = _pick_termination_cause(data_cfg, rng, _buf=_buf)

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
    termination_cause: int | None,
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
    termination_cause: int | None,
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
        cause_for_termination=termination_cause,
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
        cause_for_termination=termination_cause,
        sequence_number=sequence_number,
        rat_type="eutran",
    )

    return [sgw_record, pgw_record]


def _sample_data_duration(
    data_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> float:
    """Sample data session duration from the configured distribution."""
    dur_cfg = data_cfg["duration"]
    dist = dur_cfg["distribution"]
    raw = _sample_distribution(dist, rng, _buf=_buf)
    min_s = dur_cfg.get("min_seconds", 5)
    max_s = dur_cfg.get("max_seconds", 86400)
    return float(max(min_s, min(raw, max_s)))


def _sample_volume(
    vol_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int:
    """Sample a volume in bytes from the configured distribution."""
    dist = vol_cfg["distribution"]
    raw = _sample_distribution(dist, rng, _buf=_buf)
    min_bytes = vol_cfg.get("min_bytes", 100)
    return max(min_bytes, int(raw))


def _pick_apn(
    data_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> str:
    """Pick an APN from weighted distribution."""
    apn_weights = data_cfg.get("apn_weights", {"internet": 1.0})
    if not apn_weights:
        return "internet"

    apns = list(apn_weights.keys())
    weights = list(apn_weights.values())
    idx = _weighted_choice(weights, rng, _buf=_buf)
    return apns[idx]


def _pick_qci(
    data_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int:
    """Pick a QCI from the QoS distribution."""
    qos_dist = data_cfg.get("qos_distribution", [])
    if not qos_dist:
        return 9  # default bearer

    weights = [q["weight"] for q in qos_dist]
    idx = _weighted_choice(weights, rng, _buf=_buf)
    return qos_dist[idx]["qci"]


def _pick_termination_cause(
    data_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int | None:
    """Pick a data session termination cause code from weighted list."""
    causes = data_cfg.get("termination_causes", [])
    if not causes:
        return None

    weights = [c["weight"] for c in causes]
    idx = _weighted_choice(weights, rng, _buf=_buf)
    return causes[idx].get("code")
