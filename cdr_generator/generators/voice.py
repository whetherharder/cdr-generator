"""Voice CDR generator -- MO/MT call pairs with success/failure handling."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber
from cdr_generator.models.cdr import CDRRecord


def generate_voice_cdr(
    caller: Subscriber,
    callee: Subscriber,
    event_time: datetime,
    cell: Cell,
    msc: NetworkElement,
    voice_cfg: dict,
    rng: np.random.Generator,
    forward_target: Subscriber | None = None,
) -> list[CDRRecord]:
    """Generate voice CDR records for a single call attempt.

    Parameters
    ----------
    caller:
        A-party subscriber initiating the call.
    callee:
        B-party subscriber (or external number holder).
    event_time:
        Timestamp of the call attempt.
    cell:
        Cell where the caller is located.
    msc:
        MSC network element handling the call.
    voice_cfg:
        Voice event configuration dict (from config.events.voice).
    rng:
        Numpy random generator for deterministic sampling.
    forward_target:
        Optional C-party subscriber for call forwarding. If None,
        forwarding cannot occur even if call_forwarding_rate > 0.

    Returns
    -------
    list[CDRRecord]
        One record (MO only) for failed calls, two records (MO + MT) for
        successful non-forwarded calls, three records (MO + MT-fwd + MT-final)
        for forwarded calls. All records share a consolidation_id.
    """
    success_rate = voice_cfg.get("success_rate", 0.85)
    is_success = rng.random() < success_rate

    consolidation_id = uuid.UUID(
        bytes=bytes(rng.integers(0, 256, size=16, dtype="uint8"))
    ).hex

    if not is_success:
        return _generate_failed_call(
            caller,
            callee,
            event_time,
            cell,
            msc,
            voice_cfg,
            rng,
            consolidation_id,
        )

    # Check for call forwarding (only on successful calls with a target)
    forwarding_rate = voice_cfg.get("call_forwarding_rate", 0.03)
    is_forwarded = forward_target is not None and rng.random() < forwarding_rate

    if is_forwarded:
        return _generate_forwarded_call(
            caller,
            callee,
            forward_target,
            event_time,
            cell,
            msc,
            voice_cfg,
            rng,
            consolidation_id,
        )

    return _generate_successful_call(
        caller,
        callee,
        event_time,
        cell,
        msc,
        voice_cfg,
        rng,
        consolidation_id,
    )


def _generate_failed_call(
    caller: Subscriber,
    callee: Subscriber,
    event_time: datetime,
    cell: Cell,
    msc: NetworkElement,
    voice_cfg: dict,
    rng: np.random.Generator,
    consolidation_id: str,
) -> list[CDRRecord]:
    """Generate a single MO CDR for a failed call attempt."""
    cause_code = _pick_failure_cause(voice_cfg, rng)

    mo = CDRRecord(
        record_type="mo_call",
        served_imsi=caller.imsi,
        served_msisdn=caller.msisdn,
        served_imei=caller.imei,
        event_timestamp=event_time,
        calling_number=caller.msisdn,
        called_number=callee.msisdn,
        duration_seconds=0.0,
        cause_for_termination=cause_code,
        first_cell_id=cell.cell_id,
        last_cell_id=cell.cell_id,
        serving_ne_id=msc.id,
        consolidation_id=consolidation_id,
        rat_type="eutran",
    )
    return [mo]


def _generate_successful_call(
    caller: Subscriber,
    callee: Subscriber,
    event_time: datetime,
    cell: Cell,
    msc: NetworkElement,
    voice_cfg: dict,
    rng: np.random.Generator,
    consolidation_id: str,
) -> list[CDRRecord]:
    """Generate MO + MT CDR pair for a successful call."""
    duration = _sample_duration(voice_cfg, rng)
    cause_code = _pick_normal_termination(voice_cfg, rng)
    jitter_ms = voice_cfg.get("paired_timestamp_jitter_ms", 1000)

    # MT event_time has a small jitter relative to MO
    mt_jitter = timedelta(milliseconds=int(rng.integers(0, max(1, jitter_ms))))
    mt_event_time = event_time + mt_jitter

    answer_time = event_time + timedelta(seconds=1)
    release_time = answer_time + timedelta(seconds=duration)

    mo = CDRRecord(
        record_type="mo_call",
        served_imsi=caller.imsi,
        served_msisdn=caller.msisdn,
        served_imei=caller.imei,
        event_timestamp=event_time,
        answer_timestamp=answer_time,
        release_timestamp=release_time,
        calling_number=caller.msisdn,
        called_number=callee.msisdn,
        duration_seconds=duration,
        cause_for_termination=cause_code,
        first_cell_id=cell.cell_id,
        last_cell_id=cell.cell_id,
        serving_ne_id=msc.id,
        consolidation_id=consolidation_id,
        rat_type="eutran",
    )

    mt = CDRRecord(
        record_type="mt_call",
        served_imsi=callee.imsi,
        served_msisdn=callee.msisdn,
        served_imei=callee.imei,
        event_timestamp=mt_event_time,
        answer_timestamp=answer_time + mt_jitter,
        release_timestamp=release_time + mt_jitter,
        calling_number=caller.msisdn,
        called_number=callee.msisdn,
        duration_seconds=duration,
        cause_for_termination=cause_code,
        first_cell_id=callee.home_cell_id,
        last_cell_id=callee.home_cell_id,
        serving_ne_id=msc.id,
        consolidation_id=consolidation_id,
        rat_type="eutran",
    )

    return [mo, mt]


def _generate_forwarded_call(
    caller: Subscriber,
    callee: Subscriber,
    forward_target: Subscriber,
    event_time: datetime,
    cell: Cell,
    msc: NetworkElement,
    voice_cfg: dict,
    rng: np.random.Generator,
    consolidation_id: str,
) -> list[CDRRecord]:
    """Generate MO + MT-forwarding + MT-final CDR triple for a forwarded call.

    A forwarded call produces three CDRs:
    1. MO (A->B): caller initiates, same as normal MO.
    2. MT forwarding (B): served by callee, with redirecting_number=C.
    3. MT final (C): served by forward_target, the actual answerer.

    All three share the same consolidation_id.
    """
    duration = _sample_duration(voice_cfg, rng)
    cause_code = _pick_normal_termination(voice_cfg, rng)
    jitter_ms = voice_cfg.get("paired_timestamp_jitter_ms", 1000)

    mt_jitter = timedelta(milliseconds=int(rng.integers(0, max(1, jitter_ms))))
    mt_event_time = event_time + mt_jitter

    answer_time = event_time + timedelta(seconds=1)
    release_time = answer_time + timedelta(seconds=duration)

    # 1. MO record: A calls B (same as normal MO)
    mo = CDRRecord(
        record_type="mo_call",
        served_imsi=caller.imsi,
        served_msisdn=caller.msisdn,
        served_imei=caller.imei,
        event_timestamp=event_time,
        answer_timestamp=answer_time,
        release_timestamp=release_time,
        calling_number=caller.msisdn,
        called_number=callee.msisdn,
        duration_seconds=duration,
        cause_for_termination=cause_code,
        first_cell_id=cell.cell_id,
        last_cell_id=cell.cell_id,
        serving_ne_id=msc.id,
        consolidation_id=consolidation_id,
        rat_type="eutran",
    )

    # 2. MT forwarding record: served by B, redirecting to C
    mt_fwd = CDRRecord(
        record_type="mt_call",
        served_imsi=callee.imsi,
        served_msisdn=callee.msisdn,
        served_imei=callee.imei,
        event_timestamp=mt_event_time,
        answer_timestamp=answer_time + mt_jitter,
        release_timestamp=release_time + mt_jitter,
        calling_number=caller.msisdn,
        called_number=callee.msisdn,
        duration_seconds=duration,
        cause_for_termination=cause_code,
        first_cell_id=callee.home_cell_id,
        last_cell_id=callee.home_cell_id,
        serving_ne_id=msc.id,
        consolidation_id=consolidation_id,
        redirecting_number=forward_target.msisdn,
        rat_type="eutran",
    )

    # 3. MT final record: served by C (forward target)
    mt_final_jitter = timedelta(milliseconds=int(rng.integers(0, max(1, jitter_ms))))
    mt_final_event_time = event_time + mt_final_jitter

    mt_final = CDRRecord(
        record_type="mt_call",
        served_imsi=forward_target.imsi,
        served_msisdn=forward_target.msisdn,
        served_imei=forward_target.imei,
        event_timestamp=mt_final_event_time,
        answer_timestamp=answer_time + mt_final_jitter,
        release_timestamp=release_time + mt_final_jitter,
        calling_number=caller.msisdn,
        called_number=forward_target.msisdn,
        duration_seconds=duration,
        cause_for_termination=cause_code,
        first_cell_id=forward_target.home_cell_id,
        last_cell_id=forward_target.home_cell_id,
        serving_ne_id=msc.id,
        consolidation_id=consolidation_id,
        rat_type="eutran",
    )

    return [mo, mt_fwd, mt_final]


def _sample_duration(voice_cfg: dict, rng: np.random.Generator) -> float:
    """Sample call duration from the configured distribution."""
    dur_cfg = voice_cfg["duration"]
    dist = dur_cfg["distribution"]

    raw = _sample_distribution(dist, rng)

    min_s = dur_cfg.get("min_seconds", 1)
    max_s = dur_cfg.get("max_seconds", 7200)
    return float(max(min_s, min(raw, max_s)))


def _sample_distribution(dist: dict, rng: np.random.Generator) -> float:
    """Sample a single value from a distribution descriptor."""
    dist_type = dist["type"]
    params = dist.get("params", {})

    if dist_type == "lognormal":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        return float(rng.lognormal(mu, sigma))
    elif dist_type == "normal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        return float(rng.normal(mean, std))
    elif dist_type == "constant":
        return float(params.get("value", 0.0))
    elif dist_type == "uniform":
        low = params.get("low", 0.0)
        high = params.get("high", 1.0)
        return float(rng.uniform(low, high))
    elif dist_type == "poisson":
        lam = params.get("lambda", 1.0)
        return float(rng.poisson(lam))
    else:
        raise ValueError(f"Unknown distribution type: {dist_type!r}")


def _pick_failure_cause(voice_cfg: dict, rng: np.random.Generator) -> int:
    """Pick a failure cause code from weighted list."""
    causes = voice_cfg.get("failure_causes", [])
    if not causes:
        return 19  # default: no_answer

    weights = [c["weight"] for c in causes]
    idx = _weighted_choice(weights, rng)
    cause = causes[idx]
    return cause.get("code", 19)


def _pick_normal_termination(voice_cfg: dict, rng: np.random.Generator) -> int:
    """Pick a normal termination cause code from weighted list."""
    causes = voice_cfg.get("normal_termination_causes", [])
    if not causes:
        return 16  # default: normal_clearing

    weights = [c["weight"] for c in causes]
    idx = _weighted_choice(weights, rng)
    cause = causes[idx]
    return cause.get("code", 16)


def _weighted_choice(weights: list[float], rng: np.random.Generator) -> int:
    """Select an index from a weighted list."""
    total = sum(weights)
    if total <= 0:
        return 0
    probs = [w / total for w in weights]
    return int(rng.choice(len(weights), p=probs))
