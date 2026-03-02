"""Voice CDR generator -- MO/MT call pairs with success/failure handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber
from cdr_generator.models.cdr import CDRRecord

if TYPE_CHECKING:
    from cdr_generator.engine.rng_buffer import _RngBuffer


@dataclass(slots=True)
class _VoiceCfgCache:
    """Pre-computed static config for zero-dict-lookup hot path.

    Build once with ``build_voice_cfg_cache(voice_cfg)`` and pass as
    ``_vcfg`` to ``generate_voice_cdr`` to bypass all dict.get calls
    in the inner generation loop.
    """

    success_rate: float
    forwarding_rate: float
    dur_mu: float
    dur_sigma: float
    min_dur: float
    max_dur: float
    jitter_hi: int
    fail_cause_list: list
    fail_cause_probs: tuple
    term_cause_list: list
    term_cause_probs: tuple


def build_voice_cfg_cache(voice_cfg: dict) -> _VoiceCfgCache:
    """Pre-extract all static config values from voice_cfg into a fast-access object."""
    dur_cfg = voice_cfg.get("duration", {})
    dur_dist = dur_cfg.get("distribution", {})
    dur_params = dur_dist.get("params", {})

    jitter_ms = voice_cfg.get("paired_timestamp_jitter_ms", 1000)

    fail_causes = voice_cfg.get("failure_causes", [])
    if fail_causes:
        fail_codes = [c.get("code", 19) for c in fail_causes]
        raw_fail_w = [c["weight"] for c in fail_causes]
        total_fail = sum(raw_fail_w) or 1.0
        fail_probs = tuple(w / total_fail for w in raw_fail_w)
    else:
        fail_codes = [19]
        fail_probs = (1.0,)

    term_causes = voice_cfg.get("normal_termination_causes", [])
    if term_causes:
        term_codes = [c.get("code", 16) for c in term_causes]
        raw_term_w = [c["weight"] for c in term_causes]
        total_term = sum(raw_term_w) or 1.0
        term_probs = tuple(w / total_term for w in raw_term_w)
    else:
        term_codes = [16]
        term_probs = (1.0,)

    return _VoiceCfgCache(
        success_rate=float(voice_cfg.get("success_rate", 0.85)),
        forwarding_rate=float(voice_cfg.get("call_forwarding_rate", 0.03)),
        dur_mu=float(dur_params.get("mu", 0.0)),
        dur_sigma=float(dur_params.get("sigma", 1.0)),
        min_dur=float(dur_cfg.get("min_seconds", 1)),
        max_dur=float(dur_cfg.get("max_seconds", 7200)),
        jitter_hi=max(1, jitter_ms),
        fail_cause_list=fail_codes,
        fail_cause_probs=fail_probs,
        term_cause_list=term_codes,
        term_cause_probs=term_probs,
    )


def generate_voice_cdr(
    caller: Subscriber,
    callee: Subscriber,
    event_time: datetime,
    cell: Cell,
    msc: NetworkElement,
    voice_cfg: dict,
    rng: np.random.Generator,
    forward_target: Subscriber | None = None,
    _buf: _RngBuffer | None = None,
    _vcfg: _VoiceCfgCache | None = None,
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
    _buf:
        Optional pre-filled RNG buffer for high-throughput generation.
    _vcfg:
        Optional pre-computed config cache (build with
        ``build_voice_cfg_cache``).  When provided, all dict.get calls
        and weight normalizations are bypassed.

    Returns
    -------
    list[CDRRecord]
        One record (MO only) for failed calls, two records (MO + MT) for
        successful non-forwarded calls, three records (MO + MT-fwd + MT-final)
        for forwarded calls. All records share a consolidation_id.
    """
    if _vcfg is not None:
        success_rate = _vcfg.success_rate
    else:
        success_rate = voice_cfg.get("success_rate", 0.85)
    roll = _buf.get_float() if _buf is not None else float(rng.random())
    is_success = roll < success_rate

    if _buf is not None:
        consolidation_id = _buf.get_uuid()
    else:
        consolidation_id = bytes(
            rng.integers(0, 256, size=16, dtype="uint8")
        ).hex()

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
            _buf=_buf,
            _vcfg=_vcfg,
        )

    # Check for call forwarding (only on successful calls with a target)
    if _vcfg is not None:
        forwarding_rate = _vcfg.forwarding_rate
    else:
        forwarding_rate = voice_cfg.get("call_forwarding_rate", 0.03)
    fwd_roll = _buf.get_float() if _buf is not None else float(rng.random())
    is_forwarded = forward_target is not None and fwd_roll < forwarding_rate

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
            _buf=_buf,
            _vcfg=_vcfg,
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
        _buf=_buf,
        _vcfg=_vcfg,
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
    _buf: _RngBuffer | None = None,
    _vcfg: _VoiceCfgCache | None = None,
) -> list[CDRRecord]:
    """Generate a single MO CDR for a failed call attempt."""
    if _vcfg is not None and _buf is not None:
        cause_code = _vcfg.fail_cause_list[
            _buf.get_choice(len(_vcfg.fail_cause_list), _vcfg.fail_cause_probs)
        ]
    else:
        cause_code = _pick_failure_cause(voice_cfg, rng, _buf=_buf)

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
    _buf: _RngBuffer | None = None,
    _vcfg: _VoiceCfgCache | None = None,
) -> list[CDRRecord]:
    """Generate MO + MT CDR pair for a successful call."""
    if _vcfg is not None and _buf is not None:
        raw = _buf.get_lognormal(_vcfg.dur_mu, _vcfg.dur_sigma)
        duration = float(max(_vcfg.min_dur, min(raw, _vcfg.max_dur)))
        cause_code = _vcfg.term_cause_list[
            _buf.get_choice(len(_vcfg.term_cause_list), _vcfg.term_cause_probs)
        ]
        jitter_val = _buf.get_int(0, _vcfg.jitter_hi)
    else:
        duration = _sample_duration(voice_cfg, rng, _buf=_buf)
        cause_code = _pick_normal_termination(voice_cfg, rng, _buf=_buf)
        jitter_ms = voice_cfg.get("paired_timestamp_jitter_ms", 1000)
        hi = max(1, jitter_ms)
        if _buf is not None:
            jitter_val = _buf.get_int(0, hi)
        else:
            jitter_val = int(rng.integers(0, hi))
    mt_jitter = timedelta(milliseconds=jitter_val)
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
    _buf: _RngBuffer | None = None,
    _vcfg: _VoiceCfgCache | None = None,
) -> list[CDRRecord]:
    """Generate MO + MT-forwarding + MT-final CDR triple for a forwarded call.

    A forwarded call produces three CDRs:
    1. MO (A->B): caller initiates, same as normal MO.
    2. MT forwarding (B): served by callee, with redirecting_number=C.
    3. MT final (C): served by forward_target, the actual answerer.

    All three share the same consolidation_id.
    """
    if _vcfg is not None and _buf is not None:
        raw = _buf.get_lognormal(_vcfg.dur_mu, _vcfg.dur_sigma)
        duration = float(max(_vcfg.min_dur, min(raw, _vcfg.max_dur)))
        cause_code = _vcfg.term_cause_list[
            _buf.get_choice(len(_vcfg.term_cause_list), _vcfg.term_cause_probs)
        ]
        _jitter_hi = _vcfg.jitter_hi
        jitter_val = _buf.get_int(0, _jitter_hi)
    else:
        duration = _sample_duration(voice_cfg, rng, _buf=_buf)
        cause_code = _pick_normal_termination(voice_cfg, rng, _buf=_buf)
        jitter_ms = voice_cfg.get("paired_timestamp_jitter_ms", 1000)
        _jitter_hi = max(1, jitter_ms)
        if _buf is not None:
            jitter_val = _buf.get_int(0, _jitter_hi)
        else:
            jitter_val = int(rng.integers(0, _jitter_hi))
    mt_jitter = timedelta(milliseconds=jitter_val)
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
    if _buf is not None:
        final_jitter_val = _buf.get_int(0, _jitter_hi)
    else:
        final_jitter_val = int(rng.integers(0, _jitter_hi))
    mt_final_jitter = timedelta(milliseconds=final_jitter_val)
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


def _sample_duration(
    voice_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> float:
    """Sample call duration from the configured distribution."""
    dur_cfg = voice_cfg["duration"]
    dist = dur_cfg["distribution"]

    raw = _sample_distribution(dist, rng, _buf=_buf)

    min_s = dur_cfg.get("min_seconds", 1)
    max_s = dur_cfg.get("max_seconds", 7200)
    return float(max(min_s, min(raw, max_s)))


def _sample_distribution(
    dist: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> float:
    """Sample a single value from a distribution descriptor."""
    dist_type = dist["type"]
    params = dist.get("params", {})

    if dist_type == "lognormal":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        if _buf is not None:
            return _buf.get_lognormal(mu, sigma)
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
        if _buf is not None:
            # Use get_float scaled to range; or just use rng for less common type
            return float(rng.uniform(low, high))
        return float(rng.uniform(low, high))
    elif dist_type == "poisson":
        lam = params.get("lambda", 1.0)
        return float(rng.poisson(lam))
    else:
        raise ValueError(f"Unknown distribution type: {dist_type!r}")


def _pick_failure_cause(
    voice_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int:
    """Pick a failure cause code from weighted list."""
    causes = voice_cfg.get("failure_causes", [])
    if not causes:
        return 19  # default: no_answer

    weights = [c["weight"] for c in causes]
    idx = _weighted_choice(weights, rng, _buf=_buf)
    cause = causes[idx]
    return cause.get("code", 19)


def _pick_normal_termination(
    voice_cfg: dict,
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int:
    """Pick a normal termination cause code from weighted list."""
    causes = voice_cfg.get("normal_termination_causes", [])
    if not causes:
        return 16  # default: normal_clearing

    weights = [c["weight"] for c in causes]
    idx = _weighted_choice(weights, rng, _buf=_buf)
    cause = causes[idx]
    return cause.get("code", 16)


def _weighted_choice(
    weights: list[float],
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int:
    """Select an index from a weighted list."""
    total = sum(weights)
    if total <= 0:
        return 0
    if _buf is not None:
        probs = tuple(w / total for w in weights)
        return _buf.get_choice(len(weights), probs)
    probs_list = [w / total for w in weights]
    return int(rng.choice(len(weights), p=probs_list))
