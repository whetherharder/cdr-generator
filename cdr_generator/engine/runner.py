"""Single-threaded CDR generation runner.

Orchestrates the full generation pipeline: time-step iteration, event
sampling, CDR generation, and output writing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cdr_generator.assets.contact_book import build_contact_book
from cdr_generator.assets.external_numbers import generate_external_numbers
from cdr_generator.assets.generator import generate_all_assets
from cdr_generator.assets.models import NetworkElement, Subscriber
from cdr_generator.config.models import CDRGeneratorConfig
from cdr_generator.engine.anomalies import AnomalyPipeline, AnomalyStats
from cdr_generator.engine.b_party import BPartyResult, select_b_party
from cdr_generator.engine.mobility import resolve_position
from cdr_generator.engine.poisson import sample_count
from cdr_generator.engine.rates import effective_rate
from cdr_generator.engine.rng_buffer import _RngBuffer
from cdr_generator.engine.special_events import SpecialEventEngine
from cdr_generator.generators.data import build_data_cfg_cache, generate_data_cdr
from cdr_generator.generators.extensions import generate_extensions
from cdr_generator.generators.sms import generate_sms_cdr
from cdr_generator.generators.voice import build_voice_cfg_cache, generate_voice_cdr
from cdr_generator.models.cdr import CDRRecord
from cdr_generator.writer.csv_writer import CsvWriter, create_empty_output


@dataclass(slots=True)
class GenerationStats:
    """Statistics collected during CDR generation."""

    total_records: int = 0
    voice_records: int = 0
    sms_records: int = 0
    data_records: int = 0
    files_written: int = 0
    anomaly_stats: AnomalyStats | None = None


def run_generation(
    config: CDRGeneratorConfig,
    dry_run: bool = False,
) -> GenerationStats:
    """Run single-threaded CDR generation.

    Parameters
    ----------
    config:
        Fully loaded and validated CDR generator config.
    dry_run:
        If True, only create header-only output files (no records).

    Returns
    -------
    GenerationStats
        Summary statistics of the generation run.
    """
    stats = GenerationStats()

    # Generate assets from config
    cells, network_elements, subscribers = generate_all_assets(config)

    # Build lookup structures
    cells_by_id = {c.cell_id: c for c in cells}
    nes_by_id = {ne.id: ne for ne in network_elements}

    # Build NE lookup maps by TAC and type
    tac_to_ne: dict[str, dict[int, NetworkElement]] = {
        "msc": {},
        "sgw": {},
        "pgw": {},
        "smsc": {},
    }
    for ne in network_elements:
        ne_type = ne.ne_type
        if ne_type in tac_to_ne:
            for tac in ne.serves_tacs:
                tac_to_ne[ne_type].setdefault(tac, ne)

    # PGW may not have serves_tacs; assign the first PGW to all
    pgw_ne: NetworkElement | None = None
    for ne in network_elements:
        if ne.ne_type == "pgw":
            pgw_ne = ne
            break

    output_dir = Path(config.meta.output.path)

    if dry_run:
        paths = create_empty_output(config, network_elements)
        stats.files_written = len(paths)
        return stats

    # Prepare numpy RNG for Poisson sampling and generators
    np_rng = np.random.default_rng(config.meta.seed)

    # PERFORMANCE: Pre-filled RNG buffer to amortise per-event Python overhead.
    rng_buf = _RngBuffer(np_rng)

    # Phase 4: Initialize special events engine and anomaly pipeline
    special_event_engine = SpecialEventEngine(config.special_events)
    anomaly_pipeline = AnomalyPipeline(config.anomalies, np_rng)

    # Phase 3: Build contact book and external number pool
    contact_book_cfg = config.subscribers.contact_book.model_dump()
    contact_book = build_contact_book(subscribers, contact_book_cfg, np_rng)

    ext_numbers_cfg = config.subscribers.external_numbers.model_dump()
    external_numbers = generate_external_numbers(ext_numbers_cfg, np_rng)

    # Time range — convert to UTC if tz-aware, assume UTC if naive
    _start = config.meta.time_range.start
    _end = config.meta.time_range.end
    if _start.tzinfo is not None:
        start_dt = _start.astimezone(timezone.utc)
    else:
        start_dt = _start.replace(tzinfo=timezone.utc)
    if _end.tzinfo is not None:
        end_dt = _end.astimezone(timezone.utc)
    else:
        end_dt = _end.replace(tzinfo=timezone.utc)
    step_seconds = config.meta.time_step_seconds

    # Date range for output file bucketing — clamp any spillover records
    _start_date = start_dt.date()
    _end_date = end_dt.date()

    # Voice/SMS/Data config as dicts for generators
    voice_cfg = config.events.voice.model_dump()
    sms_cfg = config.events.sms.model_dump()
    data_cfg = config.events.data.model_dump()

    # Accumulate CDRs per (ne_id, date) for sorting before write
    records_by_ne_date: dict[tuple[str, date], list[CDRRecord]] = defaultdict(list)

    # Vendor extension config lookup
    vendor_ext_cfg = config.vendor_extensions

    # ------------------------------------------------------------------
    # PERFORMANCE: Pre-compute per-profile data outside the time loop
    # ------------------------------------------------------------------

    class _ProfileCache:
        """Holds pre-computed profile data for the inner loop."""

        __slots__ = (
            "voice_lambda",
            "sms_lambda",
            "data_lambda",
            "voice_weights",
            "sms_weights",
            "data_weights",
            "voice_dow",
            "sms_dow",
            "data_dow",
            "voice_weight_sum",
            "sms_weight_sum",
            "data_weight_sum",
            "mobility",
        )

    profile_cache: dict[str, _ProfileCache] = {}
    for p in config.subscribers.profiles:
        pc = _ProfileCache()
        pc.voice_lambda = p.daily_rates.mo_call.params.get("lambda", 0)
        pc.sms_lambda = p.daily_rates.mo_sms.params.get("lambda", 0)
        pc.data_lambda = p.daily_rates.data_session.params.get("lambda", 0)
        pc.voice_weights = p.hourly_weights.voice
        pc.sms_weights = p.hourly_weights.sms
        pc.data_weights = p.hourly_weights.data
        pc.voice_dow = p.day_of_week_multipliers.voice
        pc.sms_dow = p.day_of_week_multipliers.sms
        pc.data_dow = p.day_of_week_multipliers.data
        pc.voice_weight_sum = sum(pc.voice_weights)
        pc.sms_weight_sum = sum(pc.sms_weights)
        pc.data_weight_sum = sum(pc.data_weights)
        pc.mobility = p.mobility
        profile_cache[p.name] = pc

    # PERFORMANCE: Pre-compute subscriber->profile_cache mapping and
    # validate home_cell existence once, filtering invalid subscribers.
    sub_profile_pairs: list[tuple[Subscriber, _ProfileCache]] = []
    for sub in subscribers:
        pc = profile_cache.get(sub.profile_name)
        if pc is None:
            continue
        home_cell = cells_by_id.get(sub.home_cell_id)
        if home_cell is None:
            continue
        sub_profile_pairs.append((sub, pc))

    # PERFORMANCE: Pre-build IMSI lookup for b_party (avoids rebuilding
    # the dict on every select_b_party call).
    sub_by_imsi: dict[str, Subscriber] = {s.imsi: s for s in subscribers}

    # PERFORMANCE: Pre-compute cell ID list for mobility roaming.
    all_cell_ids = list(cells_by_id.keys())

    # PERFORMANCE: Extract concurrency limits as plain ints to avoid
    # Pydantic attribute access per subscriber per timestep.
    _conc = config.events.concurrency
    conc_max_voice = _conc.max_voice
    conc_max_sms = _conc.max_sms
    conc_max_data = _conc.max_data

    # PERFORMANCE: Pre-compute per-profile data config with volume
    # multipliers applied (avoids repeated dict copy in inner loop).
    vol_mults = data_cfg.get("profile_volume_multipliers", {})
    data_cfg_by_profile: dict[str, dict] = {}
    for pname, pc in profile_cache.items():
        if pname in vol_mults:
            mult = vol_mults[pname]
            cfg_copy = dict(data_cfg)
            cfg_copy["_volume_multiplier_uplink"] = mult.get("uplink", 1.0)
            cfg_copy["_volume_multiplier_downlink"] = mult.get("downlink", 1.0)
            data_cfg_by_profile[pname] = cfg_copy
        else:
            data_cfg_by_profile[pname] = data_cfg

    # PERFORMANCE: Alias tac_to_ne sub-dicts for direct access.
    msc_by_tac = tac_to_ne["msc"]
    smsc_by_tac = tac_to_ne["smsc"]
    sgw_by_tac = tac_to_ne["sgw"]

    # Cache timedelta for step advancement
    _step_delta = timedelta(seconds=step_seconds)

    # PERFORMANCE: Step duration factor (seconds → per-hour fraction)
    _step_factor = step_seconds / 3600.0

    # PERFORMANCE: Pre-compute generator config caches (eliminates all
    # dict.get calls inside the hot path for data/voice generators).
    voice_cfg_cache = build_voice_cfg_cache(voice_cfg)
    data_cfg_cache_by_profile: dict[str, Any] = {
        pname: build_data_cfg_cache(cfg) for pname, cfg in data_cfg_by_profile.items()
    }

    # PERFORMANCE: Pre-build contact list per subscriber (indexed by
    # sub_profile_pairs index).  Avoids contact_book.get() on every
    # b_party selection call.
    sub_contacts: list[list[str]] = [
        contact_book.get(sub.imsi, []) for sub, _ in sub_profile_pairs
    ]

    # PERFORMANCE: Pre-extract b_party config constants (avoids 2
    # config.get() calls per b_party selection call).
    _b_ext_ratio: float = float(contact_book_cfg.get("external_call_ratio", 0.15))
    _b_repeat_prob: float = float(contact_book_cfg.get("repeat_call_probability", 0.6))
    _b_contact_threshold: float = _b_ext_ratio + (1.0 - _b_ext_ratio) * _b_repeat_prob
    _n_ext = len(external_numbers)

    # PERFORMANCE: Pre-cache per-subscriber home cell, TAC, and NE lookups
    # to eliminate 5-7 dict lookups per active (step, sub) event pair.
    # For the common case (no roaming), cell/TAC never changes.
    sub_home_cells = [cells_by_id[sub.home_cell_id] for sub, _ in sub_profile_pairs]
    sub_home_tacs = [sub_home_cells[i].tac for i in range(len(sub_profile_pairs))]
    sub_msc_nes = [
        msc_by_tac.get(sub_home_tacs[i]) for i in range(len(sub_profile_pairs))
    ]
    sub_smsc_nes = [
        smsc_by_tac.get(sub_home_tacs[i]) for i in range(len(sub_profile_pairs))
    ]
    sub_sgw_nes = [
        sgw_by_tac.get(sub_home_tacs[i]) for i in range(len(sub_profile_pairs))
    ]
    sub_data_cfgs = [
        data_cfg_by_profile[sub.profile_name] for sub, _ in sub_profile_pairs
    ]
    sub_data_cfg_caches_list = [
        data_cfg_cache_by_profile[sub.profile_name] for sub, _ in sub_profile_pairs
    ]

    # PERFORMANCE: Skip _apply_vendor_extensions entirely when no vendor
    # extensions are configured (avoids 1 function call per CDR record).
    _has_vendor_ext = bool(vendor_ext_cfg)

    # ------------------------------------------------------------------
    # PERFORMANCE: Vectorized rate matrix builder (closure over locals)
    # ------------------------------------------------------------------

    def _build_rate_matrix(
        hour: int,
        dow: int,
        voice_mult: float,
        sms_mult: float,
        data_mult: float,
    ) -> np.ndarray:
        """Build Poisson rate matrix of shape (n_subs, 3) for one hour.

        Returns rates [voice, sms, data] per subscriber per time step,
        incorporating hourly weights, day-of-week multipliers, special
        event multipliers, and the step duration factor.
        """
        n = len(sub_profile_pairs)
        rm = np.zeros((n, 3), dtype=np.float64)
        for i, (_, pc) in enumerate(sub_profile_pairs):
            if pc.voice_lambda > 0 and pc.voice_weight_sum > 0:
                rm[i, 0] = (
                    pc.voice_lambda
                    * pc.voice_weights[hour]
                    / pc.voice_weight_sum
                    * pc.voice_dow[dow]
                    * _step_factor
                    * voice_mult
                )
            if pc.sms_lambda > 0 and pc.sms_weight_sum > 0:
                rm[i, 1] = (
                    pc.sms_lambda
                    * pc.sms_weights[hour]
                    / pc.sms_weight_sum
                    * pc.sms_dow[dow]
                    * _step_factor
                    * sms_mult
                )
            if pc.data_lambda > 0 and pc.data_weight_sum > 0:
                rm[i, 2] = (
                    pc.data_lambda
                    * pc.data_weights[hour]
                    / pc.data_weight_sum
                    * pc.data_dow[dow]
                    * _step_factor
                    * data_mult
                )
        return rm

    # ------------------------------------------------------------------
    # Main generation loop — grouped by hour for vectorized Poisson
    # ------------------------------------------------------------------

    current = start_dt
    while current <= end_dt:
        hour = current.hour
        dow = current.weekday()

        # Phase 4: Get active special event effects for this hour block
        # (using first step — recurring events are constant within an hour)
        effects = special_event_engine.get_active_effects(current)

        # Prepare voice config with failure rate override if active
        step_voice_cfg = voice_cfg
        if effects.voice_failure_rate_override is not None:
            step_voice_cfg = dict(voice_cfg)
            step_voice_cfg["success_rate"] = 1.0 - effects.voice_failure_rate_override

        # Use pre-computed voice cache only when no special-event override
        # (override changes success_rate which is pre-baked into the cache)
        _step_vcfg = voice_cfg_cache if step_voice_cfg is voice_cfg else None

        # Cache special event multipliers for this hour
        voice_mult = effects.voice_rate_multiplier
        sms_mult = effects.sms_rate_multiplier
        data_mult = effects.data_rate_multiplier
        has_disabled_cells = len(effects.disabled_cells) > 0

        # Collect all time steps within this clock hour
        hour_steps: list = []
        step_cur = current
        while step_cur <= end_dt and step_cur.hour == hour:
            hour_steps.append(step_cur)
            step_cur += _step_delta

        n_steps = len(hour_steps)
        n_subs = len(sub_profile_pairs)

        if has_disabled_cells or n_subs == 0:
            # -------------------------------------------------------
            # Slow path: per-step, per-subscriber loop.
            # Used when disabled cells are active (rare) to preserve
            # exact per-step effect recalculation and failed-CDR logic.
            # -------------------------------------------------------
            for step_time in hour_steps:
                # Recalculate effects per step (time-ranged events may
                # change at arbitrary minute boundaries within the hour)
                step_effects = special_event_engine.get_active_effects(step_time)
                sv_mult = step_effects.voice_rate_multiplier
                ss_mult = step_effects.sms_rate_multiplier
                sd_mult = step_effects.data_rate_multiplier
                s_has_disabled = len(step_effects.disabled_cells) > 0
                sv_cfg = voice_cfg
                if step_effects.voice_failure_rate_override is not None:
                    sv_cfg = dict(voice_cfg)
                    sv_cfg["success_rate"] = (
                        1.0 - step_effects.voice_failure_rate_override
                    )

                s_hour = step_time.hour
                s_dow = step_time.weekday()

                for sub, pc in sub_profile_pairs:
                    home_cell = cells_by_id.get(sub.home_cell_id)
                    if home_cell is None:
                        continue

                    first_cell_id, last_cell_id = resolve_position(
                        home_cell_id=sub.home_cell_id,
                        work_cell_id=sub.work_cell_id,
                        mobility=pc.mobility,
                        timestamp=step_time,
                        cells_by_id=cells_by_id,
                        rng=np_rng,
                        all_cell_ids=all_cell_ids,
                    )
                    cell = cells_by_id.get(first_cell_id, home_cell)

                    if s_has_disabled and step_effects.is_cell_disabled(first_cell_id):
                        overflow_cell_id = step_effects.pick_overflow_cell(np_rng)
                        if overflow_cell_id is not None:
                            overflow_cell = cells_by_id.get(overflow_cell_id)
                            if overflow_cell is not None:
                                first_cell_id = overflow_cell_id
                                last_cell_id = overflow_cell_id
                                cell = overflow_cell
                            else:
                                continue
                        else:
                            msc_ne = msc_by_tac.get(home_cell.tac)
                            if msc_ne is not None:
                                failed_cdr = CDRRecord(
                                    record_type="mo_call",
                                    served_imsi=sub.imsi,
                                    served_msisdn=sub.msisdn,
                                    served_imei=sub.imei,
                                    event_timestamp=step_time,
                                    calling_number=sub.msisdn,
                                    called_number="",
                                    duration_seconds=0.0,
                                    cause_for_termination=38,
                                    first_cell_id=first_cell_id,
                                    last_cell_id=first_cell_id,
                                    serving_ne_id=msc_ne.id,
                                    rat_type="eutran",
                                )
                                ne_id = failed_cdr.serving_ne_id
                                cdr_date = min(
                                    failed_cdr.event_timestamp.date(), _end_date
                                )
                                records_by_ne_date[(ne_id, cdr_date)].append(failed_cdr)
                                stats.voice_records += 1
                                stats.total_records += 1
                            continue

                    home_tac = cell.tac

                    voice_rate = (
                        effective_rate(
                            base_lambda=pc.voice_lambda,
                            hourly_weights=pc.voice_weights,
                            dow_multipliers=pc.voice_dow,
                            hour=s_hour,
                            dow=s_dow,
                            time_step_seconds=step_seconds,
                            weight_sum=pc.voice_weight_sum,
                        )
                        * sv_mult
                    )
                    sms_rate = (
                        effective_rate(
                            base_lambda=pc.sms_lambda,
                            hourly_weights=pc.sms_weights,
                            dow_multipliers=pc.sms_dow,
                            hour=s_hour,
                            dow=s_dow,
                            time_step_seconds=step_seconds,
                            weight_sum=pc.sms_weight_sum,
                        )
                        * ss_mult
                    )
                    data_rate = (
                        effective_rate(
                            base_lambda=pc.data_lambda,
                            hourly_weights=pc.data_weights,
                            dow_multipliers=pc.data_dow,
                            hour=s_hour,
                            dow=s_dow,
                            time_step_seconds=step_seconds,
                            weight_sum=pc.data_weight_sum,
                        )
                        * sd_mult
                    )

                    n_voice = sample_count(voice_rate, np_rng)
                    n_sms = sample_count(sms_rate, np_rng)
                    n_data = sample_count(data_rate, np_rng)

                    if conc_max_voice is not None and n_voice > conc_max_voice:
                        n_voice = conc_max_voice
                    if conc_max_sms is not None and n_sms > conc_max_sms:
                        n_sms = conc_max_sms
                    if conc_max_data is not None and n_data > conc_max_data:
                        n_data = conc_max_data

                    # --- Voice MO ---
                    msc_ne = msc_by_tac.get(home_tac)
                    if n_voice > 0 and msc_ne is not None:
                        for _ in range(n_voice):
                            b_result = select_b_party(
                                sub,
                                contact_book,
                                subscribers,
                                external_numbers,
                                contact_book_cfg,
                                np_rng,
                                sub_by_imsi,
                            )
                            callee = _b_party_to_subscriber(b_result)

                            fwd_b_result = select_b_party(
                                sub,
                                contact_book,
                                subscribers,
                                external_numbers,
                                contact_book_cfg,
                                np_rng,
                                sub_by_imsi,
                            )
                            fwd_sub = _b_party_to_subscriber(fwd_b_result)
                            fwd_target = (
                                fwd_sub if fwd_sub.msisdn != callee.msisdn else None
                            )

                            cdrs = generate_voice_cdr(
                                caller=sub,
                                callee=callee,
                                event_time=step_time,
                                cell=cell,
                                msc=msc_ne,
                                voice_cfg=sv_cfg,
                                rng=np_rng,
                                forward_target=fwd_target,
                                _buf=rng_buf,
                            )
                            for cdr in cdrs:
                                if cdr.served_imsi == sub.imsi:
                                    cdr.first_cell_id = first_cell_id
                                    cdr.last_cell_id = last_cell_id
                                _apply_vendor_extensions(
                                    cdr, msc_ne, vendor_ext_cfg, np_rng
                                )
                                ne_id = cdr.serving_ne_id
                                cdr_date = min(cdr.event_timestamp.date(), _end_date)
                                records_by_ne_date[(ne_id, cdr_date)].append(cdr)
                                stats.voice_records += 1
                                stats.total_records += 1

                    # --- SMS MO ---
                    smsc_ne = smsc_by_tac.get(home_tac)
                    if n_sms > 0 and smsc_ne is not None:
                        for _ in range(n_sms):
                            b_result = select_b_party(
                                sub,
                                contact_book,
                                subscribers,
                                external_numbers,
                                contact_book_cfg,
                                np_rng,
                                sub_by_imsi,
                            )
                            recipient = _b_party_to_subscriber(b_result)

                            cdrs = generate_sms_cdr(
                                sender=sub,
                                recipient=recipient,
                                event_time=step_time,
                                cell=cell,
                                smsc=smsc_ne,
                                sms_cfg=sms_cfg,
                                rng=np_rng,
                                _buf=rng_buf,
                            )
                            for cdr in cdrs:
                                if cdr.served_imsi == sub.imsi:
                                    cdr.first_cell_id = first_cell_id
                                    cdr.last_cell_id = last_cell_id
                                _apply_vendor_extensions(
                                    cdr, smsc_ne, vendor_ext_cfg, np_rng
                                )
                                ne_id = cdr.serving_ne_id
                                cdr_date = min(cdr.event_timestamp.date(), _end_date)
                                records_by_ne_date[(ne_id, cdr_date)].append(cdr)
                                stats.sms_records += 1
                                stats.total_records += 1

                    # --- Data session ---
                    sgw_ne = sgw_by_tac.get(home_tac)
                    if n_data > 0 and sgw_ne is not None and pgw_ne is not None:
                        step_data_cfg = data_cfg_by_profile[sub.profile_name]
                        for _ in range(n_data):
                            cdrs = generate_data_cdr(
                                subscriber=sub,
                                event_time=step_time,
                                cell=cell,
                                sgw=sgw_ne,
                                pgw=pgw_ne,
                                data_cfg=step_data_cfg,
                                rng=np_rng,
                                _buf=rng_buf,
                            )
                            for cdr in cdrs:
                                cdr.first_cell_id = first_cell_id
                                cdr.last_cell_id = last_cell_id
                                serving_ne = nes_by_id.get(cdr.serving_ne_id)
                                if serving_ne is not None:
                                    _apply_vendor_extensions(
                                        cdr, serving_ne, vendor_ext_cfg, np_rng
                                    )
                                ne_id = cdr.serving_ne_id
                                cdr_date = min(cdr.event_timestamp.date(), _end_date)
                                records_by_ne_date[(ne_id, cdr_date)].append(cdr)
                                stats.data_records += 1
                                stats.total_records += 1

        else:
            # -------------------------------------------------------
            # Fast path: vectorized Poisson sampling.
            # Builds a rate matrix for all subscribers, samples all
            # counts for every step in this hour in one numpy call,
            # then iterates only over non-zero (step, sub) pairs.
            # This eliminates ~98% of empty iterations.
            # -------------------------------------------------------
            rate_matrix = _build_rate_matrix(hour, dow, voice_mult, sms_mult, data_mult)

            # Sample counts for all steps × subscribers × event types
            # rate_matrix shape: (n_subs, 3)
            # counts shape:      (n_steps, n_subs, 3)
            counts = np_rng.poisson(rate_matrix, size=(n_steps, n_subs, 3))

            # Apply concurrency limits in-place
            if conc_max_voice is not None:
                np.clip(counts[:, :, 0], 0, conc_max_voice, out=counts[:, :, 0])
            if conc_max_sms is not None:
                np.clip(counts[:, :, 1], 0, conc_max_sms, out=counts[:, :, 1])
            if conc_max_data is not None:
                np.clip(counts[:, :, 2], 0, conc_max_data, out=counts[:, :, 2])

            # Find (step_idx, sub_idx) pairs with at least one event
            # counts.any(axis=2) is True wherever any event type > 0
            active_pairs = np.argwhere(counts.any(axis=2))

            # PERFORMANCE: Pre-extract counts for all active pairs with one
            # vectorized fancy-index call + single .tolist() conversion.
            # Avoids N×3 numpy scalar lookups + N×3 int() casts per loop.
            if len(active_pairs):
                _active_counts = counts[
                    active_pairs[:, 0], active_pairs[:, 1], :
                ].tolist()
            else:
                _active_counts = []
            _active_pairs_list = active_pairs.tolist()

            for _ap_i, step_sub in enumerate(_active_pairs_list):
                step_idx, sub_idx = step_sub
                step_time = hour_steps[step_idx]
                sub, pc = sub_profile_pairs[sub_idx]

                # PERFORMANCE: Use pre-cached home cell (avoids dict lookup)
                home_cell = sub_home_cells[sub_idx]

                # Resolve position only for subscribers with actual events
                first_cell_id, last_cell_id = resolve_position(
                    home_cell_id=sub.home_cell_id,
                    work_cell_id=sub.work_cell_id,
                    mobility=pc.mobility,
                    timestamp=step_time,
                    cells_by_id=cells_by_id,
                    rng=np_rng,
                    all_cell_ids=all_cell_ids,
                    _buf=rng_buf,
                )
                # PERFORMANCE: Fast path when no mobility occurred (common case)
                if first_cell_id == sub.home_cell_id:
                    cell = home_cell
                    msc_ne = sub_msc_nes[sub_idx]
                    smsc_ne = sub_smsc_nes[sub_idx]
                    sgw_ne = sub_sgw_nes[sub_idx]
                else:
                    cell = cells_by_id.get(first_cell_id, home_cell)
                    _tac = cell.tac
                    if _tac == sub_home_tacs[sub_idx]:
                        msc_ne = sub_msc_nes[sub_idx]
                        smsc_ne = sub_smsc_nes[sub_idx]
                        sgw_ne = sub_sgw_nes[sub_idx]
                    else:
                        msc_ne = msc_by_tac.get(_tac)
                        smsc_ne = smsc_by_tac.get(_tac)
                        sgw_ne = sgw_by_tac.get(_tac)

                n_voice, n_sms, n_data = _active_counts[_ap_i]

                # PERFORMANCE: Pre-compute CDR date once per active pair.
                # step_time is always within [start_dt, end_dt] by construction
                # so step_time.date() <= _end_date always holds.  MT CDR jitter
                # is sub-minute and won't cross a day boundary for the MO step.
                step_cdr_date = step_time.date()

                _sub_contacts = sub_contacts[sub_idx]

                # --- Voice MO ---
                if n_voice > 0 and msc_ne is not None:
                    # Use pre-built contact list and pre-extracted ratio constants
                    for _ in range(n_voice):
                        b_result = select_b_party(
                            sub,
                            contact_book,
                            subscribers,
                            external_numbers,
                            contact_book_cfg,
                            np_rng,
                            sub_by_imsi,
                            a_party_contacts=_sub_contacts,
                            ext_ratio=_b_ext_ratio,
                            contact_threshold=_b_contact_threshold,
                        )
                        callee = _b_party_to_subscriber(b_result)

                        fwd_b_result = select_b_party(
                            sub,
                            contact_book,
                            subscribers,
                            external_numbers,
                            contact_book_cfg,
                            np_rng,
                            sub_by_imsi,
                            a_party_contacts=_sub_contacts,
                            ext_ratio=_b_ext_ratio,
                            contact_threshold=_b_contact_threshold,
                        )
                        fwd_sub = _b_party_to_subscriber(fwd_b_result)
                        # Compare MSISDNs, not IMSIs: all external numbers
                        # share imsi="external", so IMSI comparison would
                        # wrongly suppress forwarding when both are external.
                        fwd_target = (
                            fwd_sub if fwd_sub.msisdn != callee.msisdn else None
                        )

                        cdrs = generate_voice_cdr(
                            caller=sub,
                            callee=callee,
                            event_time=step_time,
                            cell=cell,
                            msc=msc_ne,
                            voice_cfg=step_voice_cfg,
                            rng=np_rng,
                            forward_target=fwd_target,
                            _buf=rng_buf,
                            _vcfg=_step_vcfg,
                        )
                        for cdr in cdrs:
                            if cdr.served_imsi == sub.imsi:
                                cdr.first_cell_id = first_cell_id
                                cdr.last_cell_id = last_cell_id
                            if _has_vendor_ext:
                                _apply_vendor_extensions(
                                    cdr, msc_ne, vendor_ext_cfg, np_rng
                                )
                            records_by_ne_date[
                                (cdr.serving_ne_id, step_cdr_date)
                            ].append(cdr)
                            stats.voice_records += 1
                            stats.total_records += 1

                # --- SMS MO ---
                if n_sms > 0 and smsc_ne is not None:
                    for _ in range(n_sms):
                        b_result = select_b_party(
                            sub,
                            contact_book,
                            subscribers,
                            external_numbers,
                            contact_book_cfg,
                            np_rng,
                            sub_by_imsi,
                            a_party_contacts=_sub_contacts,
                            ext_ratio=_b_ext_ratio,
                            contact_threshold=_b_contact_threshold,
                        )
                        recipient = _b_party_to_subscriber(b_result)

                        cdrs = generate_sms_cdr(
                            sender=sub,
                            recipient=recipient,
                            event_time=step_time,
                            cell=cell,
                            smsc=smsc_ne,
                            sms_cfg=sms_cfg,
                            rng=np_rng,
                            _buf=rng_buf,
                        )
                        for cdr in cdrs:
                            if cdr.served_imsi == sub.imsi:
                                cdr.first_cell_id = first_cell_id
                                cdr.last_cell_id = last_cell_id
                            if _has_vendor_ext:
                                _apply_vendor_extensions(
                                    cdr, smsc_ne, vendor_ext_cfg, np_rng
                                )
                            # SMS MT delivery delay can span up to 86400s;
                            # use actual event_timestamp date (not step date)
                            # to match slow-path bucketing.
                            _sms_date = min(cdr.event_timestamp.date(), _end_date)
                            records_by_ne_date[(cdr.serving_ne_id, _sms_date)].append(
                                cdr
                            )
                            stats.sms_records += 1
                            stats.total_records += 1

                # --- Data session ---
                if n_data > 0 and sgw_ne is not None and pgw_ne is not None:
                    # PERFORMANCE: use pre-cached data config (avoids 2 dict lookups)
                    step_data_cfg = sub_data_cfgs[sub_idx]
                    step_data_cfg_cache = sub_data_cfg_caches_list[sub_idx]
                    for _ in range(n_data):
                        cdrs = generate_data_cdr(
                            subscriber=sub,
                            event_time=step_time,
                            cell=cell,
                            sgw=sgw_ne,
                            pgw=pgw_ne,
                            data_cfg=step_data_cfg,
                            rng=np_rng,
                            _buf=rng_buf,
                            _cfg=step_data_cfg_cache,
                        )
                        for cdr in cdrs:
                            cdr.first_cell_id = first_cell_id
                            cdr.last_cell_id = last_cell_id
                            if _has_vendor_ext:
                                serving_ne = nes_by_id.get(cdr.serving_ne_id)
                                if serving_ne is not None:
                                    _apply_vendor_extensions(
                                        cdr, serving_ne, vendor_ext_cfg, np_rng
                                    )
                            records_by_ne_date[
                                (cdr.serving_ne_id, step_cdr_date)
                            ].append(cdr)
                            stats.data_records += 1
                            stats.total_records += 1

        current = step_cur

    # Phase 4: Apply anomaly pipeline to all records before writing
    combined_anomaly_stats = AnomalyStats()

    # Write output files: sort records and write per (NE, date)
    output_cfg = config.meta.output
    writer = CsvWriter(
        output_dir=output_dir,
        delimiter=output_cfg.csv_delimiter,
        include_metadata=output_cfg.include_metadata_comment,
        filename_template=output_cfg.filename_template,
    )

    for (ne_id, file_date), records in sorted(records_by_ne_date.items()):
        # Apply anomaly pipeline per (ne_id, date) batch
        records, anomaly_batch_stats = anomaly_pipeline.apply(records)

        # Accumulate anomaly stats
        combined_anomaly_stats.orphaned_count += anomaly_batch_stats.orphaned_count
        combined_anomaly_stats.missing_fields_count += (
            anomaly_batch_stats.missing_fields_count
        )
        combined_anomaly_stats.corrupt_values_count += (
            anomaly_batch_stats.corrupt_values_count
        )
        combined_anomaly_stats.timestamp_anomalies_count += (
            anomaly_batch_stats.timestamp_anomalies_count
        )
        combined_anomaly_stats.duplicate_count += anomaly_batch_stats.duplicate_count

        # Update total_records to reflect anomaly changes
        stats.total_records += (
            anomaly_batch_stats.duplicate_count - anomaly_batch_stats.orphaned_count
        )

        # Sort by event_timestamp
        records.sort(key=lambda r: r.event_timestamp)
        writer.write_file(ne_id=ne_id, file_date=file_date, records=records)
        stats.files_written += 1

    stats.anomaly_stats = combined_anomaly_stats

    # Also create empty files for NEs/dates with no records
    current_date = _start_date
    while current_date <= _end_date:
        for ne in network_elements:
            key = (ne.id, current_date)
            if key not in records_by_ne_date:
                writer.write_file(ne_id=ne.id, file_date=current_date, records=[])
                stats.files_written += 1
        current_date += timedelta(days=1)

    return stats


def _b_party_to_subscriber(b_result: BPartyResult) -> Subscriber:
    """Convert a BPartyResult to a Subscriber for use in generators."""
    if b_result.subscriber is not None:
        return b_result.subscriber

    # External number: create a synthetic Subscriber
    return Subscriber(
        imsi=b_result.imsi,
        msisdn=b_result.msisdn,
        imei="00000000000000",
        profile_name="external",
        home_cell_id=0,
        serving_ne_id="external",
    )


def _apply_vendor_extensions(
    cdr: CDRRecord,
    ne: NetworkElement,
    vendor_ext_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Apply vendor-specific extensions to a CDR record."""
    if not vendor_ext_cfg:
        return

    ext_cfg = vendor_ext_cfg.get(ne.vendor)
    if ext_cfg is None:
        return

    event_ctx = {
        "first_cell_id": cdr.first_cell_id,
        "last_cell_id": cdr.last_cell_id,
        "apn": cdr.apn,
    }
    ext = generate_extensions(
        vendor=ne.vendor,
        vendor_config=ext_cfg,
        event_context=event_ctx,
        rng=rng,
    )
    if ext is not None:
        cdr.vendor_extensions = ext
