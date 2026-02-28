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
from cdr_generator.engine.special_events import SpecialEventEngine
from cdr_generator.generators.extensions import generate_extensions
from cdr_generator.models.cdr import CDRRecord
from cdr_generator.writer.csv_writer import CsvWriter, create_empty_output


@dataclass
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

    # Import generators
    from cdr_generator.generators.data import generate_data_cdr
    from cdr_generator.generators.sms import generate_sms_cdr
    from cdr_generator.generators.voice import generate_voice_cdr

    # Vendor extension config lookup
    vendor_ext_cfg = config.vendor_extensions

    # ------------------------------------------------------------------
    # PERFORMANCE: Pre-compute per-profile data outside the time loop
    # ------------------------------------------------------------------
    # For each profile, cache the rate parameters so we avoid repeated
    # Pydantic attribute traversals and dict lookups in the inner loop.

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

    current = start_dt
    while current <= end_dt:
        hour = current.hour
        dow = current.weekday()

        # Phase 4: Get active special event effects for this time step
        effects = special_event_engine.get_active_effects(current)

        # Prepare voice config with failure rate override if active
        step_voice_cfg = voice_cfg
        if effects.voice_failure_rate_override is not None:
            step_voice_cfg = dict(voice_cfg)
            step_voice_cfg["success_rate"] = 1.0 - effects.voice_failure_rate_override

        # Cache special event multipliers for this time step
        voice_mult = effects.voice_rate_multiplier
        sms_mult = effects.sms_rate_multiplier
        data_mult = effects.data_rate_multiplier
        has_disabled_cells = len(effects.disabled_cells) > 0

        for sub, pc in sub_profile_pairs:
            home_cell = cells_by_id.get(sub.home_cell_id)
            if home_cell is None:
                continue

            # Phase 3: Resolve subscriber position using mobility model
            first_cell_id, last_cell_id = resolve_position(
                home_cell_id=sub.home_cell_id,
                work_cell_id=sub.work_cell_id,
                mobility=pc.mobility,
                timestamp=current,
                cells_by_id=cells_by_id,
                rng=np_rng,
                all_cell_ids=all_cell_ids,
            )
            cell = cells_by_id.get(first_cell_id, home_cell)

            # Phase 4/5: Handle disabled cells — relocate to overflow or
            # generate a failed CDR with cause_code=38
            if has_disabled_cells and effects.is_cell_disabled(first_cell_id):
                overflow_cell_id = effects.pick_overflow_cell(np_rng)
                if overflow_cell_id is not None:
                    overflow_cell = cells_by_id.get(overflow_cell_id)
                    if overflow_cell is not None:
                        first_cell_id = overflow_cell_id
                        last_cell_id = overflow_cell_id
                        cell = overflow_cell
                    else:
                        continue  # overflow cell not found, skip subscriber
                else:
                    # Phase 5: No overflow — generate failed CDR with cause_code=38
                    msc_ne = msc_by_tac.get(home_cell.tac)
                    if msc_ne is not None:
                        failed_cdr = CDRRecord(
                            record_type="mo_call",
                            served_imsi=sub.imsi,
                            served_msisdn=sub.msisdn,
                            served_imei=sub.imei,
                            event_timestamp=current,
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
                        cdr_date = min(failed_cdr.event_timestamp.date(), _end_date)
                        records_by_ne_date[(ne_id, cdr_date)].append(failed_cdr)
                        stats.voice_records += 1
                        stats.total_records += 1
                    continue  # skip normal event generation for this subscriber

            home_tac = cell.tac

            # --- Compute event counts (using cached profile data) ---
            voice_rate = effective_rate(
                base_lambda=pc.voice_lambda,
                hourly_weights=pc.voice_weights,
                dow_multipliers=pc.voice_dow,
                hour=hour,
                dow=dow,
                time_step_seconds=step_seconds,
                weight_sum=pc.voice_weight_sum,
            )
            voice_rate *= voice_mult
            n_voice = sample_count(voice_rate, np_rng)

            sms_rate = effective_rate(
                base_lambda=pc.sms_lambda,
                hourly_weights=pc.sms_weights,
                dow_multipliers=pc.sms_dow,
                hour=hour,
                dow=dow,
                time_step_seconds=step_seconds,
                weight_sum=pc.sms_weight_sum,
            )
            sms_rate *= sms_mult
            n_sms = sample_count(sms_rate, np_rng)

            data_rate = effective_rate(
                base_lambda=pc.data_lambda,
                hourly_weights=pc.data_weights,
                dow_multipliers=pc.data_dow,
                hour=hour,
                dow=dow,
                time_step_seconds=step_seconds,
                weight_sum=pc.data_weight_sum,
            )
            data_rate *= data_mult
            n_data = sample_count(data_rate, np_rng)

            # Phase 5: Apply concurrency limits (inlined to avoid
            # Pydantic attribute access per subscriber per timestep)
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

                    # Phase 5: Select forwarding target from contact book
                    fwd_target = None
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
                    if fwd_sub.imsi != callee.imsi:
                        fwd_target = fwd_sub

                    cdrs = generate_voice_cdr(
                        caller=sub,
                        callee=callee,
                        event_time=current,
                        cell=cell,
                        msc=msc_ne,
                        voice_cfg=step_voice_cfg,
                        rng=np_rng,
                        forward_target=fwd_target,
                    )
                    for cdr in cdrs:
                        # Apply mobility cell IDs
                        if cdr.served_imsi == sub.imsi:
                            cdr.first_cell_id = first_cell_id
                            cdr.last_cell_id = last_cell_id
                        # Apply vendor extensions
                        _apply_vendor_extensions(
                            cdr,
                            msc_ne,
                            vendor_ext_cfg,
                            np_rng,
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
                        event_time=current,
                        cell=cell,
                        smsc=smsc_ne,
                        sms_cfg=sms_cfg,
                        rng=np_rng,
                    )
                    for cdr in cdrs:
                        if cdr.served_imsi == sub.imsi:
                            cdr.first_cell_id = first_cell_id
                            cdr.last_cell_id = last_cell_id
                        _apply_vendor_extensions(
                            cdr,
                            smsc_ne,
                            vendor_ext_cfg,
                            np_rng,
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
                        event_time=current,
                        cell=cell,
                        sgw=sgw_ne,
                        pgw=pgw_ne,
                        data_cfg=step_data_cfg,
                        rng=np_rng,
                    )
                    for cdr in cdrs:
                        cdr.first_cell_id = first_cell_id
                        cdr.last_cell_id = last_cell_id
                        serving_ne = nes_by_id.get(cdr.serving_ne_id)
                        if serving_ne is not None:
                            _apply_vendor_extensions(
                                cdr,
                                serving_ne,
                                vendor_ext_cfg,
                                np_rng,
                            )
                        ne_id = cdr.serving_ne_id
                        cdr_date = min(cdr.event_timestamp.date(), _end_date)
                        records_by_ne_date[(ne_id, cdr_date)].append(cdr)
                        stats.data_records += 1
                        stats.total_records += 1

        current += _step_delta

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
