"""Single-threaded CDR generation runner.

Orchestrates the full generation pipeline: time-step iteration, event
sampling, CDR generation, and output writing.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from cdr_generator.assets.generator import generate_all_assets
from cdr_generator.assets.models import Cell, NetworkElement, Subscriber
from cdr_generator.config.models import CDRGeneratorConfig, SubscriberProfile
from cdr_generator.engine.poisson import sample_count
from cdr_generator.engine.rates import effective_rate
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
    profiles_by_name = {p.name: p for p in config.subscribers.profiles}

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
    py_rng = random.Random(config.meta.seed)

    # Time range
    start_dt = config.meta.time_range.start.replace(tzinfo=timezone.utc)
    end_dt = config.meta.time_range.end.replace(tzinfo=timezone.utc)
    step_seconds = config.meta.time_step_seconds

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

    current = start_dt
    while current <= end_dt:
        hour = current.hour
        dow = current.weekday()

        for sub in subscribers:
            profile = profiles_by_name.get(sub.profile_name)
            if profile is None:
                continue

            home_cell = cells_by_id.get(sub.home_cell_id)
            if home_cell is None:
                continue

            home_tac = home_cell.tac

            # --- Voice MO ---
            mo_call_lambda = profile.daily_rates.mo_call.params.get("lambda", 0)
            voice_rate = effective_rate(
                base_lambda=mo_call_lambda,
                hourly_weights=profile.hourly_weights.voice,
                dow_multipliers=profile.day_of_week_multipliers.voice,
                hour=hour,
                dow=dow,
                time_step_seconds=step_seconds,
            )
            n_voice = sample_count(voice_rate, np_rng)
            msc_ne = tac_to_ne["msc"].get(home_tac)
            if n_voice > 0 and msc_ne is not None:
                for _ in range(n_voice):
                    # B-party = random subscriber (Phase 2 simplification)
                    callee = subscribers[py_rng.randint(0, len(subscribers) - 1)]
                    if callee.imsi == sub.imsi:
                        callee = subscribers[(subscribers.index(sub) + 1) % len(subscribers)]

                    cdrs = generate_voice_cdr(
                        caller=sub,
                        callee=callee,
                        event_time=current,
                        cell=home_cell,
                        msc=msc_ne,
                        voice_cfg=voice_cfg,
                        rng=np_rng,
                    )
                    for cdr in cdrs:
                        ne_id = cdr.serving_ne_id
                        cdr_date = cdr.event_timestamp.date()
                        records_by_ne_date[(ne_id, cdr_date)].append(cdr)
                        stats.voice_records += 1
                        stats.total_records += 1

            # --- SMS MO ---
            mo_sms_lambda = profile.daily_rates.mo_sms.params.get("lambda", 0)
            sms_rate = effective_rate(
                base_lambda=mo_sms_lambda,
                hourly_weights=profile.hourly_weights.sms,
                dow_multipliers=profile.day_of_week_multipliers.sms,
                hour=hour,
                dow=dow,
                time_step_seconds=step_seconds,
            )
            n_sms = sample_count(sms_rate, np_rng)
            smsc_ne = tac_to_ne["smsc"].get(home_tac)
            if n_sms > 0 and smsc_ne is not None:
                for _ in range(n_sms):
                    recipient = subscribers[py_rng.randint(0, len(subscribers) - 1)]
                    if recipient.imsi == sub.imsi:
                        recipient = subscribers[(subscribers.index(sub) + 1) % len(subscribers)]

                    cdrs = generate_sms_cdr(
                        sender=sub,
                        recipient=recipient,
                        event_time=current,
                        cell=home_cell,
                        smsc=smsc_ne,
                        sms_cfg=sms_cfg,
                        rng=np_rng,
                    )
                    for cdr in cdrs:
                        ne_id = cdr.serving_ne_id
                        cdr_date = cdr.event_timestamp.date()
                        records_by_ne_date[(ne_id, cdr_date)].append(cdr)
                        stats.sms_records += 1
                        stats.total_records += 1

            # --- Data session ---
            data_lambda = profile.daily_rates.data_session.params.get("lambda", 0)
            data_rate = effective_rate(
                base_lambda=data_lambda,
                hourly_weights=profile.hourly_weights.data,
                dow_multipliers=profile.day_of_week_multipliers.data,
                hour=hour,
                dow=dow,
                time_step_seconds=step_seconds,
            )
            n_data = sample_count(data_rate, np_rng)
            sgw_ne = tac_to_ne["sgw"].get(home_tac)
            if n_data > 0 and sgw_ne is not None and pgw_ne is not None:
                for _ in range(n_data):
                    cdrs = generate_data_cdr(
                        subscriber=sub,
                        event_time=current,
                        cell=home_cell,
                        sgw=sgw_ne,
                        pgw=pgw_ne,
                        data_cfg=data_cfg,
                        rng=np_rng,
                    )
                    for cdr in cdrs:
                        ne_id = cdr.serving_ne_id
                        cdr_date = cdr.event_timestamp.date()
                        records_by_ne_date[(ne_id, cdr_date)].append(cdr)
                        stats.data_records += 1
                        stats.total_records += 1

        current += timedelta(seconds=step_seconds)

    # Write output files: sort records and write per (NE, date)
    writer = CsvWriter(output_dir=output_dir)

    for (ne_id, file_date), records in sorted(records_by_ne_date.items()):
        # Sort by event_timestamp
        records.sort(key=lambda r: r.event_timestamp)
        writer.write_file(ne_id=ne_id, file_date=file_date, records=records)
        stats.files_written += 1

    # Also create empty files for NEs/dates with no records
    start_date = start_dt.date()
    end_date = end_dt.date()
    current_date = start_date
    while current_date <= end_date:
        for ne in network_elements:
            key = (ne.id, current_date)
            if key not in records_by_ne_date:
                writer.write_file(ne_id=ne.id, file_date=current_date, records=[])
                stats.files_written += 1
        current_date += timedelta(days=1)

    return stats
