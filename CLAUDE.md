# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CDR Generator** — a Python CLI tool for generating synthetic Call Detail Records (CDR) for telecom network testing. Produces realistic CSV+gzip files per Network Element per day, with configurable subscriber profiles, anomalies, and special events.

**Status**: Implementation in progress. See `plan.md` for the 6-phase roadmap. `cdr_generator_config.yaml` is the reference config (v0.7).

## Commands

Once implemented, the CLI entry point is `cdrgen`:

```bash
cdrgen generate-assets --config config.yaml --assets-dir ./assets
cdrgen validate-assets --assets-dir ./assets
cdrgen validate-config --config config.yaml
cdrgen generate --config config.yaml --assets-dir ./assets --output-dir ./output
cdrgen generate --config config.yaml --workers 16 --progress
cdrgen estimate --config config.yaml
```

Key flags: `--dry-run` (write headers only), `--workers N`, `--override dot.path=value`.

For development/testing (Python):
```bash
pip install -e .
pytest tests/
pytest tests/test_config.py::test_schema_validation  # single test
python -m cdrgen ...  # before install
```

## Architecture

The planned module layout (from `plan.md`):

```
config/
  loader.py        # YAML → dataclass, jsonschema validation, --override support
  schema.json      # JSON Schema for config file

assets/
  models.py        # Dataclasses: Cell, NetworkElement, Subscriber
  generator.py     # Generate cells, NE, subscribers from config
  store.py         # Save/load/validate assets, manifest with config hash
  contact_book.py  # Zipf-distributed contact graph, asymmetric
  external_numbers.py

engine/
  time_step.py     # Main iteration loop over time steps
  rates.py         # effective_rate = base × hourly_weight × dow × special_event
  poisson.py       # Poisson sampling for event counts
  mobility.py      # Pure function: position(home, work, profile, T, seed)
  b_party.py       # B-party selection: contact_book / external / random
  anomalies.py     # Pipeline: orphaned → missing → corrupt → timestamps → duplicates
  special_events.py
  concurrency.py   # Enforce 1 voice + 1 data + N sms per subscriber per step
  orchestrator.py  # Phase 6: shard subscribers, spawn generator + writer workers
  generator_worker.py

generators/
  voice.py         # MO/MT voice, call forwarding (produces 2 CDRs), duration, causes
  data.py          # SGW+PGW paired records, partial records for long sessions
  sms.py           # MO/MT SMS
  extensions.py    # Vendor-specific fields → base64 JSON

models/
  cdr.py           # CDR record dataclass → CSV row dict

writer/
  csv_writer.py    # CSV+gzip writer, one file per (NE, date), sorted by event_timestamp
  writer_worker.py # Phase 6: receive CDRs, sort, write

cli.py             # Click/argparse entrypoint
```

## Key Design Concepts

**Config system**: YAML loaded into dataclasses via `config/loader.py`. Dot-path `--override` applied after load (e.g. `subscribers.total_count=50000`). Config hash stored in assets manifest to detect staleness.

**Asset generation** (run once, reused): Generates subscriber list, cell topology, contact books. Assets stored on disk with a manifest; regenerated only when config changes.

**Rate calculation**: `effective_rate = poisson_lambda × hourly_weight[h] × dow_multiplier[d] × special_event_multiplier`. Normalized internally — weights don't need to sum to 1.

**CDR pairing**: Voice MO+MT share a `consolidation_id`. SGW+PGW data records share a `charging_id`. Orphaned anomalies intentionally break these pairs.

**Output structure**: One CSV+gzip file per `(ne_id, date)`. Filename: `CDR_{ne_id}_{YYYYMMDD}.csv.gz`. Records sorted by `event_timestamp` within each file.

**Multiprocess architecture** (Phase 6): Orchestrator shards subscribers across N generator workers. Each (ne_id, date) gets a dedicated writer worker. Worker seed = `f(global_seed, shard_id)` for determinism. Assets loaded via shared memory (not copied per-worker).

**Anomaly pipeline** order: orphaned → missing_fields → corrupt_values → timestamp_anomalies → duplicate_records. Applied by generator workers before sending to writers.

## Configuration Reference

See `cdr_generator_config.yaml` for full annotated config. Top-level sections:
- `meta` — seed, time_range, output format, parallelism
- `network` — operator, NE elements (msc/sgw/pgw/smsc), cells (inline or file)
- `subscribers` — total_count, profiles with weights, contact_book, external_numbers
- `events` — voice/data/sms parameters, distributions, concurrency limits
- `anomalies` — per-type rates and targets
- `special_events` — time-ranged or recurring event multipliers
- `vendor_extensions` — per-vendor field generators

Distributions use: `{type: lognormal|poisson|zipf|categorical|..., params: {...}}`.
