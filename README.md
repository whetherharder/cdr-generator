# CDR Generator

Synthetic Call Detail Record generator for telecom network testing. Produces realistic CSV+gzip files per Network Element per day.

## Features

- Voice (MO/MT), SMS (MO/MT), and Data (SGW/PGW) CDR generation
- Realistic temporal patterns: hourly weights, day-of-week multipliers
- Paired records with consolidation IDs and charging IDs
- Contact book model with Zipf-distributed social graph
- Subscriber mobility and cell handover simulation
- Anomaly injection: orphaned records, missing fields, corrupt values, timestamp anomalies, duplicates
- Special event modeling: concerts, outages, recurring patterns with compound effects
- Call forwarding with 3-record CDR chains
- Vendor-specific extensions (Ericsson, Huawei, Nokia)
- Deterministic seeding for reproducible output
- Multiprocess generation with subscriber sharding
- Gzipped CSV output partitioned by network element and date

## Prerequisites

- Python 3.11+
- pip

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Docker

```bash
docker compose build
docker compose run claude
```

Inside the container the virtualenv activates automatically.

## Quick Start

```bash
# 1. Validate config
cdrgen validate-config --config cdr_generator_config.yaml

# 2. Generate assets (cells, network elements, subscribers)
cdrgen generate-assets --config cdr_generator_config.yaml --assets-dir ./assets

# 3. Generate CDRs
cdrgen generate --config cdr_generator_config.yaml --assets-dir ./assets --output-dir ./output

# 4. Check output
ls -lh output/
zcat output/CDR_msc-01_20250101.csv.gz | head
```

Override parameters without editing the config:

```bash
cdrgen generate --config cdr_generator_config.yaml \
  --override subscribers.total_count=500 \
  --override meta.time_step_seconds=30
```

Multiprocess generation:

```bash
cdrgen generate --config cdr_generator_config.yaml --workers 16
```

Dry run (create header-only files, no CDRs):

```bash
cdrgen generate --config cdr_generator_config.yaml --dry-run
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `cdrgen validate-config --config FILE` | Validate YAML config against Pydantic schema |
| `cdrgen generate-assets --config FILE --assets-dir DIR` | Pre-generate cells, NEs, subscribers to JSON |
| `cdrgen validate-assets --assets-dir DIR` | Check asset referential integrity |
| `cdrgen generate --config FILE [OPTIONS]` | Generate CDR files |
| `cdrgen estimate --config FILE` | Estimate output volume (planned) |

### `generate` options

| Flag | Description |
|------|-------------|
| `--assets-dir DIR` | Reuse previously generated assets |
| `--output-dir DIR` | Override output path from config |
| `--workers N` | Number of parallel workers (default: 1) |
| `--dry-run` | Write headers only, no CDR records |
| `--override KEY=VALUE` | Dot-path config override (repeatable) |

## Configuration

See [`cdr_generator_config.yaml`](cdr_generator_config.yaml) for the full reference configuration (v0.7).

| Section | Description |
|---------|-------------|
| `meta` | Seed, time range, output format, parallelism |
| `network` | Operator info, network elements, cell definitions |
| `subscribers` | Profiles with daily rates, hourly weights, mobility config |
| `events` | Voice/SMS/Data parameters, distributions, concurrency limits |
| `anomalies` | Controlled anomaly injection (orphaned, missing, corrupt, timestamp, duplicates) |
| `special_events` | Time-bounded event overlays with rate multipliers |
| `vendor_extensions` | Vendor-specific CDR field templates |

## Architecture

```
YAML config
  -> config/loader.py (parse + validate + apply --override)
  -> CDRGeneratorConfig (Pydantic v2 model tree)
  |-> assets/generator.py -> Cells, NEs, Subscribers -> assets/store.py (JSON + manifest)
  |-> engine/runner.py (single-threaded orchestrator)
       | engine/time_step.py (iterate start -> end by step_seconds)
       | engine/rates.py (effective_rate per subscriber per event type)
       | engine/poisson.py (sample event count from rate)
       | engine/b_party.py (contact book / external / random)
       | engine/mobility.py (position by time-of-day + handover)
       | engine/anomalies.py (5-stage anomaly pipeline)
       | engine/special_events.py (event-based rate multipliers)
       | generators/voice.py | sms.py | data.py (produce CDR pairs)
       | generators/extensions.py (vendor fields)
       | writer/csv_writer.py -> CDR_{ne_id}_{YYYYMMDD}.csv.gz
  |-> engine/orchestrator.py (multiprocess: shard subscribers, merge results)
```

## Output Format

One gzipped CSV file per network element per day:

```
CDR_{ne_id}_{YYYYMMDD}.csv.gz
```

Records are sorted by `event_timestamp`. Each CSV contains 26 fields including `record_type`, `served_imsi`, `served_msisdn`, `event_timestamp`, `duration`, `cell_id`, `serving_ne_id`, `consolidation_id`, and vendor extensions.

## Performance

### Requirements

| Metric | Target |
|--------|--------|
| Single-core throughput | >= 50,000 CDR/sec |
| Scaling | Linear with subscriber count (<=20% degradation at 2x) |

Production scale: 10M+ subscribers over months of simulated time (billions of CDR records). At 50k CDR/sec per core this is viable with multiprocess parallelism.

Current single-threaded baseline: ~1,500 records/sec. Optimization is in progress.

### Running benchmarks

```bash
# Pytest benchmark suite (includes 50k threshold gate test)
pytest tests/test_performance.py -v

# Detailed profiling
python tests/benchmark_generation.py --subscribers 500 --profile

# Skip benchmarks for faster test runs
pytest tests/ -q -m "not benchmark"
```

## Testing

```bash
# All tests
pytest tests/ -q --tb=short

# Skip slow benchmarks
pytest tests/ -q -m "not benchmark"

# Single test file
pytest tests/test_config.py -q

# Single test
pytest tests/test_config.py::TestClass::test_name -v

# Benchmarks only
pytest tests/test_performance.py -v
```

## Development

### Gitflow

- `main` and `develop` are protected branches (PR with review required)
- Feature branches: `feature/*` from `develop`
- Hotfix branches: `hotfix/*` from `main`

```bash
git checkout develop && git pull
git checkout -b feature/my-feature
# ... develop ...
pytest tests/ -q --tb=short
git push -u origin feature/my-feature
gh pr create --base develop
```

### Code quality

```bash
ruff format cdr_generator tests
ruff check cdr_generator tests
```

## License

MIT
