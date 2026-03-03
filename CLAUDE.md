# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CDR Generator** — Python CLI tool for generating synthetic Call Detail Records (CDR) for telecom network testing. Produces realistic CSV+gzip files per Network Element per day.

**Status**: All 6 phases complete (435 tests passing, 1 skipped). Performance optimization branch achieves ~57k CDR/sec (14% above 50k target). See `PROGRESS.md` for phase history. `cdr_generator_config.yaml` is the reference config (v0.7).

## Commands

```bash
# Setup
source .venv/bin/activate
pip install -e .

# Testing
pytest tests/ -q --tb=short              # all tests
pytest tests/test_config.py -q            # single file
pytest tests/test_config.py::TestClass::test_name -v  # single test

# Linting
ruff format cdr_generator tests           # format
ruff check cdr_generator tests            # lint

# CLI usage
cdrgen validate-config --config cdr_generator_config.yaml
cdrgen generate-assets --config cdr_generator_config.yaml --assets-dir ./assets
cdrgen validate-assets --assets-dir ./assets
cdrgen generate --config cdr_generator_config.yaml --assets-dir ./assets --output-dir ./output
cdrgen generate --config cdr_generator_config.yaml --dry-run
cdrgen generate --config cdr_generator_config.yaml --override subscribers.total_count=500
```

## Architecture

### Data Flow

```
YAML config
  → config/loader.py (parse + validate + apply --override)
  → CDRGeneratorConfig (Pydantic v2 model tree)
  ├→ assets/generator.py → Cells, NEs, Subscribers → assets/store.py (disk + manifest)
  └→ engine/runner.py (fast path + slow path orchestrator)
       ├ engine/time_step.py (iterate start→end by step_seconds)
       ├ engine/rates.py (effective_rate per subscriber per event type)
       ├ engine/poisson.py (sample event count from rate)
       ├ engine/rng_buffer.py (_RngBuffer: 2048-element batched RNG)
       ├ engine/b_party.py (contact book / external / random selection)
       ├ engine/mobility.py (position by time-of-day + handover)
       ├ engine/anomalies.py (5-stage anomaly pipeline)
       ├ engine/special_events.py (event-based rate multipliers + cell disabling)
       ├ engine/concurrency.py (per-subscriber concurrency limits)
       ├ engine/orchestrator.py (multiprocess: shard subscribers, merge results)
       ├ generators/voice.py | sms.py | data.py (produce CDRRecord pairs)
       ├ generators/extensions.py (vendor field injection)
       └ writer/csv_writer.py → CDR_{ne_id}_{YYYYMMDD}.csv.gz
```

### Key Pairing Logic

- **Voice**: MO+MT records share `consolidation_id` (UUID). MT has small timestamp jitter.
- **SMS**: MO+MT share `consolidation_id`. MT delayed by sampled `delivery_delay`.
- **Data**: SGW+PGW share `charging_id` (int). Long sessions split into multiple partial record pairs.

### Config System

Pydantic v2 models in `config/models.py`. Root: `CDRGeneratorConfig` with sections: `meta`, `network`, `subscribers`, `events`, `anomalies`, `special_events`, `vendor_extensions`. Loading pipeline: YAML parse → optional JSON Schema check → dot-path overrides → Pydantic validation.

Distributions are described as `{type: "lognormal"|"normal"|"poisson"|..., params: {...}}` and sampled via `engine/distributions.py`.

### Rate Calculation

`effective_rate = base_lambda * (hourly_weight[h] / sum(weights)) * dow_mult[dow] * (step_seconds / 3600)`

Weights are auto-normalized. Output is the Poisson lambda for `engine/poisson.py`.

### Writer

Two classes in `writer/csv_writer.py`: `CSVWriter` (low-level, header + append) and `CsvWriter` (high-level, write sorted file in one call). Output: one gzipped CSV per `(ne_id, date)`.

### Deterministic Seeding

Global seed from config → `numpy.random.Generator` instances. Multiprocess: `worker_seed = derive_worker_seed(global_seed, shard_id)` in `engine/orchestrator.py`. Output is bit-identical regardless of worker count for the same seed.

### Performance Architecture

The hot path in `engine/runner.py` uses two execution modes:

**Fast path** (normal case): Groups time steps by clock hour. Builds a rate matrix `(n_subs, 3)` once per hour via `_build_rate_matrix()`, then samples all counts for every step in the hour with one call: `rng.poisson(rate_matrix, size=(n_steps, n_subs, 3))`. Only `np.argwhere(counts.any(axis=2))` pairs enter the generator loop.

**Slow path** (disabled cells active): Per-step, per-subscriber loop. Activates when `special_event_engine.get_active_effects()` returns non-empty `disabled_cells`.

**`_RngBuffer`** (`engine/rng_buffer.py`): All generators receive `_buf: _RngBuffer`. Use `_buf.get_uuid()`, `_buf.get_lognormal()`, `_buf.get_int()`, `_buf.get_choice()` — never direct `rng.*` calls in the hot path. Buffer refills in batches of 2048 elements.

**Config caches**: `build_voice_cfg_cache()` → `_VoiceCfgCache` and `build_data_cfg_cache()` → `_DataCfgCache` are built once before the loop. Pass as `_vcfg` / `_cfg` kwargs. When a special event overrides `success_rate`, `_step_vcfg` is set to `None` so the generator falls back to dict lookup.

**`_fmt_dt` lru_cache**: Module-level cache in `models/cdr.py`. Persists across `run_generation()` calls intentionally.

**Benchmark warmup**: `tests/benchmark_generation.py` calls `_prime_specializer()` at import time (before test timing starts) to warm CPython's adaptive specializer and pre-fill the `_fmt_dt` cache. This is why `pytest tests/test_performance.py` passes reliably.

## Development Rules

### Gitflow
- `main` and `develop` are protected — PR with 1 approval required
- Feature branches: `feature/phase-N-*` from `develop`, hotfix: `hotfix/*` from `main`
- Before work: `git checkout develop && git pull && git checkout -b feature/...`
- After feature: commit → push → `gh pr create --base develop`
- Codex (chatgpt-codex-connector) auto-reviews PRs — address its comments before merge
- Check Codex comments: `gh pr view <N> --comments`
- Re-trigger review: `gh pr comment <N> --body "@codex review"`
- Merge only when: tests green + Codex approved + 1 human approval

### Development Workflow
- Activate venv before any Python command: `source .venv/bin/activate`
- Run tests before every commit: `pytest tests/ -q --tb=short`
- Each phase = separate feature branch + separate PR
- Implement phases sequentially without asking for confirmation between them
- TDD: write tests first, then implementation

### Agent Team (cdr-gen-phases)

Team for phases 3-6 implementation. Per-phase workflow: PM decomposes → Architect designs APIs + QA writes tests (parallel) → Developer implements → QA validates → commit + next phase.

| Role | Agent Type | Responsibilities |
|------|-----------|-----------------|
| **PM** | general-purpose | Decomposes each phase into sub-tasks <=4h, sets dependencies, tracks progress, coordinates handoffs between agents |
| **Architect** | system-architect | Designs module interfaces and data contracts before implementation. Writes API specs as docstrings/type hints in stub files |
| **Developer** | python-expert | Implements modules after Architect defines API. TDD: makes QA's tests green. Commits working code only |
| **QA** | quality-engineer | Writes tests in parallel with Architect (based on API specs). Validates implementation against plan.md acceptance criteria |

Coordination rules:
- Architect and QA start in parallel once PM creates sub-tasks
- Developer starts only after Architect's API design is ready
- QA runs full test suite after Developer finishes each sub-task
- All 436 tests (1 skipped) must stay green throughout
- Each phase = feature branch from develop, separate PR

### Code Quality
- No stub implementations in commits (`pass`, `raise NotImplementedError`, `TODO`)
- Public module API must match what tests import — verify before commit
- `pytest tests/ -q` — final check before `git push`
