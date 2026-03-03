"""Tests for Phase 6: Multiprocess CDR generation orchestration.

Covers:
- Subscriber sharding correctness
- Worker seed derivation (deterministic, unique per shard)
- Orchestrate function with single and multiple workers
- Determinism: workers=1 and workers=N produce identical CDR content
- Edge cases: more workers than subscribers, invalid worker counts
- Output file count invariance (NE x days regardless of worker count)
- Record sorting within each output file
"""

from __future__ import annotations

import csv
import gzip
import pathlib
from collections import Counter

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_CONFIG_PATH = REPO_ROOT / "cdr_generator_config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_gz_csv(path: pathlib.Path) -> tuple[list[str], list[list[str]]]:
    """Read a gzip CSV and return (header, rows). Skips metadata comment lines."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(lines)
    header = next(reader)
    rows = list(reader)
    return header, rows


def _collect_records_by_file(
    output_dir: pathlib.Path,
) -> dict[str, list[dict[str, str]]]:
    """Collect CDR rows grouped by filename from all .csv.gz files."""
    result: dict[str, list[dict[str, str]]] = {}
    for gz_file in sorted(output_dir.rglob("*.csv.gz")):
        header, rows = _read_gz_csv(gz_file)
        records = [dict(zip(header, row)) for row in rows]
        result[gz_file.name] = records
    return result


def _collect_all_records(output_dir: pathlib.Path) -> list[dict[str, str]]:
    """Collect all CDR rows from all .csv.gz files in the output directory."""
    all_rows = []
    for gz_file in sorted(output_dir.rglob("*.csv.gz")):
        header, rows = _read_gz_csv(gz_file)
        for row in rows:
            all_rows.append(dict(zip(header, row)))
    return all_rows


def _make_multiprocess_config(tmp_path: pathlib.Path, workers: int = 1) -> pathlib.Path:
    """Create a small config optimized for fast multiprocess tests.

    Uses 30 subscribers, 1 day, step=3600 for speed.
    """
    with open(SAMPLE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    cfg["meta"]["seed"] = 42
    cfg["meta"]["time_range"]["start"] = "2025-01-15T00:00:00Z"
    cfg["meta"]["time_range"]["end"] = "2025-01-15T23:59:59Z"
    cfg["meta"]["time_step_seconds"] = 3600
    cfg["meta"]["parallelism"]["workers"] = workers

    cfg["subscribers"]["total_count"] = 30

    cfg["anomalies"]["enabled"] = False
    cfg["special_events"] = []

    out_dir = tmp_path / "output"
    out_dir.mkdir(exist_ok=True)
    cfg["meta"]["output"]["path"] = str(out_dir)

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path


# ---------------------------------------------------------------------------
# Module availability check
# ---------------------------------------------------------------------------

try:
    from cdr_generator.engine.orchestrator import orchestrate, shard_subscribers

    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False

skip_if_not_implemented = pytest.mark.skipif(
    not _ORCHESTRATOR_AVAILABLE,
    reason="cdr_generator.engine.orchestrator not yet implemented",
)


# ---------------------------------------------------------------------------
# Tests: shard_subscribers
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestShardSubscribers:
    """shard_subscribers must split subscribers into N shards correctly."""

    def test_even_split(self) -> None:
        """10 subscribers into 2 shards -> 5 each."""
        items = list(range(10))
        shards = shard_subscribers(items, 2)
        assert len(shards) == 2
        assert len(shards[0]) == 5
        assert len(shards[1]) == 5

    def test_uneven_split(self) -> None:
        """7 subscribers into 3 shards -> sizes 3, 2, 2 (or similar)."""
        items = list(range(7))
        shards = shard_subscribers(items, 3)
        assert len(shards) == 3
        total = sum(len(s) for s in shards)
        assert total == 7

    def test_every_subscriber_exactly_once(self) -> None:
        """Every subscriber appears in exactly one shard, no duplicates."""
        items = list(range(50))
        shards = shard_subscribers(items, 4)
        flat = []
        for shard in shards:
            flat.extend(shard)
        assert sorted(flat) == items, "All subscribers must appear exactly once"

    def test_single_shard(self) -> None:
        """Single shard contains all subscribers."""
        items = list(range(20))
        shards = shard_subscribers(items, 1)
        assert len(shards) == 1
        assert shards[0] == items

    def test_more_shards_than_subscribers(self) -> None:
        """3 subscribers into 8 shards -> 3 non-empty + 5 empty shards."""
        items = list(range(3))
        shards = shard_subscribers(items, 8)
        assert len(shards) == 8
        non_empty = [s for s in shards if len(s) > 0]
        flat = []
        for shard in shards:
            flat.extend(shard)
        assert sorted(flat) == items, "All subscribers must still appear exactly once"
        # At least some shards should be empty when more shards than subscribers
        assert len(non_empty) <= len(items)

    def test_empty_subscribers(self) -> None:
        """Empty subscriber list into any number of shards -> all empty."""
        shards = shard_subscribers([], 4)
        assert len(shards) == 4
        for shard in shards:
            assert len(shard) == 0

    def test_preserves_order_within_shards(self) -> None:
        """Subscribers within each shard should maintain relative order."""
        items = list(range(12))
        shards = shard_subscribers(items, 3)
        for shard in shards:
            # Each shard should be a sorted subsequence (round-robin or chunk)
            assert shard == sorted(shard), (
                "Subscriber order within shard should be preserved"
            )


# ---------------------------------------------------------------------------
# Tests: worker seed derivation
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestWorkerSeedDerivation:
    """Worker seeds must be deterministic and unique per shard."""

    def test_different_seeds_per_shard(self) -> None:
        """Each shard should get a different seed."""
        # Import the seed derivation function (may be internal)
        try:
            from cdr_generator.engine.orchestrator import _derive_worker_seed
        except ImportError:
            pytest.skip("_derive_worker_seed not exposed")

        global_seed = 42
        seeds = [_derive_worker_seed(global_seed, i) for i in range(10)]
        assert len(set(seeds)) == 10, "All worker seeds must be unique"

    def test_seeds_are_deterministic(self) -> None:
        """Same global_seed and shard_id always produce same worker seed."""
        try:
            from cdr_generator.engine.orchestrator import _derive_worker_seed
        except ImportError:
            pytest.skip("_derive_worker_seed not exposed")

        global_seed = 42
        seed_a = _derive_worker_seed(global_seed, 3)
        seed_b = _derive_worker_seed(global_seed, 3)
        assert seed_a == seed_b, "Worker seed must be deterministic"

    def test_different_global_seeds_produce_different_worker_seeds(self) -> None:
        """Different global seeds produce different worker seeds for same shard."""
        try:
            from cdr_generator.engine.orchestrator import _derive_worker_seed
        except ImportError:
            pytest.skip("_derive_worker_seed not exposed")

        seed_a = _derive_worker_seed(42, 0)
        seed_b = _derive_worker_seed(99, 0)
        assert seed_a != seed_b, (
            "Different global seeds should give different worker seeds"
        )


# ---------------------------------------------------------------------------
# Tests: orchestrate basic functionality
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestOrchestrateBasic:
    """Basic orchestrate() functionality."""

    def test_single_worker_produces_records(self, tmp_path: pathlib.Path) -> None:
        """orchestrate with workers=1 should produce CDR records."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=1)
        config = load_config(config_path)
        stats = orchestrate(config, workers=1, dry_run=False)

        assert stats.total_records > 0, "Must produce records"
        assert stats.files_written > 0, "Must write output files"

    def test_multi_worker_produces_records(self, tmp_path: pathlib.Path) -> None:
        """orchestrate with workers=2 should produce CDR records."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=2)
        config = load_config(config_path)
        stats = orchestrate(config, workers=2, dry_run=False)

        assert stats.total_records > 0, "Must produce records"
        assert stats.files_written > 0, "Must write output files"

    def test_dry_run_creates_header_only_files(self, tmp_path: pathlib.Path) -> None:
        """orchestrate with dry_run=True should create header-only files."""
        from cdr_generator.config.loader import load_config
        from cdr_generator.models.cdr import CDR_FIELDS

        config_path = _make_multiprocess_config(tmp_path, workers=2)
        config = load_config(config_path)
        stats = orchestrate(config, workers=2, dry_run=True)

        assert stats.files_written > 0
        output_dir = pathlib.Path(config.meta.output.path)
        for gz_file in output_dir.rglob("*.csv.gz"):
            header, rows = _read_gz_csv(gz_file)
            assert header == CDR_FIELDS
            assert len(rows) == 0, f"Dry run must produce 0 records in {gz_file.name}"

    def test_returns_generation_stats(self, tmp_path: pathlib.Path) -> None:
        """orchestrate should return a GenerationStats object with counts."""
        from cdr_generator.config.loader import load_config
        from cdr_generator.engine.runner import GenerationStats

        config_path = _make_multiprocess_config(tmp_path, workers=1)
        config = load_config(config_path)
        stats = orchestrate(config, workers=1, dry_run=False)

        assert isinstance(stats, GenerationStats)
        assert stats.voice_records > 0
        assert stats.data_records > 0
        assert stats.total_records > 0
        # Sub-counts (voice/sms/data) are pre-anomaly; total_records is
        # adjusted post-anomaly for duplicates and orphaned records, so
        # the two will differ when anomalies are applied.
        assert stats.total_records >= stats.voice_records


# ---------------------------------------------------------------------------
# Tests: output file count invariance
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestOutputFileCount:
    """Output file count should always be NE x days regardless of worker count."""

    def test_file_count_same_for_different_workers(
        self, tmp_path: pathlib.Path
    ) -> None:
        """workers=1 and workers=3 must produce the same number of output files."""
        from cdr_generator.config.loader import load_config

        # Run with workers=1
        dir1 = tmp_path / "run1"
        dir1.mkdir()
        config_path_1 = _make_multiprocess_config(dir1, workers=1)
        config_1 = load_config(config_path_1)
        stats_1 = orchestrate(config_1, workers=1, dry_run=False)

        # Run with workers=3
        dir2 = tmp_path / "run2"
        dir2.mkdir()
        config_path_2 = _make_multiprocess_config(dir2, workers=3)
        config_2 = load_config(config_path_2)
        stats_2 = orchestrate(config_2, workers=3, dry_run=False)

        assert stats_1.files_written == stats_2.files_written, (
            f"File count must be invariant: workers=1 produced {stats_1.files_written}, "
            f"workers=3 produced {stats_2.files_written}"
        )

    def test_file_count_matches_ne_times_days(self, tmp_path: pathlib.Path) -> None:
        """File count should equal number of NEs times number of days."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=2)
        config = load_config(config_path)
        stats = orchestrate(config, workers=2, dry_run=False)

        n_nes = len(config.network.elements)
        # 1 day in our test config
        n_days = 1
        expected_files = n_nes * n_days

        assert stats.files_written == expected_files, (
            f"Expected {expected_files} files ({n_nes} NEs x {n_days} days), "
            f"got {stats.files_written}"
        )


# ---------------------------------------------------------------------------
# Tests: record sorting
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestRecordSorting:
    """Records within each output file must be sorted by event_timestamp."""

    def test_sorted_with_single_worker(self, tmp_path: pathlib.Path) -> None:
        """Records sorted by event_timestamp when workers=1."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=1)
        config = load_config(config_path)
        orchestrate(config, workers=1, dry_run=False)

        output_dir = pathlib.Path(config.meta.output.path)
        self._assert_all_files_sorted(output_dir)

    def test_sorted_with_multiple_workers(self, tmp_path: pathlib.Path) -> None:
        """Records sorted by event_timestamp when workers=3."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=3)
        config = load_config(config_path)
        orchestrate(config, workers=3, dry_run=False)

        output_dir = pathlib.Path(config.meta.output.path)
        self._assert_all_files_sorted(output_dir)

    @staticmethod
    def _assert_all_files_sorted(output_dir: pathlib.Path) -> None:
        """Verify all output files have records sorted by event_timestamp."""
        checked = 0
        for gz_file in sorted(output_dir.rglob("*.csv.gz")):
            header, rows = _read_gz_csv(gz_file)
            if not rows:
                continue
            ts_idx = header.index("event_timestamp")
            timestamps = [row[ts_idx] for row in rows]
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i - 1], (
                    f"Records not sorted in {gz_file.name}: "
                    f"'{timestamps[i - 1]}' > '{timestamps[i]}' at row {i}"
                )
            checked += 1
        assert checked > 0, "Should have at least one file with records"


# ---------------------------------------------------------------------------
# Tests: determinism (critical)
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestDeterminism:
    """Multiprocess generation must be deterministic.

    Workers=1 and workers=N should produce identical CDR content per output file.
    """

    def test_single_vs_multi_worker_same_records(self, tmp_path: pathlib.Path) -> None:
        """workers=1 and workers=2 must produce identical CDR records per file."""
        from cdr_generator.config.loader import load_config

        # Run with workers=1
        dir1 = tmp_path / "run1"
        dir1.mkdir()
        config_path_1 = _make_multiprocess_config(dir1, workers=1)
        config_1 = load_config(config_path_1)
        orchestrate(config_1, workers=1, dry_run=False)

        # Run with workers=2
        dir2 = tmp_path / "run2"
        dir2.mkdir()
        config_path_2 = _make_multiprocess_config(dir2, workers=2)
        config_2 = load_config(config_path_2)
        orchestrate(config_2, workers=2, dry_run=False)

        output_dir_1 = pathlib.Path(config_1.meta.output.path)
        output_dir_2 = pathlib.Path(config_2.meta.output.path)

        records_1 = _collect_records_by_file(output_dir_1)
        records_2 = _collect_records_by_file(output_dir_2)

        # Same set of output files
        assert set(records_1.keys()) == set(records_2.keys()), (
            f"Output file names must match.\n"
            f"  workers=1: {sorted(records_1.keys())}\n"
            f"  workers=2: {sorted(records_2.keys())}"
        )

        # Same record count per file
        for fname in sorted(records_1.keys()):
            recs_1 = records_1[fname]
            recs_2 = records_2[fname]
            assert len(recs_1) == len(recs_2), (
                f"Record count mismatch in {fname}: "
                f"workers=1 has {len(recs_1)}, workers=2 has {len(recs_2)}"
            )

        # Same record content per file (comparing field by field)
        for fname in sorted(records_1.keys()):
            recs_1 = records_1[fname]
            recs_2 = records_2[fname]
            for i, (r1, r2) in enumerate(zip(recs_1, recs_2)):
                assert r1 == r2, (
                    f"Record mismatch in {fname} at row {i}:\n"
                    f"  workers=1: {r1}\n"
                    f"  workers=2: {r2}"
                )

    def test_repeated_runs_same_output(self, tmp_path: pathlib.Path) -> None:
        """Two runs with the same config produce identical output."""
        from cdr_generator.config.loader import load_config

        # First run
        dir1 = tmp_path / "run1"
        dir1.mkdir()
        config_path_1 = _make_multiprocess_config(dir1, workers=2)
        config_1 = load_config(config_path_1)
        orchestrate(config_1, workers=2, dry_run=False)

        # Second run
        dir2 = tmp_path / "run2"
        dir2.mkdir()
        config_path_2 = _make_multiprocess_config(dir2, workers=2)
        config_2 = load_config(config_path_2)
        orchestrate(config_2, workers=2, dry_run=False)

        output_dir_1 = pathlib.Path(config_1.meta.output.path)
        output_dir_2 = pathlib.Path(config_2.meta.output.path)

        records_1 = _collect_records_by_file(output_dir_1)
        records_2 = _collect_records_by_file(output_dir_2)

        assert set(records_1.keys()) == set(records_2.keys())

        for fname in sorted(records_1.keys()):
            recs_1 = records_1[fname]
            recs_2 = records_2[fname]
            assert len(recs_1) == len(recs_2), (
                f"Record count mismatch in {fname} across runs"
            )
            for i, (r1, r2) in enumerate(zip(recs_1, recs_2)):
                assert r1 == r2, f"Non-deterministic output in {fname} at row {i}"

    def test_total_records_match_across_worker_counts(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Total record count must be identical regardless of worker count."""
        from cdr_generator.config.loader import load_config

        # Run with workers=1
        dir1 = tmp_path / "run1"
        dir1.mkdir()
        config_path_1 = _make_multiprocess_config(dir1, workers=1)
        config_1 = load_config(config_path_1)
        stats_1 = orchestrate(config_1, workers=1, dry_run=False)

        # Run with workers=4
        dir2 = tmp_path / "run2"
        dir2.mkdir()
        config_path_2 = _make_multiprocess_config(dir2, workers=4)
        config_2 = load_config(config_path_2)
        stats_2 = orchestrate(config_2, workers=4, dry_run=False)

        assert stats_1.total_records == stats_2.total_records, (
            f"Total records mismatch: workers=1 has {stats_1.total_records}, "
            f"workers=4 has {stats_2.total_records}"
        )
        assert stats_1.voice_records == stats_2.voice_records
        assert stats_1.sms_records == stats_2.sms_records
        assert stats_1.data_records == stats_2.data_records


# ---------------------------------------------------------------------------
# Tests: CDR pairing integrity with multiprocess
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestPairingIntegrity:
    """CDR pairing must remain correct when using multiple workers."""

    def test_sgw_pgw_pairs_complete(self, tmp_path: pathlib.Path) -> None:
        """Every charging_id must appear in both SGW and PGW records."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=2)
        config = load_config(config_path)
        orchestrate(config, workers=2, dry_run=False)

        output_dir = pathlib.Path(config.meta.output.path)
        all_records = _collect_all_records(output_dir)

        sgw_ids: Counter[str] = Counter()
        pgw_ids: Counter[str] = Counter()
        for rec in all_records:
            rt = rec.get("record_type", "")
            cid = rec.get("charging_id", "")
            if not cid:
                continue
            if rt == "sgw_data":
                sgw_ids[cid] += 1
            elif rt == "pgw_data":
                pgw_ids[cid] += 1

        assert set(sgw_ids.keys()) == set(pgw_ids.keys()), (
            f"SGW and PGW charging_ids must match.\n"
            f"  SGW-only: {set(sgw_ids.keys()) - set(pgw_ids.keys())}\n"
            f"  PGW-only: {set(pgw_ids.keys()) - set(sgw_ids.keys())}"
        )

    def test_mo_mt_pairs_complete(self, tmp_path: pathlib.Path) -> None:
        """Every consolidation_id in voice records should appear with MO and MT."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=2)
        config = load_config(config_path)
        # Use high success rate for cleaner pairing
        config.events.voice.success_rate = 1.0
        orchestrate(config, workers=2, dry_run=False)

        output_dir = pathlib.Path(config.meta.output.path)
        all_records = _collect_all_records(output_dir)

        voice_records = [
            r for r in all_records if r.get("record_type", "") in ("mo_call", "mt_call")
        ]
        if not voice_records:
            pytest.skip("No voice records generated")

        by_cid: dict[str, list[str]] = {}
        for rec in voice_records:
            cid = rec.get("consolidation_id", "")
            if not cid:
                continue
            by_cid.setdefault(cid, []).append(rec["record_type"])

        for cid, types in by_cid.items():
            assert "mo_call" in types, f"consolidation_id={cid} missing MO record"
            assert "mt_call" in types, f"consolidation_id={cid} missing MT record"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestEdgeCases:
    """Edge cases for orchestrate()."""

    def test_workers_zero_raises_error(self, tmp_path: pathlib.Path) -> None:
        """workers=0 should raise a ValueError."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=1)
        config = load_config(config_path)

        with pytest.raises((ValueError, Exception)):
            orchestrate(config, workers=0, dry_run=False)

    def test_workers_negative_raises_error(self, tmp_path: pathlib.Path) -> None:
        """workers=-1 should raise a ValueError."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=1)
        config = load_config(config_path)

        with pytest.raises((ValueError, Exception)):
            orchestrate(config, workers=-1, dry_run=False)

    def test_more_workers_than_subscribers(self, tmp_path: pathlib.Path) -> None:
        """8 workers with 30 subscribers should work fine."""
        from cdr_generator.config.loader import load_config

        config_path = _make_multiprocess_config(tmp_path, workers=8)
        config = load_config(config_path)
        stats = orchestrate(config, workers=8, dry_run=False)

        assert stats.total_records > 0, (
            "Should produce records even with excess workers"
        )
        assert stats.files_written > 0

    def test_single_subscriber(self, tmp_path: pathlib.Path) -> None:
        """A single subscriber with multiple workers should work."""
        from cdr_generator.config.loader import load_config

        with open(SAMPLE_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        cfg["meta"]["seed"] = 42
        cfg["meta"]["time_range"]["start"] = "2025-01-15T00:00:00Z"
        cfg["meta"]["time_range"]["end"] = "2025-01-15T23:59:59Z"
        cfg["meta"]["time_step_seconds"] = 3600
        cfg["meta"]["parallelism"]["workers"] = 3
        cfg["subscribers"]["total_count"] = 1
        cfg["anomalies"]["enabled"] = False
        cfg["special_events"] = []

        out_dir = tmp_path / "output"
        out_dir.mkdir(exist_ok=True)
        cfg["meta"]["output"]["path"] = str(out_dir)

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml.dump(cfg, default_flow_style=False))
        config = load_config(config_path)

        stats = orchestrate(config, workers=3, dry_run=False)
        assert stats.files_written > 0


# ---------------------------------------------------------------------------
# Tests: orchestrate matches run_generation output
# ---------------------------------------------------------------------------


@skip_if_not_implemented
class TestOrchestrateMatchesRunner:
    """orchestrate(workers=1) should produce output equivalent to run_generation."""

    def test_single_worker_matches_runner_record_count(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Record counts from orchestrate(workers=1) should match run_generation."""
        from cdr_generator.config.loader import load_config
        from cdr_generator.engine.runner import run_generation

        # Run with run_generation
        dir1 = tmp_path / "runner"
        dir1.mkdir()
        config_path_1 = _make_multiprocess_config(dir1, workers=1)
        config_1 = load_config(config_path_1)
        stats_runner = run_generation(config_1, dry_run=False)

        # Run with orchestrate(workers=1)
        dir2 = tmp_path / "orchestrator"
        dir2.mkdir()
        config_path_2 = _make_multiprocess_config(dir2, workers=1)
        config_2 = load_config(config_path_2)
        stats_orch = orchestrate(config_2, workers=1, dry_run=False)

        # File counts should match
        assert stats_runner.files_written == stats_orch.files_written, (
            f"File count mismatch: runner={stats_runner.files_written}, "
            f"orchestrator={stats_orch.files_written}"
        )

        # Total record counts should match
        assert stats_runner.total_records == stats_orch.total_records, (
            f"Total record count mismatch: runner={stats_runner.total_records}, "
            f"orchestrator={stats_orch.total_records}"
        )

    def test_single_worker_matches_runner_file_content(
        self, tmp_path: pathlib.Path
    ) -> None:
        """File content from orchestrate(workers=1) should match run_generation."""
        from cdr_generator.config.loader import load_config
        from cdr_generator.engine.runner import run_generation

        # Run with run_generation
        dir1 = tmp_path / "runner"
        dir1.mkdir()
        config_path_1 = _make_multiprocess_config(dir1, workers=1)
        config_1 = load_config(config_path_1)
        run_generation(config_1, dry_run=False)

        # Run with orchestrate(workers=1)
        dir2 = tmp_path / "orchestrator"
        dir2.mkdir()
        config_path_2 = _make_multiprocess_config(dir2, workers=1)
        config_2 = load_config(config_path_2)
        orchestrate(config_2, workers=1, dry_run=False)

        output_dir_1 = pathlib.Path(config_1.meta.output.path)
        output_dir_2 = pathlib.Path(config_2.meta.output.path)

        records_runner = _collect_records_by_file(output_dir_1)
        records_orch = _collect_records_by_file(output_dir_2)

        assert set(records_runner.keys()) == set(records_orch.keys()), (
            "Output file names must match between runner and orchestrator"
        )

        for fname in sorted(records_runner.keys()):
            recs_runner = records_runner[fname]
            recs_orch = records_orch[fname]
            assert len(recs_runner) == len(recs_orch), (
                f"Record count mismatch in {fname}: "
                f"runner={len(recs_runner)}, orchestrator={len(recs_orch)}"
            )
            for i, (r1, r2) in enumerate(zip(recs_runner, recs_orch)):
                assert r1 == r2, (
                    f"Content mismatch in {fname} at row {i}:\n"
                    f"  runner:       {r1}\n"
                    f"  orchestrator: {r2}"
                )
