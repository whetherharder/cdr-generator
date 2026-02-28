"""Tests for the CDR generation runner (engine/runner.py) and CLI integration.

Phase 2 acceptance criteria:
- CDR files with real records, sorted by event_timestamp
- Record count ~ estimate (+/-10%)
- SGW+PGW paired (charging_id), MO+MT paired (consolidation_id)
"""

from __future__ import annotations

import copy
import csv
import gzip
import pathlib
from collections import Counter
from datetime import datetime, timezone

import pytest
import yaml

from cdr_generator.config.loader import load_config
from cdr_generator.models.cdr import CDR_FIELDS


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_CONFIG_PATH = REPO_ROOT / "cdr_generator_config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_config(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal config for fast test runs: 1 day, 50 subscribers."""
    with open(SAMPLE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Shrink to 1 day and 50 subscribers for speed
    cfg["meta"]["time_range"]["start"] = "2025-01-15T00:00:00Z"
    cfg["meta"]["time_range"]["end"] = "2025-01-15T23:59:59Z"
    cfg["meta"]["parallelism"]["workers"] = 1
    cfg["subscribers"]["total_count"] = 50

    # Disable anomalies for clean testing
    cfg["anomalies"]["enabled"] = False
    cfg["special_events"] = []

    out_dir = tmp_path / "output"
    out_dir.mkdir(exist_ok=True)
    cfg["meta"]["output"]["path"] = str(out_dir)

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False))
    return config_path


def _read_gz_csv(path: pathlib.Path) -> tuple[list[str], list[list[str]]]:
    """Read a gzip CSV and return (header, rows). Skips metadata comment lines."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(lines)
    header = next(reader)
    rows = list(reader)
    return header, rows


def _collect_all_records(output_dir: pathlib.Path) -> list[dict[str, str]]:
    """Collect all CDR rows from all .csv.gz files in the output directory."""
    all_rows = []
    for gz_file in sorted(output_dir.rglob("*.csv.gz")):
        header, rows = _read_gz_csv(gz_file)
        for row in rows:
            all_rows.append(dict(zip(header, row)))
    return all_rows


# ---------------------------------------------------------------------------
# Tests: dry run
# ---------------------------------------------------------------------------


class TestDryRunCreatesFiles:
    """--dry-run must create output files (header-only, no data records)."""

    def test_dry_run_creates_files(self, tmp_path: pathlib.Path) -> None:
        """Output directory must contain .csv.gz files after dry run."""
        from cdr_generator.engine.runner import run_generation

        config_path = _make_small_config(tmp_path)
        config = load_config(config_path)

        run_generation(config, dry_run=True)

        output_dir = pathlib.Path(config.meta.output.path)
        gz_files = list(output_dir.rglob("*.csv.gz"))
        assert len(gz_files) > 0, "Dry run must create at least one .csv.gz file"


class TestDryRunHeaderOnly:
    """--dry-run files must contain only the header, zero data records."""

    def test_no_data_records(self, tmp_path: pathlib.Path) -> None:
        from cdr_generator.engine.runner import run_generation

        config_path = _make_small_config(tmp_path)
        config = load_config(config_path)

        run_generation(config, dry_run=True)

        output_dir = pathlib.Path(config.meta.output.path)
        for gz_file in output_dir.rglob("*.csv.gz"):
            header, rows = _read_gz_csv(gz_file)
            assert header == CDR_FIELDS, f"Header mismatch in {gz_file.name}"
            assert len(rows) == 0, (
                f"Dry run must produce 0 records, got {len(rows)} in {gz_file.name}"
            )


# ---------------------------------------------------------------------------
# Tests: actual generation
# ---------------------------------------------------------------------------


class TestGenerateProducesRecords:
    """Normal run (not dry-run) must produce > 0 CDR records."""

    def test_records_produced(self, tmp_path: pathlib.Path) -> None:
        from cdr_generator.engine.runner import run_generation

        config_path = _make_small_config(tmp_path)
        config = load_config(config_path)

        run_generation(config, dry_run=False)

        output_dir = pathlib.Path(config.meta.output.path)
        all_records = _collect_all_records(output_dir)
        assert len(all_records) > 0, (
            "Normal generation must produce at least one CDR record"
        )


class TestOutputSorted:
    """Records within each file must be sorted by event_timestamp."""

    def test_event_timestamps_sorted(self, tmp_path: pathlib.Path) -> None:
        from cdr_generator.engine.runner import run_generation

        config_path = _make_small_config(tmp_path)
        config = load_config(config_path)

        run_generation(config, dry_run=False)

        output_dir = pathlib.Path(config.meta.output.path)
        for gz_file in sorted(output_dir.rglob("*.csv.gz")):
            header, rows = _read_gz_csv(gz_file)
            if not rows:
                continue  # skip empty files

            ts_idx = header.index("event_timestamp")
            timestamps = [row[ts_idx] for row in rows]

            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i - 1], (
                    f"Records not sorted in {gz_file.name}: "
                    f"'{timestamps[i - 1]}' > '{timestamps[i]}' at row {i}"
                )


class TestSgwPgwPairsComplete:
    """Every charging_id must appear in both SGW and PGW records."""

    def test_charging_id_pairs(self, tmp_path: pathlib.Path) -> None:
        from cdr_generator.engine.runner import run_generation

        config_path = _make_small_config(tmp_path)
        config = load_config(config_path)

        run_generation(config, dry_run=False)

        output_dir = pathlib.Path(config.meta.output.path)
        all_records = _collect_all_records(output_dir)

        # Collect all data records with charging_id
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

        # Every charging_id that appears in SGW must also appear in PGW
        # and with the same count (or at least present)
        assert set(sgw_ids.keys()) == set(pgw_ids.keys()), (
            f"SGW and PGW charging_ids must match.\n"
            f"  SGW-only: {set(sgw_ids.keys()) - set(pgw_ids.keys())}\n"
            f"  PGW-only: {set(pgw_ids.keys()) - set(sgw_ids.keys())}"
        )

        # Each charging_id should have equal counts in SGW and PGW
        for cid in sgw_ids:
            assert sgw_ids[cid] == pgw_ids[cid], (
                f"charging_id={cid}: SGW count={sgw_ids[cid]} != PGW count={pgw_ids[cid]}"
            )


class TestMoMtPairsComplete:
    """Every consolidation_id in voice records should appear exactly twice
    (once MO, once MT) for successful calls."""

    def test_consolidation_id_pairs(self, tmp_path: pathlib.Path) -> None:
        from cdr_generator.engine.runner import run_generation

        config_path = _make_small_config(tmp_path)
        config = load_config(config_path)
        # Set high success rate for cleaner pairing test
        config.events.voice.success_rate = 1.0

        run_generation(config, dry_run=False)

        output_dir = pathlib.Path(config.meta.output.path)
        all_records = _collect_all_records(output_dir)

        voice_records = [
            r for r in all_records if r.get("record_type", "") in ("mo_call", "mt_call")
        ]
        if not voice_records:
            pytest.skip("No voice records generated")

        # Group by consolidation_id
        by_cid: dict[str, list[str]] = {}
        for rec in voice_records:
            cid = rec.get("consolidation_id", "")
            if not cid:
                continue
            by_cid.setdefault(cid, []).append(rec["record_type"])

        # Each consolidation_id should have exactly 1 MO + 1 MT
        for cid, types in by_cid.items():
            assert "mo_call" in types, f"consolidation_id={cid} missing MO record"
            assert "mt_call" in types, f"consolidation_id={cid} missing MT record"
            assert len(types) == 2, (
                f"consolidation_id={cid} should have 2 records, got {len(types)}: {types}"
            )
