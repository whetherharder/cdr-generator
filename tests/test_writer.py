"""Tests for CSV+gzip writer.

Acceptance criteria from plan.md Phase 1:
- Empty CSV+gzip files with correct names and headers
- One file per (NE, date) combination
- Filename format: CDR_{ne_id}_{YYYYMMDD}.csv.gz
- dry-run produces files with headers only, 0 data records
"""

import csv
import gzip
import pathlib
from datetime import date

import pytest


class TestCsvWriterCreatesFile:
    """Test that the writer creates properly named files."""

    def test_csv_writer_creates_file(self, tmp_output_dir: pathlib.Path) -> None:
        """Writer must create a CDR_{ne_id}_{date}.csv.gz file."""
        from cdr_generator.writer.csv_writer import CsvWriter

        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        expected = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        assert expected.exists(), f"Expected file not found: {expected}"

    def test_csv_writer_creates_ne_subdirectory(self, tmp_output_dir: pathlib.Path) -> None:
        """Writer must create a subdirectory per NE."""
        from cdr_generator.writer.csv_writer import CsvWriter

        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="sgw-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        ne_dir = tmp_output_dir / "sgw-01"
        assert ne_dir.is_dir(), f"NE subdirectory not found: {ne_dir}"

    def test_csv_writer_multiple_nes(self, tmp_output_dir: pathlib.Path) -> None:
        """Writer must handle multiple NEs creating separate directories."""
        from cdr_generator.writer.csv_writer import CsvWriter

        writer = CsvWriter(output_dir=tmp_output_dir)
        for ne_id in ["msc-01", "msc-02", "sgw-01"]:
            writer.write_file(ne_id=ne_id, file_date=date(2025, 1, 1), records=[])
        writer.close()

        for ne_id in ["msc-01", "msc-02", "sgw-01"]:
            expected = tmp_output_dir / ne_id / f"CDR_{ne_id}_20250101.csv.gz"
            assert expected.exists(), f"File not found for NE {ne_id}: {expected}"

    def test_csv_writer_multiple_dates(self, tmp_output_dir: pathlib.Path) -> None:
        """Writer must create separate files per date."""
        from cdr_generator.writer.csv_writer import CsvWriter

        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 2), records=[])
        writer.close()

        for d in ["20250101", "20250102"]:
            expected = tmp_output_dir / "msc-01" / f"CDR_msc-01_{d}.csv.gz"
            assert expected.exists(), f"File not found for date {d}: {expected}"


class TestCsvWriterHeader:
    """Test that the CSV header matches the CDR schema."""

    def test_csv_writer_header_matches_schema(self, tmp_output_dir: pathlib.Path) -> None:
        """CSV header row must contain the expected CDR field names."""
        from cdr_generator.writer.csv_writer import CsvWriter

        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        with gzip.open(file_path, "rt") as f:
            # Skip metadata comment line if present
            lines = f.readlines()

        # Find the header line (first non-comment line)
        header_line = None
        for line in lines:
            if not line.startswith("#"):
                header_line = line.strip()
                break

        assert header_line is not None, "No header line found in CSV"
        fields = header_line.split(",")
        assert len(fields) > 0, "Header must have at least one field"

        # Verify essential CDR fields are present
        essential_fields = {
            "record_type",
            "event_timestamp",
            "served_imsi",
            "served_msisdn",
        }
        header_set = set(fields)
        for field in essential_fields:
            assert field in header_set, (
                f"Essential field '{field}' missing from header. "
                f"Got: {fields}"
            )

    def test_csv_writer_is_valid_gzip(self, tmp_output_dir: pathlib.Path) -> None:
        """Output file must be valid gzip."""
        from cdr_generator.writer.csv_writer import CsvWriter

        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        # Should decompress without error
        with gzip.open(file_path, "rt") as f:
            content = f.read()
        assert isinstance(content, str)


class TestDryRun:
    """Test dry-run mode produces headers only."""

    def test_empty_output_dry_run(self, tmp_output_dir: pathlib.Path) -> None:
        """Dry-run must produce files with header only and zero data records."""
        from cdr_generator.writer.csv_writer import CsvWriter

        writer = CsvWriter(output_dir=tmp_output_dir)
        # Write with empty records list simulates dry-run output
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        with gzip.open(file_path, "rt") as f:
            lines = f.readlines()

        # Filter out comment lines
        data_lines = [line for line in lines if not line.startswith("#")]

        # Should have exactly 1 line (header) and 0 data rows
        assert len(data_lines) == 1, (
            f"Dry-run should have header only (1 line), got {len(data_lines)} lines"
        )

    def test_dry_run_all_nes_get_files(
        self,
        sample_config_path: pathlib.Path,
        tmp_assets_dir: pathlib.Path,
        tmp_output_dir: pathlib.Path,
    ) -> None:
        """Dry-run for full config should create one file per (NE, date)."""
        from cdr_generator.config.loader import load_config
        from cdr_generator.assets.generator import generate_assets
        from cdr_generator.writer.csv_writer import CsvWriter

        config = load_config(sample_config_path)

        # In dry-run, each NE that produces CDRs should get a file
        writer = CsvWriter(output_dir=tmp_output_dir)
        ne_ids = [e.id for e in config.network.elements]

        for ne_id in ne_ids:
            writer.write_file(ne_id=ne_id, file_date=date(2025, 1, 1), records=[])
        writer.close()

        for ne_id in ne_ids:
            expected = tmp_output_dir / ne_id / f"CDR_{ne_id}_20250101.csv.gz"
            assert expected.exists(), f"Dry-run file missing for NE {ne_id}"


class TestCsvWriterDelimiter:
    """Test CSV delimiter configuration."""

    def test_default_comma_delimiter(self, tmp_output_dir: pathlib.Path) -> None:
        """Default delimiter should be comma."""
        from cdr_generator.writer.csv_writer import CsvWriter

        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        with gzip.open(file_path, "rt") as f:
            lines = f.readlines()

        # Find header (non-comment line)
        for line in lines:
            if not line.startswith("#"):
                # Header should use comma as delimiter
                assert "," in line, "Default delimiter should be comma"
                break
