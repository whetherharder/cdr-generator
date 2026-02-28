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
from datetime import date, datetime

import pytest

from cdr_generator.models.cdr import CDR_FIELDS, CDRRecord, to_csv_row
from cdr_generator.writer.csv_writer import CSVWriter, CsvWriter, create_empty_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides) -> CDRRecord:
    """Create a minimal valid CDRRecord with optional overrides."""
    defaults = {
        "record_type": "mo_call",
        "served_imsi": "250010000000001",
        "event_timestamp": datetime(2025, 1, 1, 12, 0, 0),
        "serving_ne_id": "msc-01",
    }
    defaults.update(overrides)
    return CDRRecord(**defaults)


def _read_gz_lines(path: pathlib.Path) -> list[str]:
    """Read all lines from a gzip file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.readlines()


def _read_gz_csv_rows(path: pathlib.Path) -> list[list[str]]:
    """Read CSV rows from gzip, skipping comment lines."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        lines = f.readlines()
    data_lines = [l for l in lines if not l.startswith("#")]
    reader = csv.reader(data_lines)
    return list(reader)


# ===========================================================================
# CDR Model Tests
# ===========================================================================


class TestCDRFields:
    """Test CDR_FIELDS definition and CDRRecord dataclass."""

    def test_cdr_fields_is_list_of_strings(self) -> None:
        assert isinstance(CDR_FIELDS, list)
        assert all(isinstance(f, str) for f in CDR_FIELDS)

    def test_cdr_fields_contains_mandatory_fields(self) -> None:
        for field in ("record_type", "served_imsi", "event_timestamp", "serving_ne_id"):
            assert field in CDR_FIELDS, (
                f"Mandatory field {field!r} missing from CDR_FIELDS"
            )

    def test_cdr_fields_count(self) -> None:
        assert len(CDR_FIELDS) == 26

    def test_cdr_fields_no_duplicates(self) -> None:
        assert len(CDR_FIELDS) == len(set(CDR_FIELDS)), "CDR_FIELDS contains duplicates"


class TestCDRRecord:
    """Test CDRRecord dataclass creation and defaults."""

    def test_create_minimal_record(self) -> None:
        record = _make_record()
        assert record.record_type == "mo_call"
        assert record.served_imsi == "250010000000001"
        assert record.serving_ne_id == "msc-01"

    def test_optional_fields_default_to_none(self) -> None:
        record = _make_record()
        assert record.served_msisdn is None
        assert record.duration_seconds is None
        assert record.first_cell_id is None
        assert record.apn is None
        assert record.vendor_extensions is None

    def test_create_full_record(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0)
        record = CDRRecord(
            record_type="mo_call",
            served_imsi="250010000000001",
            event_timestamp=ts,
            serving_ne_id="msc-01",
            served_msisdn="+79160000001",
            duration_seconds=120.5,
            first_cell_id=20011,
            last_cell_id=20012,
            cause_for_termination=16,
        )
        assert record.served_msisdn == "+79160000001"
        assert record.duration_seconds == 120.5
        assert record.first_cell_id == 20011

    def test_missing_mandatory_field_raises(self) -> None:
        with pytest.raises(TypeError):
            CDRRecord(record_type="mo_call")  # missing other mandatory fields


class TestToCsvRow:
    """Test to_csv_row serialization."""

    def test_row_length_matches_fields(self) -> None:
        record = _make_record()
        row = to_csv_row(record)
        assert len(row) == len(CDR_FIELDS)

    def test_row_values_are_strings(self) -> None:
        record = _make_record()
        row = to_csv_row(record)
        assert all(isinstance(v, str) for v in row)

    def test_none_fields_become_empty_string(self) -> None:
        record = _make_record()
        row = to_csv_row(record)
        field_map = dict(zip(CDR_FIELDS, row))
        assert field_map["served_msisdn"] == ""
        assert field_map["duration_seconds"] == ""
        assert field_map["apn"] == ""

    def test_datetime_format(self) -> None:
        ts = datetime(2025, 1, 15, 8, 30, 45, 123000)
        record = _make_record(event_timestamp=ts)
        row = to_csv_row(record)
        field_map = dict(zip(CDR_FIELDS, row))
        assert field_map["event_timestamp"] == "2025-01-15T08:30:45.123Z"

    def test_float_format_no_trailing_zeros(self) -> None:
        record = _make_record(duration_seconds=120.0)
        row = to_csv_row(record)
        field_map = dict(zip(CDR_FIELDS, row))
        assert field_map["duration_seconds"] == "120"

    def test_float_format_preserves_precision(self) -> None:
        record = _make_record(duration_seconds=120.567)
        row = to_csv_row(record)
        field_map = dict(zip(CDR_FIELDS, row))
        assert field_map["duration_seconds"] == "120.567"

    def test_integer_fields(self) -> None:
        record = _make_record(first_cell_id=20011, cause_for_termination=16)
        row = to_csv_row(record)
        field_map = dict(zip(CDR_FIELDS, row))
        assert field_map["first_cell_id"] == "20011"
        assert field_map["cause_for_termination"] == "16"

    def test_field_order_matches_cdr_fields(self) -> None:
        record = _make_record(
            record_type="mo_call",
            served_imsi="250010000000001",
            serving_ne_id="msc-01",
        )
        row = to_csv_row(record)
        assert row[CDR_FIELDS.index("record_type")] == "mo_call"
        assert row[CDR_FIELDS.index("served_imsi")] == "250010000000001"
        assert row[CDR_FIELDS.index("serving_ne_id")] == "msc-01"


# ===========================================================================
# CsvWriter (High-Level) Tests
# ===========================================================================


class TestCsvWriterCreatesFile:
    """Test that CsvWriter creates properly named files."""

    def test_csv_writer_creates_file(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        expected = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        assert expected.exists(), f"Expected file not found: {expected}"

    def test_csv_writer_creates_ne_subdirectory(
        self, tmp_output_dir: pathlib.Path
    ) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="sgw-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        ne_dir = tmp_output_dir / "sgw-01"
        assert ne_dir.is_dir()

    def test_csv_writer_multiple_nes(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        for ne_id in ["msc-01", "msc-02", "sgw-01"]:
            writer.write_file(ne_id=ne_id, file_date=date(2025, 1, 1), records=[])
        writer.close()

        for ne_id in ["msc-01", "msc-02", "sgw-01"]:
            expected = tmp_output_dir / ne_id / f"CDR_{ne_id}_20250101.csv.gz"
            assert expected.exists(), f"File not found for NE {ne_id}"

    def test_csv_writer_multiple_dates(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 2), records=[])
        writer.close()

        for d in ["20250101", "20250102"]:
            expected = tmp_output_dir / "msc-01" / f"CDR_msc-01_{d}.csv.gz"
            assert expected.exists()

    def test_write_file_returns_path(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        result = writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1))
        writer.close()

        expected = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        assert result == expected


class TestCsvWriterContextManager:
    """Test CsvWriter context manager support."""

    def test_context_manager(self, tmp_output_dir: pathlib.Path) -> None:
        with CsvWriter(output_dir=tmp_output_dir) as writer:
            writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1))

        expected = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        assert expected.exists()

    def test_context_manager_returns_self(self, tmp_output_dir: pathlib.Path) -> None:
        writer_instance = CsvWriter(output_dir=tmp_output_dir)
        with writer_instance as w:
            assert w is writer_instance


class TestCsvWriterHeader:
    """Test that the CSV header matches the CDR schema."""

    def test_csv_writer_header_matches_schema(
        self, tmp_output_dir: pathlib.Path
    ) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        rows = _read_gz_csv_rows(file_path)
        assert len(rows) >= 1
        assert rows[0] == CDR_FIELDS

    def test_csv_writer_is_valid_gzip(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        with gzip.open(file_path, "rt") as f:
            content = f.read()
        assert isinstance(content, str)


class TestCsvWriterMetadata:
    """Test metadata comment writing."""

    def test_metadata_comment_present_by_default(
        self, tmp_output_dir: pathlib.Path
    ) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir, include_metadata=True)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1))

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        lines = _read_gz_lines(file_path)
        assert lines[0].startswith("#")
        assert "ne_id=msc-01" in lines[0]

    def test_metadata_comment_disabled(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir, include_metadata=False)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1))

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        lines = _read_gz_lines(file_path)
        assert not lines[0].startswith("#"), (
            "No comment expected when metadata disabled"
        )

    def test_metadata_contains_date(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir, include_metadata=True)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 3, 15))

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250315.csv.gz"
        lines = _read_gz_lines(file_path)
        assert "date=20250315" in lines[0]


class TestCsvWriterRecords:
    """Test writing actual CDR records."""

    def test_write_single_record(self, tmp_output_dir: pathlib.Path) -> None:
        record = _make_record()
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[record])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        rows = _read_gz_csv_rows(file_path)
        assert len(rows) == 2  # header + 1 data row

    def test_write_multiple_records(self, tmp_output_dir: pathlib.Path) -> None:
        records = [
            _make_record(event_timestamp=datetime(2025, 1, 1, h, 0, 0))
            for h in range(5)
        ]
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=records)
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        rows = _read_gz_csv_rows(file_path)
        assert len(rows) == 6  # header + 5 data rows

    def test_record_data_integrity(self, tmp_output_dir: pathlib.Path) -> None:
        record = _make_record(
            record_type="mo_call",
            served_imsi="250010000000099",
            served_msisdn="+79160000099",
            duration_seconds=180.5,
        )
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[record])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        rows = _read_gz_csv_rows(file_path)
        data_row = rows[1]
        field_map = dict(zip(CDR_FIELDS, data_row))
        assert field_map["record_type"] == "mo_call"
        assert field_map["served_imsi"] == "250010000000099"
        assert field_map["served_msisdn"] == "+79160000099"
        assert field_map["duration_seconds"] == "180.5"

    def test_none_records_treated_as_empty(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=None)
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        rows = _read_gz_csv_rows(file_path)
        assert len(rows) == 1  # header only


class TestCsvWriterDelimiter:
    """Test CSV delimiter configuration."""

    def test_default_comma_delimiter(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        record = _make_record()
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[record])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        lines = _read_gz_lines(file_path)
        # Find the header line (non-comment)
        for line in lines:
            if not line.startswith("#"):
                assert "," in line
                break

    def test_pipe_delimiter(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir, delimiter="|")
        record = _make_record()
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[record])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        lines = _read_gz_lines(file_path)
        header_line = next(l for l in lines if not l.startswith("#"))
        assert "|" in header_line
        fields = header_line.strip().split("|")
        assert len(fields) == len(CDR_FIELDS)


# ===========================================================================
# CSVWriter (Low-Level) Tests
# ===========================================================================


class TestCSVWriterLowLevel:
    """Test the lower-level CSVWriter class used by create_empty_output."""

    def test_write_header_creates_file(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir)
        path = writer.write_header("msc-01", "20250101", metadata={"ne_id": "msc-01"})
        assert path.exists()

    def test_write_header_returns_correct_path(
        self, tmp_output_dir: pathlib.Path
    ) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir)
        path = writer.write_header("msc-01", "20250101")
        expected = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        assert path == expected

    def test_write_header_with_metadata(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir)
        path = writer.write_header(
            "msc-01", "20250101", metadata={"ne_id": "msc-01", "ne_type": "msc"}
        )
        lines = _read_gz_lines(path)
        assert lines[0].startswith("# ")
        assert "ne_id=msc-01" in lines[0]
        assert "ne_type=msc" in lines[0]

    def test_write_header_without_metadata(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir)
        path = writer.write_header("msc-01", "20250101")
        lines = _read_gz_lines(path)
        assert not lines[0].startswith("#")

    def test_write_header_metadata_disabled(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir, include_metadata=False)
        path = writer.write_header("msc-01", "20250101", metadata={"ne_id": "msc-01"})
        lines = _read_gz_lines(path)
        assert not lines[0].startswith("#")

    def test_write_header_contains_cdr_fields(
        self, tmp_output_dir: pathlib.Path
    ) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir)
        path = writer.write_header("msc-01", "20250101")
        rows = _read_gz_csv_rows(path)
        assert rows[0] == CDR_FIELDS

    def test_custom_filename_template(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(
            output_dir=tmp_output_dir,
            filename_template="CDR-{ne_id}-{date}.csv.gz",
        )
        path = writer.write_header("msc-01", "20250115")
        expected = tmp_output_dir / "msc-01" / "CDR-msc-01-20250115.csv.gz"
        assert path == expected
        assert path.exists()

    def test_custom_delimiter(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir, delimiter=";")
        path = writer.write_header("msc-01", "20250101")
        lines = _read_gz_lines(path)
        header_line = next(l for l in lines if not l.startswith("#"))
        assert ";" in header_line


class TestCSVWriterRecords:
    """Test CSVWriter.write_records appending."""

    def test_write_records_appends(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir)
        writer.write_header("msc-01", "20250101")
        records = [_make_record(), _make_record(served_imsi="250010000000002")]
        count = writer.write_records("msc-01", "20250101", records)

        assert count == 2
        path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        rows = _read_gz_csv_rows(path)
        assert len(rows) == 3  # header + 2 records

    def test_write_records_returns_count(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir)
        writer.write_header("msc-01", "20250101")
        records = [_make_record() for _ in range(7)]
        count = writer.write_records("msc-01", "20250101", records)
        assert count == 7

    def test_write_records_empty_list(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir)
        writer.write_header("msc-01", "20250101")
        count = writer.write_records("msc-01", "20250101", [])
        assert count == 0

    def test_write_records_data_integrity(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CSVWriter(output_dir=tmp_output_dir, include_metadata=False)
        writer.write_header("msc-01", "20250101")
        record = _make_record(
            record_type="data_session",
            served_imsi="250010000000042",
            apn="internet",
            uplink_volume_bytes=1024,
            downlink_volume_bytes=4096,
        )
        writer.write_records("msc-01", "20250101", [record])

        path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        rows = _read_gz_csv_rows(path)
        data_row = rows[1]
        field_map = dict(zip(CDR_FIELDS, data_row))
        assert field_map["record_type"] == "data_session"
        assert field_map["served_imsi"] == "250010000000042"
        assert field_map["apn"] == "internet"
        assert field_map["uplink_volume_bytes"] == "1024"
        assert field_map["downlink_volume_bytes"] == "4096"


# ===========================================================================
# DryRun / create_empty_output Tests
# ===========================================================================


class TestDryRun:
    """Test dry-run mode produces headers only."""

    def test_empty_output_dry_run(self, tmp_output_dir: pathlib.Path) -> None:
        writer = CsvWriter(output_dir=tmp_output_dir)
        writer.write_file(ne_id="msc-01", file_date=date(2025, 1, 1), records=[])
        writer.close()

        file_path = tmp_output_dir / "msc-01" / "CDR_msc-01_20250101.csv.gz"
        rows = _read_gz_csv_rows(file_path)
        assert len(rows) == 1, "Dry-run should have header only (1 row)"

    def test_dry_run_all_nes_get_files(
        self,
        sample_config_path: pathlib.Path,
        tmp_output_dir: pathlib.Path,
    ) -> None:
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)
        writer = CsvWriter(output_dir=tmp_output_dir)
        ne_ids = [e.id for e in config.network.elements]

        for ne_id in ne_ids:
            writer.write_file(ne_id=ne_id, file_date=date(2025, 1, 1), records=[])
        writer.close()

        for ne_id in ne_ids:
            expected = tmp_output_dir / ne_id / f"CDR_{ne_id}_20250101.csv.gz"
            assert expected.exists(), f"Dry-run file missing for NE {ne_id}"


class TestCreateEmptyOutput:
    """Test create_empty_output function used by CLI dry-run."""

    def test_creates_files_for_all_nes_and_dates(
        self,
        sample_config_path: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        from cdr_generator.assets.generator import generate_all_assets
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)
        config.meta.output.path = str(tmp_path / "output")
        _, nes, _ = generate_all_assets(config)

        paths = create_empty_output(config, nes)
        assert len(paths) > 0
        for p in paths:
            assert p.exists()

    def test_file_count_matches_ne_times_dates(
        self,
        tmp_config_file,
        minimal_valid_config_dict: dict,
        tmp_path: pathlib.Path,
    ) -> None:
        from cdr_generator.assets.generator import generate_all_assets
        from cdr_generator.config.loader import load_config

        output_dir = tmp_path / "output"
        minimal_valid_config_dict["meta"]["output"]["path"] = str(output_dir)
        cfg_path = tmp_config_file(minimal_valid_config_dict)
        config = load_config(cfg_path)
        _, nes, _ = generate_all_assets(config)

        paths = create_empty_output(config, nes)

        start_d = config.meta.time_range.start.date()
        end_d = config.meta.time_range.end.date()
        day_count = (end_d - start_d).days + 1
        expected_count = len(nes) * day_count
        assert len(paths) == expected_count

    def test_files_have_headers_only(
        self,
        tmp_config_file,
        minimal_valid_config_dict: dict,
        tmp_path: pathlib.Path,
    ) -> None:
        from cdr_generator.assets.generator import generate_all_assets
        from cdr_generator.config.loader import load_config

        output_dir = tmp_path / "output"
        minimal_valid_config_dict["meta"]["output"]["path"] = str(output_dir)
        cfg_path = tmp_config_file(minimal_valid_config_dict)
        config = load_config(cfg_path)
        _, nes, _ = generate_all_assets(config)

        paths = create_empty_output(config, nes)
        for p in paths:
            rows = _read_gz_csv_rows(p)
            assert len(rows) == 1, (
                f"File {p} should have header only, got {len(rows)} rows"
            )
            assert rows[0] == CDR_FIELDS

    def test_files_have_metadata_comments(
        self,
        tmp_config_file,
        minimal_valid_config_dict: dict,
        tmp_path: pathlib.Path,
    ) -> None:
        from cdr_generator.assets.generator import generate_all_assets
        from cdr_generator.config.loader import load_config

        output_dir = tmp_path / "output"
        minimal_valid_config_dict["meta"]["output"]["path"] = str(output_dir)
        cfg_path = tmp_config_file(minimal_valid_config_dict)
        config = load_config(cfg_path)
        _, nes, _ = generate_all_assets(config)

        paths = create_empty_output(config, nes)
        for p in paths:
            lines = _read_gz_lines(p)
            assert lines[0].startswith("# "), f"File {p} missing metadata comment"
            assert "ne_id=" in lines[0]
            assert "ne_type=" in lines[0]


class TestDateRange:
    """Test the internal _date_range helper via create_empty_output."""

    def test_single_day_range(
        self,
        tmp_config_file,
        minimal_valid_config_dict: dict,
        tmp_path: pathlib.Path,
    ) -> None:
        """Config with start/end on same day should produce 1 file per NE."""
        from cdr_generator.assets.generator import generate_all_assets
        from cdr_generator.config.loader import load_config

        output_dir = tmp_path / "output"
        minimal_valid_config_dict["meta"]["output"]["path"] = str(output_dir)
        cfg_path = tmp_config_file(minimal_valid_config_dict)
        config = load_config(cfg_path)
        _, nes, _ = generate_all_assets(config)

        paths = create_empty_output(config, nes)
        assert len(paths) == len(nes)

    def test_multi_day_range(
        self,
        tmp_config_file,
        minimal_valid_config_dict: dict,
        tmp_path: pathlib.Path,
    ) -> None:
        """Config spanning 3 days should produce 3 files per NE."""
        from cdr_generator.assets.generator import generate_all_assets
        from cdr_generator.config.loader import load_config

        output_dir = tmp_path / "output"
        minimal_valid_config_dict["meta"]["time_range"]["start"] = (
            "2025-01-01T00:00:00Z"
        )
        minimal_valid_config_dict["meta"]["time_range"]["end"] = "2025-01-03T23:59:59Z"
        minimal_valid_config_dict["meta"]["output"]["path"] = str(output_dir)
        cfg_path = tmp_config_file(minimal_valid_config_dict)
        config = load_config(cfg_path)
        _, nes, _ = generate_all_assets(config)

        paths = create_empty_output(config, nes)
        assert len(paths) == len(nes) * 3
