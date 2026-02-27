"""Phase 3 integration tests -- end-to-end generation with contact book,
mobility, B-party selection, and vendor extensions validation.

Validates that all Phase 3 features work together when run through the
``run_generation`` pipeline with a realistic (but small) config.

Acceptance criteria checked:
- Contact book / B-party distribution: ~15% external calls
- Mobility: cell IDs populated, handovers detected (first != last cell)
- Vendor extensions: valid base64-encoded JSON with vendor.* keys
- Record integrity: all record types generated, timestamps in range
"""

from __future__ import annotations

import base64
import csv
import gzip
import json
import pathlib
from collections import Counter
from datetime import datetime, timezone

import pytest
import yaml

from cdr_generator.config.loader import load_config
from cdr_generator.engine.runner import run_generation

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_CONFIG_PATH = REPO_ROOT / "cdr_generator_config.yaml"

# External number prefixes from the reference config
EXTERNAL_PREFIXES = ("+7495", "+7926", "+7985", "+4420", "+1212", "+7812")
# Internal subscriber MSISDN prefix from the reference config
INTERNAL_PREFIX = "+7916"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase3_config(
    tmp_path: pathlib.Path,
    start_hour: int = 10,
    end_hour: int = 11,
) -> pathlib.Path:
    """Create a config suitable for Phase 3 integration testing.

    Returns the path to the written YAML config file.
    Uses 100 subscribers, a narrow time window, anomalies disabled,
    and no special events, for fast deterministic execution.
    """
    with open(SAMPLE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # 100 subscribers, 1-hour window on a Wednesday (2025-01-15)
    cfg["meta"]["time_range"]["start"] = f"2025-01-15T{start_hour:02d}:00:00Z"
    cfg["meta"]["time_range"]["end"] = f"2025-01-15T{end_hour:02d}:00:00Z"
    cfg["meta"]["parallelism"]["workers"] = 1
    cfg["subscribers"]["total_count"] = 100

    # Ensure SMSC has serves_tacs so SMS records are actually generated
    for elem in cfg["network"]["elements"]:
        if elem["type"] == "smsc":
            elem["serves_tacs"] = [2001, 2002, 2003, 2004]

    # Disable anomalies and special events for clean testing
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
    all_rows: list[dict[str, str]] = []
    for gz_file in sorted(output_dir.rglob("*.csv.gz")):
        header, rows = _read_gz_csv(gz_file)
        for row in rows:
            all_rows.append(dict(zip(header, row)))
    return all_rows


def _is_external_number(msisdn: str) -> bool:
    """Check if a MSISDN belongs to the external number pool."""
    return any(msisdn.startswith(p) for p in EXTERNAL_PREFIXES)


def _decode_vendor_extensions(encoded: str) -> dict:
    """Decode a base64-encoded JSON vendor extensions field."""
    raw = base64.b64decode(encoded)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Shared fixture: run generation once, reuse across tests in the module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def phase3_output(tmp_path_factory):
    """Run a single Phase 3 generation and return (config, records, output_dir)."""
    tmp_path = tmp_path_factory.mktemp("phase3")
    config_path = _make_phase3_config(tmp_path, start_hour=10, end_hour=11)
    config = load_config(config_path)
    stats = run_generation(config, dry_run=False)
    output_dir = pathlib.Path(config.meta.output.path)
    records = _collect_all_records(output_dir)
    return config, records, output_dir, stats


# ---------------------------------------------------------------------------
# Test: End-to-end generation
# ---------------------------------------------------------------------------

class TestPhase3EndToEnd:
    """Verify that Phase 3 generation produces records of all expected types."""

    def test_records_produced(self, phase3_output):
        """Generation must produce a non-trivial number of CDR records."""
        _, records, _, stats = phase3_output
        assert stats.total_records > 0, "Generation produced zero records"
        assert len(records) > 0, "No records collected from output files"

    def test_voice_records_produced(self, phase3_output):
        """At least some voice records (mo_call or mt_call) must be generated."""
        _, records, _, stats = phase3_output
        assert stats.voice_records > 0, "No voice records generated"
        voice_types = {"mo_call", "mt_call"}
        voice_records = [r for r in records if r["record_type"] in voice_types]
        assert len(voice_records) > 0

    def test_data_records_produced(self, phase3_output):
        """At least some data records (sgw_data or pgw_data) must be generated."""
        _, records, _, stats = phase3_output
        assert stats.data_records > 0, "No data records generated"
        data_types = {"sgw_data", "pgw_data"}
        data_records = [r for r in records if r["record_type"] in data_types]
        assert len(data_records) > 0

    def test_sms_records_produced(self, phase3_output):
        """At least some SMS records (mo_sms or mt_sms) must be generated."""
        _, records, _, stats = phase3_output
        assert stats.sms_records > 0, "No SMS records generated"
        sms_types = {"mo_sms", "mt_sms"}
        sms_records = [r for r in records if r["record_type"] in sms_types]
        assert len(sms_records) > 0

    def test_stats_match_record_count(self, phase3_output):
        """GenerationStats.total_records must match the actual CDR count."""
        _, records, _, stats = phase3_output
        assert stats.total_records == len(records), (
            f"Stats report {stats.total_records} records but "
            f"{len(records)} found in output files"
        )

    def test_files_written(self, phase3_output):
        """Output directory must contain .csv.gz files."""
        _, _, output_dir, stats = phase3_output
        gz_files = list(output_dir.rglob("*.csv.gz"))
        assert len(gz_files) > 0, "No .csv.gz files in output directory"
        assert stats.files_written > 0


# ---------------------------------------------------------------------------
# Test: B-party / external call distribution
# ---------------------------------------------------------------------------

class TestPhase3BPartyDistribution:
    """Verify B-party selection produces a realistic mix of internal/external."""

    def test_external_call_ratio(self, phase3_output):
        """Voice MO called_numbers should include ~15% external numbers.

        The config sets external_call_ratio=0.15. With statistical noise,
        we accept a range of 3% to 35%.
        """
        _, records, _, _ = phase3_output
        mo_calls = [
            r for r in records
            if r["record_type"] == "mo_call" and r.get("called_number", "")
        ]

        if len(mo_calls) < 20:
            pytest.skip("Too few MO calls for statistical validation")

        external_count = sum(
            1 for r in mo_calls if _is_external_number(r["called_number"])
        )
        ratio = external_count / len(mo_calls)

        assert 0.03 <= ratio <= 0.35, (
            f"External call ratio {ratio:.3f} outside expected range [0.03, 0.35]. "
            f"{external_count}/{len(mo_calls)} calls to external numbers."
        )

    def test_internal_calls_use_subscriber_prefix(self, phase3_output):
        """Internal B-party numbers should use the subscriber MSISDN prefix."""
        _, records, _, _ = phase3_output
        mo_calls = [
            r for r in records
            if r["record_type"] == "mo_call" and r.get("called_number", "")
        ]

        internal_calls = [
            r for r in mo_calls
            if not _is_external_number(r["called_number"])
        ]

        if not internal_calls:
            pytest.skip("No internal calls found")

        for rec in internal_calls:
            called = rec["called_number"]
            assert called.startswith(INTERNAL_PREFIX) or called.startswith("external"), (
                f"Internal call to {called} does not use subscriber prefix {INTERNAL_PREFIX}"
            )


# ---------------------------------------------------------------------------
# Test: Mobility -- cell assignment and handover
# ---------------------------------------------------------------------------

class TestPhase3Mobility:
    """Verify mobility model: cell IDs are populated and handovers occur."""

    def test_cell_ids_populated(self, phase3_output):
        """All MO records should have first_cell_id and last_cell_id set."""
        _, records, _, _ = phase3_output
        mo_records = [
            r for r in records
            if r["record_type"] in ("mo_call", "mo_sms", "sgw_data")
        ]

        if not mo_records:
            pytest.skip("No MO-side records found")

        missing_first = sum(1 for r in mo_records if not r.get("first_cell_id"))
        missing_last = sum(1 for r in mo_records if not r.get("last_cell_id"))

        assert missing_first == 0, (
            f"{missing_first}/{len(mo_records)} MO records missing first_cell_id"
        )
        assert missing_last == 0, (
            f"{missing_last}/{len(mo_records)} MO records missing last_cell_id"
        )

    def test_valid_cell_ids(self, phase3_output):
        """Cell IDs in CDR records should correspond to cells defined in the config."""
        config, records, _, _ = phase3_output

        # Collect valid cell IDs from config
        valid_cell_ids = set()
        for cell_item in config.network.cells.items:
            valid_cell_ids.add(str(cell_item.cell_id))

        mo_records = [
            r for r in records
            if r["record_type"] in ("mo_call", "mo_sms", "sgw_data")
            and r.get("first_cell_id")
        ]

        if not mo_records:
            pytest.skip("No MO-side records with cell IDs")

        invalid_cells = []
        for rec in mo_records:
            first = rec["first_cell_id"]
            last = rec["last_cell_id"]
            if first and first not in valid_cell_ids:
                invalid_cells.append(("first", first, rec["record_type"]))
            if last and last not in valid_cell_ids:
                invalid_cells.append(("last", last, rec["record_type"]))

        assert len(invalid_cells) == 0, (
            f"Found {len(invalid_cells)} records with invalid cell IDs: "
            f"{invalid_cells[:5]}"
        )

    def test_handover_detected(self, phase3_output):
        """Some records should show handover (first_cell_id != last_cell_id).

        Config profiles have handover_during_call between 0.01 and 0.20,
        so over enough records we should see at least one handover.
        """
        _, records, _, _ = phase3_output

        # Check MO voice records specifically (most likely to show handover)
        mo_records = [
            r for r in records
            if r["record_type"] in ("mo_call", "mo_sms", "sgw_data")
            and r.get("first_cell_id")
            and r.get("last_cell_id")
        ]

        if len(mo_records) < 10:
            pytest.skip("Too few records for handover detection")

        handovers = sum(
            1 for r in mo_records
            if r["first_cell_id"] != r["last_cell_id"]
        )

        assert handovers > 0, (
            f"No handovers detected in {len(mo_records)} MO records. "
            "Expected at least one handover given configured probabilities."
        )


# ---------------------------------------------------------------------------
# Test: Vendor extensions
# ---------------------------------------------------------------------------

class TestPhase3VendorExtensions:
    """Verify vendor extension fields are correctly generated and encoded."""

    def test_ericsson_extensions_present(self, phase3_output):
        """Records from Ericsson NEs (msc-01, sgw-01, smsc-01) should have
        vendor_extensions with ericsson.* keys."""
        _, records, _, _ = phase3_output

        ericsson_nes = {"msc-01", "sgw-01", "smsc-01"}
        ericsson_records = [
            r for r in records
            if r.get("serving_ne_id") in ericsson_nes
            and r.get("vendor_extensions", "")
        ]

        if not ericsson_records:
            pytest.skip("No records from Ericsson NEs with vendor_extensions")

        for rec in ericsson_records[:20]:  # Check a sample
            ext = _decode_vendor_extensions(rec["vendor_extensions"])
            assert isinstance(ext, dict), "Extensions must decode to a dict"
            ericsson_keys = [k for k in ext if k.startswith("ericsson.")]
            assert len(ericsson_keys) > 0, (
                f"Record from {rec['serving_ne_id']} has no ericsson.* keys: {ext}"
            )

    def test_nokia_extensions_present(self, phase3_output):
        """Records from Nokia NE (pgw-01) should have vendor_extensions
        with nokia.* keys."""
        _, records, _, _ = phase3_output

        nokia_records = [
            r for r in records
            if r.get("serving_ne_id") == "pgw-01"
            and r.get("vendor_extensions", "")
        ]

        if not nokia_records:
            pytest.skip("No records from Nokia NE with vendor_extensions")

        for rec in nokia_records[:20]:
            ext = _decode_vendor_extensions(rec["vendor_extensions"])
            assert isinstance(ext, dict)
            nokia_keys = [k for k in ext if k.startswith("nokia.")]
            assert len(nokia_keys) > 0, (
                f"Record from pgw-01 has no nokia.* keys: {ext}"
            )

    def test_huawei_extensions_present(self, phase3_output):
        """Records from Huawei NE (msc-02) should have vendor_extensions
        with huawei.* keys."""
        _, records, _, _ = phase3_output

        huawei_records = [
            r for r in records
            if r.get("serving_ne_id") == "msc-02"
            and r.get("vendor_extensions", "")
        ]

        if not huawei_records:
            pytest.skip("No records from Huawei NE with vendor_extensions")

        for rec in huawei_records[:20]:
            ext = _decode_vendor_extensions(rec["vendor_extensions"])
            assert isinstance(ext, dict)
            huawei_keys = [k for k in ext if k.startswith("huawei.")]
            assert len(huawei_keys) > 0, (
                f"Record from msc-02 has no huawei.* keys: {ext}"
            )

    def test_all_extensions_valid_base64_json(self, phase3_output):
        """Every non-empty vendor_extensions field must be valid base64-encoded JSON."""
        _, records, _, _ = phase3_output

        records_with_ext = [
            r for r in records if r.get("vendor_extensions", "")
        ]

        if not records_with_ext:
            pytest.skip("No records with vendor_extensions")

        errors = []
        for i, rec in enumerate(records_with_ext):
            try:
                decoded = _decode_vendor_extensions(rec["vendor_extensions"])
                assert isinstance(decoded, dict), "Must decode to dict"
            except Exception as e:
                errors.append(
                    f"Record {i} (type={rec['record_type']}, "
                    f"ne={rec['serving_ne_id']}): {e}"
                )

        assert len(errors) == 0, (
            f"{len(errors)} records with invalid vendor_extensions:\n"
            + "\n".join(errors[:10])
        )

    def test_ericsson_from_event_fields(self, phase3_output):
        """Ericsson extensions with 'from_event' should contain cell IDs
        matching the CDR record's cell IDs."""
        _, records, _, _ = phase3_output

        ericsson_records = [
            r for r in records
            if r.get("serving_ne_id") in ("msc-01", "smsc-01")
            and r.get("vendor_extensions", "")
            and r.get("first_cell_id", "")
        ]

        if not ericsson_records:
            pytest.skip("No ericsson records with extensions and cell IDs")

        for rec in ericsson_records[:10]:
            ext = _decode_vendor_extensions(rec["vendor_extensions"])
            # Ericsson config has from_event for first_cell_id and last_cell_id
            if "ericsson.first_cell_id" in ext:
                assert ext["ericsson.first_cell_id"] == rec["first_cell_id"], (
                    f"Extension first_cell_id {ext['ericsson.first_cell_id']} != "
                    f"CDR first_cell_id {rec['first_cell_id']}"
                )
            if "ericsson.last_cell_id" in ext:
                assert ext["ericsson.last_cell_id"] == rec["last_cell_id"], (
                    f"Extension last_cell_id {ext['ericsson.last_cell_id']} != "
                    f"CDR last_cell_id {rec['last_cell_id']}"
                )


# ---------------------------------------------------------------------------
# Test: Record integrity and timestamp validation
# ---------------------------------------------------------------------------

class TestPhase3RecordIntegrity:
    """Verify fundamental record correctness across the generated output."""

    def test_all_records_have_imsi(self, phase3_output):
        """Every CDR record must have a non-empty served_imsi."""
        _, records, _, _ = phase3_output
        missing = [
            r["record_type"] for r in records if not r.get("served_imsi", "")
        ]
        assert len(missing) == 0, (
            f"{len(missing)} records missing served_imsi: types={Counter(missing)}"
        )

    def test_all_records_have_serving_ne(self, phase3_output):
        """Every CDR record must have a non-empty serving_ne_id."""
        _, records, _, _ = phase3_output
        missing = [
            r["record_type"] for r in records if not r.get("serving_ne_id", "")
        ]
        assert len(missing) == 0, (
            f"{len(missing)} records missing serving_ne_id"
        )

    def test_timestamps_in_configured_range(self, phase3_output):
        """All event_timestamps must fall within the configured time range
        (with a small tolerance for MT jitter and SMS delivery delays)."""
        config, records, _, _ = phase3_output

        start = config.meta.time_range.start.replace(tzinfo=timezone.utc)
        end = config.meta.time_range.end.replace(tzinfo=timezone.utc)

        # Allow generous tolerance for SMS delivery delays (up to 24h in config)
        # and MT timestamp jitter
        from datetime import timedelta
        tolerance = timedelta(hours=25)

        early = []
        late = []
        for rec in records:
            ts_str = rec.get("event_timestamp", "")
            if not ts_str:
                continue
            ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
                tzinfo=timezone.utc
            )
            if ts < start - timedelta(minutes=1):
                early.append((rec["record_type"], ts_str))
            if ts > end + tolerance:
                late.append((rec["record_type"], ts_str))

        assert len(early) == 0, (
            f"{len(early)} records before start time: {early[:5]}"
        )
        assert len(late) == 0, (
            f"{len(late)} records after end time + tolerance: {late[:5]}"
        )

    def test_voice_mo_mt_pairing(self, phase3_output):
        """Successful voice calls should produce paired MO+MT records sharing
        a consolidation_id."""
        _, records, _, _ = phase3_output

        voice_records = [
            r for r in records
            if r["record_type"] in ("mo_call", "mt_call")
            and r.get("consolidation_id", "")
        ]

        if len(voice_records) < 4:
            pytest.skip("Too few voice records for pairing test")

        # Group by consolidation_id
        by_cid: dict[str, list[str]] = {}
        for rec in voice_records:
            cid = rec["consolidation_id"]
            by_cid.setdefault(cid, []).append(rec["record_type"])

        # At least some consolidation_ids should have both MO and MT
        paired = sum(
            1 for types in by_cid.values()
            if "mo_call" in types and "mt_call" in types
        )
        assert paired > 0, (
            f"No paired MO+MT voice records found among "
            f"{len(by_cid)} consolidation_ids"
        )

    def test_data_sgw_pgw_pairing(self, phase3_output):
        """Data records should produce paired SGW+PGW records sharing
        a charging_id."""
        _, records, _, _ = phase3_output

        data_records = [
            r for r in records
            if r["record_type"] in ("sgw_data", "pgw_data")
            and r.get("charging_id", "")
        ]

        if len(data_records) < 4:
            pytest.skip("Too few data records for pairing test")

        sgw_ids = {r["charging_id"] for r in data_records if r["record_type"] == "sgw_data"}
        pgw_ids = {r["charging_id"] for r in data_records if r["record_type"] == "pgw_data"}

        assert sgw_ids == pgw_ids, (
            f"SGW and PGW charging_ids do not match.\n"
            f"  SGW-only: {sgw_ids - pgw_ids}\n"
            f"  PGW-only: {pgw_ids - sgw_ids}"
        )
