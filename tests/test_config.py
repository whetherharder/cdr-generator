"""Tests for config loading and validation.

Covers PM Task #15 test cases:
1.  test_load_valid_config -- full reference config loads, spot checks
2.  test_load_config_missing_file -- FileNotFoundError
3.  test_load_config_invalid_yaml -- malformed YAML
4.  test_load_config_missing_required_field -- removed required fields
5.  test_load_config_wrong_type -- seed="abc", etc.
6.  test_load_config_invalid_hourly_weights -- 23 elements
7.  test_load_config_invalid_dow_multipliers -- 6 elements
8.  test_apply_overrides_simple -- subscribers.total_count=500
9.  test_apply_overrides_nested -- meta.time_range.end
10. test_apply_overrides_invalid_path -- nonexistent.path
11. test_config_models_validation -- Pydantic rejects invalid data
12. test_config_roundtrip -- load and access all major fields

Acceptance criteria from plan.md Phase 1:
- Config loads successfully from valid YAML
- Invalid config raises error with path to the problem field
- --override dot-path applies correctly
- Schema validation catches structural errors
"""

import copy
import pathlib
from typing import Any, Callable

import pytest
import yaml
from pydantic import ValidationError

from cdr_generator.config.models import (
    CDRGeneratorConfig,
    DayOfWeekMultipliers,
    HourlyWeights,
    SubscribersConfig,
    TimeRange,
    VoiceEventConfig,
)


# ============================================================================
# 1. test_load_valid_config
# ============================================================================


class TestLoadValidConfig:
    """Loading the reference cdr_generator_config.yaml must succeed."""

    def test_load_valid_config(self, sample_config_path: pathlib.Path) -> None:
        """Load reference config and spot-check: seed=42, total_count=10000, 4 profiles."""
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)

        assert isinstance(config, CDRGeneratorConfig)
        assert config.meta.seed == 42
        assert config.subscribers.total_count == 10000
        assert len(config.subscribers.profiles) == 4

    def test_loaded_config_has_network_elements(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """All 5 NEs from reference config are present."""
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)
        ne_ids = {e.id for e in config.network.elements}
        assert ne_ids == {"msc-01", "msc-02", "sgw-01", "pgw-01", "smsc-01"}

    def test_loaded_config_has_cells(self, sample_config_path: pathlib.Path) -> None:
        """All 5 cells from reference config are present."""
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)
        assert len(config.network.cells.items) == 5

    def test_loaded_config_profile_names(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Four profile names match the YAML."""
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)
        names = {p.name for p in config.subscribers.profiles}
        assert names == {"office_worker", "student", "retiree", "heavy_traveler"}

    def test_loaded_config_time_range(self, sample_config_path: pathlib.Path) -> None:
        """Time range parses ISO 8601: 2025-01-01 to 2025-01-31."""
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)
        assert config.meta.time_range.start.year == 2025
        assert config.meta.time_range.start.month == 1
        assert config.meta.time_range.start.day == 1
        assert config.meta.time_range.end.day == 31

    def test_loaded_config_special_events(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Reference config has 3 special events."""
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)
        assert len(config.special_events) == 3
        event_names = {e.name for e in config.special_events}
        assert "new_year_midnight" in event_names

    def test_loaded_config_anomalies_enabled(
        self, sample_config_path: pathlib.Path
    ) -> None:
        """Reference config has anomalies enabled with correct sub-config rates."""
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)
        assert config.anomalies.enabled is True
        assert config.anomalies.duplicate_records.rate == pytest.approx(0.005)
        assert config.anomalies.orphaned_records.rate == pytest.approx(0.02)


# ============================================================================
# 2. test_load_config_missing_file
# ============================================================================


class TestLoadConfigMissingFile:
    """Loading a nonexistent file must raise FileNotFoundError or OSError."""

    def test_load_config_missing_file(self) -> None:
        from cdr_generator.config.loader import load_config

        with pytest.raises((FileNotFoundError, OSError)):
            load_config(pathlib.Path("/nonexistent/path/config.yaml"))


# ============================================================================
# 3. test_load_config_invalid_yaml
# ============================================================================


class TestLoadConfigInvalidYaml:
    """Malformed YAML must raise a clear error."""

    def test_malformed_yaml(self, tmp_path: pathlib.Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("meta:\n  seed: [unclosed bracket\n  broken: {")

        from cdr_generator.config.loader import load_config

        with pytest.raises(Exception):
            load_config(bad_file)

    def test_empty_file(self, tmp_path: pathlib.Path) -> None:
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        from cdr_generator.config.loader import load_config

        with pytest.raises(Exception):
            load_config(empty_file)

    def test_non_dict_yaml(self, tmp_path: pathlib.Path) -> None:
        """YAML that parses to a list instead of a dict should fail."""
        bad_file = tmp_path / "list.yaml"
        bad_file.write_text("- item1\n- item2\n")

        from cdr_generator.config.loader import load_config

        with pytest.raises(Exception):
            load_config(bad_file)


# ============================================================================
# 4. test_load_config_missing_required_field
# ============================================================================


class TestLoadConfigMissingRequiredField:
    """Missing required fields must produce errors mentioning the field."""

    def test_missing_meta(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        del cfg["meta"]

        with pytest.raises(Exception) as exc_info:
            load_config(tmp_config_file(cfg))
        assert "meta" in str(exc_info.value).lower()

    def test_missing_time_range(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        del cfg["meta"]["time_range"]

        with pytest.raises(Exception) as exc_info:
            load_config(tmp_config_file(cfg))
        assert "time_range" in str(exc_info.value).lower()

    def test_missing_network(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        del cfg["network"]

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_missing_subscribers(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        del cfg["subscribers"]

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_missing_events(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        del cfg["events"]

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_empty_profiles(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        """At least one subscriber profile is required (min_length=1)."""
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["subscribers"]["profiles"] = []

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))


# ============================================================================
# 5. test_load_config_wrong_type
# ============================================================================


class TestLoadConfigWrongType:
    """Fields with wrong types must produce validation errors."""

    def test_seed_string(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        """seed='abc' must fail."""
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["meta"]["seed"] = "abc"

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_total_count_string(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["subscribers"]["total_count"] = "many"

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_invalid_ne_type(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        """NE type not in enum (msc/sgw/pgw/smsc) must fail."""
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["network"]["elements"][0]["type"] = "unknown_type"

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_invalid_vendor(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["network"]["elements"][0]["vendor"] = "invalid_vendor"

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))


# ============================================================================
# 6. test_load_config_invalid_hourly_weights
# ============================================================================


class TestLoadConfigInvalidHourlyWeights:
    """Hourly weights must be exactly 24 elements."""

    def test_23_elements(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["subscribers"]["profiles"][0]["hourly_weights"]["voice"] = [1.0] * 23

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_25_elements(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["subscribers"]["profiles"][0]["hourly_weights"]["data"] = [1.0] * 25

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_empty_list(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["subscribers"]["profiles"][0]["hourly_weights"]["sms"] = []

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_pydantic_model_direct(self) -> None:
        """HourlyWeights model rejects wrong lengths directly."""
        with pytest.raises(ValidationError):
            HourlyWeights(voice=[1.0] * 10, data=[1.0] * 24, sms=[1.0] * 24)


# ============================================================================
# 7. test_load_config_invalid_dow_multipliers
# ============================================================================


class TestLoadConfigInvalidDowMultipliers:
    """Day-of-week multipliers must be exactly 7 elements."""

    def test_6_elements(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["subscribers"]["profiles"][0]["day_of_week_multipliers"]["voice"] = [
            1.0
        ] * 6

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_8_elements(
        self,
        minimal_valid_config_dict: dict,
        tmp_config_file: Callable[[dict], pathlib.Path],
    ) -> None:
        from cdr_generator.config.loader import load_config

        cfg = copy.deepcopy(minimal_valid_config_dict)
        cfg["subscribers"]["profiles"][0]["day_of_week_multipliers"]["data"] = [1.0] * 8

        with pytest.raises(Exception):
            load_config(tmp_config_file(cfg))

    def test_pydantic_model_direct(self) -> None:
        """DayOfWeekMultipliers model rejects wrong lengths directly."""
        with pytest.raises(ValidationError):
            DayOfWeekMultipliers(voice=[1.0] * 3, data=[1.0] * 7, sms=[1.0] * 7)


# ============================================================================
# 8. test_apply_overrides_simple
# ============================================================================


class TestApplyOverridesSimple:
    """Simple dot-path overrides must modify config values."""

    def test_override_subscriber_count(self, sample_config_path: pathlib.Path) -> None:
        """subscribers.total_count=500 -> config has 500."""
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(["subscribers.total_count=500"])
        config = load_config(sample_config_path, overrides=overrides)
        assert config.subscribers.total_count == 500

    def test_override_seed(self, sample_config_path: pathlib.Path) -> None:
        """meta.seed=999 -> config has 999."""
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(["meta.seed=999"])
        config = load_config(sample_config_path, overrides=overrides)
        assert config.meta.seed == 999

    def test_override_workers(self, sample_config_path: pathlib.Path) -> None:
        """meta.parallelism.workers=4 -> config has 4."""
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(["meta.parallelism.workers=4"])
        config = load_config(sample_config_path, overrides=overrides)
        assert config.meta.parallelism.workers == 4

    def test_multiple_overrides(self, sample_config_path: pathlib.Path) -> None:
        """Multiple overrides all apply simultaneously."""
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(
            [
                "subscribers.total_count=50",
                "meta.seed=7",
                "meta.parallelism.workers=2",
            ]
        )
        config = load_config(sample_config_path, overrides=overrides)
        assert config.subscribers.total_count == 50
        assert config.meta.seed == 7
        assert config.meta.parallelism.workers == 2


# ============================================================================
# 9. test_apply_overrides_nested
# ============================================================================


class TestApplyOverridesNested:
    """Nested dot-path overrides for deeply nested fields."""

    def test_override_time_range_end(self, sample_config_path: pathlib.Path) -> None:
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(["meta.time_range.end=2025-01-02T23:59:59Z"])
        config = load_config(sample_config_path, overrides=overrides)
        assert config.meta.time_range.end.day == 2

    def test_override_voice_success_rate(
        self, sample_config_path: pathlib.Path
    ) -> None:
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(["events.voice.success_rate=0.5"])
        config = load_config(sample_config_path, overrides=overrides)
        assert config.events.voice.success_rate == pytest.approx(0.5)

    def test_override_anomalies_enabled(self, sample_config_path: pathlib.Path) -> None:
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(["anomalies.enabled=false"])
        config = load_config(sample_config_path, overrides=overrides)
        assert config.anomalies.enabled is False


# ============================================================================
# 10. test_apply_overrides_invalid_path
# ============================================================================


class TestApplyOverridesInvalidPath:
    """Overriding a nonexistent path must raise ValueError."""

    def test_nonexistent_path(self, sample_config_path: pathlib.Path) -> None:
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(["nonexistent.path.here=123"])
        with pytest.raises((ValueError, KeyError, Exception)):
            load_config(sample_config_path, overrides=overrides)

    def test_deeply_nonexistent_path(self, sample_config_path: pathlib.Path) -> None:
        from cdr_generator.config.loader import load_config, parse_override_strings

        overrides = parse_override_strings(["meta.nonexistent.deep.path=value"])
        with pytest.raises((ValueError, KeyError, Exception)):
            load_config(sample_config_path, overrides=overrides)

    def test_malformed_override_no_equals(self) -> None:
        """Override string without '=' should fail at parse stage."""
        from cdr_generator.config.loader import parse_override_strings

        with pytest.raises(ValueError):
            parse_override_strings(["not_valid_format"])


# ============================================================================
# 11. test_config_models_validation
# ============================================================================


class TestConfigModelsValidation:
    """Pydantic models reject invalid data with clear errors."""

    def test_negative_total_count(self) -> None:
        """SubscribersConfig rejects total_count < 1."""
        with pytest.raises(ValidationError) as exc_info:
            SubscribersConfig(
                total_count=-5,
                profiles=[],
            )
        assert (
            "total_count" in str(exc_info.value)
            or "greater" in str(exc_info.value).lower()
        )

    def test_zero_total_count(self) -> None:
        """SubscribersConfig rejects total_count=0."""
        with pytest.raises(ValidationError):
            SubscribersConfig(total_count=0, profiles=[])

    def test_time_range_end_before_start(self) -> None:
        """TimeRange rejects end <= start."""
        with pytest.raises(ValidationError) as exc_info:
            TimeRange(
                start="2025-01-02T00:00:00Z",
                end="2025-01-01T00:00:00Z",
            )
        assert (
            "end" in str(exc_info.value).lower()
            or "after" in str(exc_info.value).lower()
        )

    def test_success_rate_above_1(self) -> None:
        """VoiceEventConfig rejects success_rate > 1.0."""
        with pytest.raises(ValidationError):
            VoiceEventConfig(
                success_rate=1.5,
                duration={
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 4, "sigma": 1},
                    },
                    "min_seconds": 1,
                },
            )

    def test_success_rate_negative(self) -> None:
        """VoiceEventConfig rejects success_rate < 0."""
        with pytest.raises(ValidationError):
            VoiceEventConfig(
                success_rate=-0.1,
                duration={
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 4, "sigma": 1},
                    },
                    "min_seconds": 1,
                },
            )

    def test_hourly_weights_wrong_length(self) -> None:
        with pytest.raises(ValidationError):
            HourlyWeights(voice=[1.0] * 10, data=[1.0] * 24, sms=[1.0] * 24)

    def test_dow_multipliers_wrong_length(self) -> None:
        with pytest.raises(ValidationError):
            DayOfWeekMultipliers(voice=[1.0] * 7, data=[1.0] * 3, sms=[1.0] * 7)

    def test_valid_config_from_full_dict(self, sample_config_dict: dict) -> None:
        """Full reference config dict passes Pydantic validation directly."""
        config = CDRGeneratorConfig(**sample_config_dict)
        assert config.meta.seed == 42

    def test_valid_config_from_minimal_dict(
        self, minimal_valid_config_dict: dict
    ) -> None:
        """Minimal config dict passes Pydantic validation directly."""
        config = CDRGeneratorConfig(**minimal_valid_config_dict)
        assert config.subscribers.total_count == 100


# ============================================================================
# 12. test_config_roundtrip
# ============================================================================


class TestConfigRoundtrip:
    """Load config and access every major field path without exceptions."""

    def test_config_roundtrip(self, sample_config_path: pathlib.Path) -> None:
        from cdr_generator.config.loader import load_config

        config = load_config(sample_config_path)

        # -- meta --
        assert isinstance(config.meta.seed, int)
        assert config.meta.time_range.start is not None
        assert config.meta.time_range.end is not None
        assert config.meta.time_step_seconds > 0
        assert config.meta.output.format == "csv_gzip"
        assert len(config.meta.output.filename_template) > 0
        assert config.meta.output.csv_delimiter == ","
        assert config.meta.parallelism.workers >= 1

        # -- network --
        assert len(config.network.operator.mcc) > 0
        assert len(config.network.operator.mnc) > 0
        assert len(config.network.operator.name) > 0
        for ne in config.network.elements:
            assert len(ne.id) > 0
            assert ne.type is not None
            assert ne.vendor is not None
        for cell in config.network.cells.items:
            assert cell.cell_id > 0
            assert cell.tac > 0
            assert len(cell.ecgi) > 0

        # -- subscribers --
        assert config.subscribers.total_count > 0
        assert len(config.subscribers.imsi_prefix) > 0
        assert len(config.subscribers.msisdn_prefix) > 0
        for profile in config.subscribers.profiles:
            assert len(profile.name) > 0
            assert 0 <= profile.weight <= 1
            assert len(profile.hourly_weights.voice) == 24
            assert len(profile.hourly_weights.data) == 24
            assert len(profile.hourly_weights.sms) == 24
            assert len(profile.day_of_week_multipliers.voice) == 7
            assert len(profile.day_of_week_multipliers.data) == 7
            assert len(profile.day_of_week_multipliers.sms) == 7
            assert profile.daily_rates.mo_call.type is not None
            assert profile.mobility.home_cell_strategy is not None
        assert config.subscribers.contact_book.avg_contacts > 0
        assert config.subscribers.external_numbers.count > 0

        # -- events --
        assert 0 <= config.events.voice.success_rate <= 1
        assert config.events.voice.duration is not None
        assert config.events.data.duration is not None
        assert 0 <= config.events.sms.delivery_success_rate <= 1

        # -- anomalies --
        assert isinstance(config.anomalies.enabled, bool)
        assert 0 <= config.anomalies.duplicate_records.rate <= 1
        assert 0 <= config.anomalies.missing_fields.rate <= 1
        assert 0 <= config.anomalies.orphaned_records.rate <= 1

        # -- special events --
        for event in config.special_events:
            assert len(event.name) > 0
            assert event.effect is not None

        # -- vendor extensions --
        for vendor_name, ext in config.vendor_extensions.items():
            assert len(vendor_name) > 0
            for field in ext.fields:
                assert len(field.key) > 0
