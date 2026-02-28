"""Shared test fixtures for CDR Generator tests."""

import pathlib
from typing import Any, Callable

import pytest
import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_CONFIG_PATH = REPO_ROOT / "cdr_generator_config.yaml"


@pytest.fixture
def sample_config_path() -> pathlib.Path:
    """Path to the reference cdr_generator_config.yaml."""
    assert SAMPLE_CONFIG_PATH.exists(), f"Sample config not found: {SAMPLE_CONFIG_PATH}"
    return SAMPLE_CONFIG_PATH


@pytest.fixture
def sample_config_dict(sample_config_path: pathlib.Path) -> dict:
    """Loaded dict from the reference YAML config."""
    with open(sample_config_path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "Config YAML must be a mapping"
    return data


@pytest.fixture
def tmp_assets_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Temporary directory for asset generation output."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    return assets_dir


@pytest.fixture
def tmp_output_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Temporary directory for CDR output files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def tmp_config_file(
    tmp_path: pathlib.Path,
) -> Callable[[dict[str, Any]], pathlib.Path]:
    """Factory fixture: write a dict as YAML to a temp file and return the path."""
    counter = 0

    def _write(data: dict[str, Any]) -> pathlib.Path:
        nonlocal counter
        counter += 1
        path = tmp_path / f"config_{counter}.yaml"
        path.write_text(yaml.dump(data, default_flow_style=False))
        return path

    return _write


@pytest.fixture
def minimal_valid_config_dict() -> dict:
    """Minimal valid config dict for quick tests that don't need full config."""
    return {
        "meta": {
            "seed": 42,
            "time_range": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-01T23:59:59Z",
            },
            "time_step_seconds": 60,
            "output": {
                "format": "csv_gzip",
                "path": "./output",
                "filename_template": "CDR_{ne_id}_{date}.csv.gz",
                "csv_delimiter": ",",
                "include_metadata_comment": True,
            },
            "parallelism": {"workers": 1},
        },
        "network": {
            "operator": {"mcc": "250", "mnc": "01", "name": "TestOp"},
            "elements": [
                {
                    "id": "msc-01",
                    "type": "msc",
                    "vendor": "ericsson",
                    "address": "10.0.1.1",
                    "serves_tacs": [2001],
                    "extensions_template": {},
                    "produces": ["mo_call", "mt_call"],
                }
            ],
            "cells": {
                "source": "inline",
                "file_path": None,
                "items": [
                    {
                        "cell_id": 20011,
                        "tac": 2001,
                        "ecgi": "25001-20011",
                        "lat": 55.7558,
                        "lon": 37.6173,
                        "azimuth": 0,
                        "sector": 1,
                        "type": "urban",
                        "capacity": "high",
                        "neighbors": [],
                    }
                ],
            },
            "auto_generate": {"enabled": False},
        },
        "subscribers": {
            "total_count": 100,
            "imsi_prefix": "250010",
            "msisdn_prefix": "+7916",
            "profiles": [
                {
                    "name": "office_worker",
                    "weight": 1.0,
                    "description": "Test profile",
                    "imei_tac_pool": ["35332508"],
                    "rat_preference": ["eutran"],
                    "daily_rates": {
                        "mo_call": {"type": "poisson", "params": {"lambda": 3.5}},
                        "mo_sms": {"type": "poisson", "params": {"lambda": 1.0}},
                        "data_session": {"type": "poisson", "params": {"lambda": 8.0}},
                        "mt_call": {"type": "poisson", "params": {"lambda": 1.0}},
                        "mt_sms": {"type": "poisson", "params": {"lambda": 0.5}},
                    },
                    "hourly_weights": {
                        "voice": [1] * 24,
                        "data": [1] * 24,
                        "sms": [1] * 24,
                    },
                    "day_of_week_multipliers": {
                        "voice": [1.0] * 7,
                        "data": [1.0] * 7,
                        "sms": [1.0] * 7,
                    },
                    "mobility": {
                        "home_cell_strategy": "random_urban",
                        "work_cell_strategy": "random_urban",
                        "commute_hours": [8, 9, 17, 18],
                        "roaming_probability": 0.01,
                        "handover_during_call": 0.05,
                    },
                }
            ],
            "contact_book": {
                "avg_contacts": 15,
                "degree_distribution": {
                    "type": "zipf",
                    "params": {"a": 2.0, "min": 3, "max": 100},
                },
                "asymmetric": True,
                "intra_profile_bias": 1.5,
                "repeat_call_probability": 0.6,
                "external_call_ratio": 0.15,
            },
            "external_numbers": {
                "count": 1000,
                "prefixes": [
                    {"prefix": "+7495", "weight": 1.0, "label": "test_landline"},
                ],
            },
        },
        "events": {
            "voice": {
                "success_rate": 0.85,
                "failure_causes": [
                    {"cause": "no_answer", "code": 19, "weight": 1.0},
                ],
                "duration": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 4.0, "sigma": 1.2},
                    },
                    "min_seconds": 1,
                    "max_seconds": 7200,
                },
                "setup_duration_ms": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 7.5, "sigma": 0.5},
                    },
                },
                "call_forwarding_rate": 0.03,
                "normal_termination_causes": [
                    {"cause": "normal_clearing", "code": 16, "weight": 1.0},
                ],
                "paired_timestamp_jitter_ms": 1000,
            },
            "data": {
                "duration": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 6.0, "sigma": 1.5},
                    },
                    "min_seconds": 5,
                    "max_seconds": 86400,
                },
                "volume_uplink": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 12.0, "sigma": 2.5},
                    },
                    "min_bytes": 100,
                },
                "volume_downlink": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 14.0, "sigma": 2.5},
                    },
                    "min_bytes": 100,
                },
                "profile_volume_multipliers": {},
                "apn_weights": {"internet": 1.0},
                "qos_distribution": [
                    {"qci": 9, "weight": 1.0, "label": "default_bearer"},
                ],
                "partial_records": {
                    "enabled": True,
                    "max_record_duration_seconds": 3600,
                    "max_record_volume_bytes": 104857600,
                },
                "termination_causes": [
                    {"cause": "normal_release", "weight": 1.0},
                ],
            },
            "sms": {
                "delivery_success_rate": 0.97,
                "delivery_delay": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 1.0, "sigma": 1.5},
                    },
                    "min_seconds": 0.5,
                    "max_seconds": 86400,
                },
                "failure_causes": [
                    {"cause": "absent_subscriber", "weight": 1.0},
                ],
            },
            "concurrency": {
                "max_voice": 1,
                "max_data": 1,
                "max_sms": None,
            },
        },
        "anomalies": {"enabled": False},
        "special_events": [],
        "vendor_extensions": {},
    }
