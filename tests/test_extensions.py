"""Tests for generators/extensions.py -- Vendor-specific extensions.

Phase 3 acceptance criteria:
- Extensions: base64-encoded JSON, keys {vendor}.{field}
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from cdr_generator.config.models import (
    VendorExtensionConfig,
    VendorFieldConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def ericsson_vendor_config() -> VendorExtensionConfig:
    """Vendor extension config for Ericsson."""
    return VendorExtensionConfig(
        fields=[
            VendorFieldConfig(key="ericsson.sw_version", value="R16B"),
            VendorFieldConfig(
                key="ericsson.charging_output_type",
                value={
                    "type": "categorical",
                    "params": {
                        "values": ["1", "2", "3"],
                        "weights": [0.7, 0.2, 0.1],
                    },
                },
            ),
        ]
    )


@pytest.fixture
def nokia_vendor_config() -> VendorExtensionConfig:
    """Vendor extension config for Nokia."""
    return VendorExtensionConfig(
        fields=[
            VendorFieldConfig(key="nokia.platform", value="AirScale"),
            VendorFieldConfig(key="nokia.release", value="21.6"),
        ]
    )


@pytest.fixture
def event_context() -> dict:
    """Sample event context for extensions that reference CDR fields."""
    return {
        "first_cell_id": 20011,
        "last_cell_id": 20011,
        "apn": "internet",
    }


# ===================================================================
# Extension generation
# ===================================================================


class TestExtensionGeneration:
    """generate_extensions returns a base64-encoded JSON string."""

    def test_returns_base64_string(
        self, ericsson_vendor_config, event_context, rng
    ) -> None:
        from cdr_generator.generators.extensions import generate_extensions

        result = generate_extensions(
            "ericsson",
            ericsson_vendor_config,
            event_context,
            rng,
        )
        assert isinstance(result, str), "Extensions must be a string"
        assert len(result) > 0, "Extensions string must not be empty"

        # Must be valid base64
        try:
            base64.b64decode(result)
        except Exception as e:
            pytest.fail(f"Extensions must be valid base64: {e}")

    def test_decoded_is_valid_json(
        self, ericsson_vendor_config, event_context, rng
    ) -> None:
        from cdr_generator.generators.extensions import generate_extensions

        result = generate_extensions(
            "ericsson",
            ericsson_vendor_config,
            event_context,
            rng,
        )
        decoded = base64.b64decode(result)
        try:
            data = json.loads(decoded)
        except json.JSONDecodeError as e:
            pytest.fail(f"Decoded base64 must be valid JSON: {e}")

        assert isinstance(data, dict), "Decoded JSON must be a dict"

    def test_keys_follow_vendor_field_pattern(
        self, ericsson_vendor_config, event_context, rng
    ) -> None:
        """All keys in the decoded JSON should follow {vendor}.{field} pattern."""
        from cdr_generator.generators.extensions import generate_extensions

        result = generate_extensions(
            "ericsson",
            ericsson_vendor_config,
            event_context,
            rng,
        )
        decoded = json.loads(base64.b64decode(result))

        for key in decoded:
            assert "." in key, (
                f"Extension key '{key}' must follow {{vendor}}.{{field}} pattern"
            )
            vendor_part, field_part = key.split(".", 1)
            assert len(vendor_part) > 0, f"Vendor part of key '{key}' must not be empty"
            assert len(field_part) > 0, f"Field part of key '{key}' must not be empty"

    def test_ericsson_keys_present(
        self, ericsson_vendor_config, event_context, rng
    ) -> None:
        from cdr_generator.generators.extensions import generate_extensions

        result = generate_extensions(
            "ericsson",
            ericsson_vendor_config,
            event_context,
            rng,
        )
        decoded = json.loads(base64.b64decode(result))

        assert "ericsson.sw_version" in decoded, (
            "Ericsson NE extensions must contain ericsson.sw_version"
        )
        assert decoded["ericsson.sw_version"] == "R16B"

    def test_nokia_keys_present(self, nokia_vendor_config, event_context, rng) -> None:
        from cdr_generator.generators.extensions import generate_extensions

        result = generate_extensions(
            "nokia",
            nokia_vendor_config,
            event_context,
            rng,
        )
        decoded = json.loads(base64.b64decode(result))

        assert "nokia.platform" in decoded, (
            "Nokia NE extensions must contain nokia.platform"
        )
        assert decoded["nokia.platform"] == "AirScale"


class TestExtensionWithDistribution:
    """Extension values can be generated from distributions."""

    def test_categorical_distribution_generates_valid_values(
        self, ericsson_vendor_config, event_context, rng
    ) -> None:
        from cdr_generator.generators.extensions import generate_extensions

        result = generate_extensions(
            "ericsson",
            ericsson_vendor_config,
            event_context,
            rng,
        )
        decoded = json.loads(base64.b64decode(result))

        if "ericsson.charging_output_type" in decoded:
            assert decoded["ericsson.charging_output_type"] in ["1", "2", "3"], (
                "Categorical value must be one of the configured choices"
            )


class TestExtensionDeterministic:
    """Extension generation must be deterministic with the same seed."""

    def test_same_seed_same_result(self, ericsson_vendor_config, event_context) -> None:
        from cdr_generator.generators.extensions import generate_extensions

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = generate_extensions(
            "ericsson",
            ericsson_vendor_config,
            event_context,
            rng1,
        )
        result2 = generate_extensions(
            "ericsson",
            ericsson_vendor_config,
            event_context,
            rng2,
        )
        assert result1 == result2, "Same seed must produce identical extensions"


class TestExtensionNoneForNoConfig:
    """NE with no vendor config should get None extensions."""

    def test_no_config_returns_none(self, rng) -> None:
        from cdr_generator.generators.extensions import generate_extensions

        result = generate_extensions("huawei", None, {}, rng)
        assert result is None, (
            f"NE with no vendor config should have None extensions, got: {result}"
        )

    def test_empty_fields_returns_none(self, rng) -> None:
        from cdr_generator.generators.extensions import generate_extensions

        empty_config = VendorExtensionConfig(fields=[])
        result = generate_extensions("huawei", empty_config, {}, rng)
        assert result is None, (
            f"NE with empty fields should have None extensions, got: {result}"
        )
