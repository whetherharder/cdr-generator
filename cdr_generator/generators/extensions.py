"""Vendor extensions generator -- produces base64-encoded JSON fields.

Each network element has a vendor (ericsson, huawei, nokia).  The
vendor_extensions config section defines per-vendor field generators.
This module evaluates those generators and encodes the result as a
base64 JSON string suitable for the ``vendor_extensions`` CDR field.

Field values can be:
- A distribution descriptor (sampled at generation time)
- A literal string value
- The special string ``"from_event"`` (copies the value from the
  corresponding CDR field)

Usage::

    ext_str = generate_extensions(
        vendor="ericsson",
        vendor_config=vendor_ext_config,
        event_context=ctx,
        rng=np_rng,
    )
    # ext_str is a base64-encoded JSON string or None
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

from cdr_generator.engine.distributions import sample_distribution


def generate_extensions(
    vendor: str,
    vendor_config: Any | None,
    event_context: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> str | None:
    """Generate the vendor_extensions field value for a CDR record.

    Parameters
    ----------
    vendor:
        The vendor name (e.g. "ericsson", "nokia").
    vendor_config:
        VendorExtensionConfig Pydantic model with a ``fields`` list,
        or None if no config for this vendor.
    event_context:
        Optional dict of CDR field values for "from_event" resolution.
    rng:
        Numpy random generator for sampling distribution-based fields.

    Returns
    -------
    str | None
        Base64-encoded JSON string containing the vendor extension
        fields, or ``None`` if no config for this vendor.
    """
    if vendor_config is None:
        return None

    # Support both Pydantic model (with .fields) and plain dict
    if hasattr(vendor_config, "fields"):
        field_defs = vendor_config.fields
    elif isinstance(vendor_config, dict):
        # Dict keyed by vendor name (legacy format)
        vendor_entry = vendor_config.get(vendor)
        if vendor_entry is None:
            return None
        field_defs = vendor_entry.get("fields", [])
    else:
        return None

    if not field_defs:
        return None

    ctx = event_context or {}
    fields: dict[str, str] = {}

    for field_def in field_defs:
        # Support both Pydantic VendorFieldConfig and plain dict
        if hasattr(field_def, "key"):
            key = field_def.key
            value = field_def.value
        else:
            key = field_def.get("key", "")
            value = field_def.get("value")
        fields[key] = _resolve_field_value(value, key, ctx, rng)

    if not fields:
        return None

    return _encode_extensions(fields)


def _resolve_field_value(
    value: Any,
    key: str,
    event_context: dict[str, Any],
    rng: np.random.Generator | None,
) -> str:
    """Resolve a single vendor extension field value."""
    # Handle "from_event"
    if value == "from_event":
        field_name = key.split(".", 1)[-1] if "." in key else key
        ctx_val = event_context.get(field_name)
        return str(ctx_val) if ctx_val is not None else ""

    # Handle distribution descriptor (dict with "type")
    if isinstance(value, dict):
        dist_type = value.get("type", "")
        params = value.get("params", {})
        if dist_type == "categorical":
            values_list = params.get("values", params.get("choices", []))
            weights_list = params.get("weights", [])
            return _sample_categorical(values_list, weights_list, rng)
        if rng is not None:
            sampled = sample_distribution(dist_type, params, rng)
            if sampled == int(sampled):
                return str(int(sampled))
            return str(sampled)
        return str(value)

    # Handle Pydantic Distribution model
    if hasattr(value, "type") and hasattr(value, "params"):
        dist_type = value.type
        params = value.params if isinstance(value.params, dict) else {}
        if dist_type == "categorical":
            values_list = params.get("values", params.get("choices", []))
            weights_list = params.get("weights", [])
            return _sample_categorical(values_list, weights_list, rng)
        if rng is not None:
            sampled = sample_distribution(dist_type, params, rng)
            if sampled == int(sampled):
                return str(int(sampled))
            return str(sampled)
        return str(value)

    # Literal string value
    return str(value)


def _sample_categorical(
    values: list[str],
    weights: list[float],
    rng: np.random.Generator | None,
) -> str:
    """Sample a value from a categorical distribution."""
    if not values:
        return ""
    if rng is None:
        return values[0]
    if not weights or len(weights) != len(values):
        idx = int(rng.integers(0, len(values)))
        return values[idx]

    total = sum(weights)
    if total <= 0:
        idx = int(rng.integers(0, len(values)))
        return values[idx]

    probs = [w / total for w in weights]
    idx = int(rng.choice(len(values), p=probs))
    return values[idx]


def _encode_extensions(fields: dict[str, str]) -> str:
    """Encode extension fields as a base64 JSON string."""
    json_str = json.dumps(fields, separators=(",", ":"))
    return base64.b64encode(json_str.encode("utf-8")).decode("ascii")
