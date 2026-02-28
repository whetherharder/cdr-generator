"""YAML config loader with JSON Schema validation and dot-path overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from pydantic import ValidationError

from cdr_generator.config.models import CDRGeneratorConfig

_SCHEMA_PATH = Path(__file__).parent / "schema.json"


def load_config(
    path: str | Path,
    overrides: list[str] | dict[str, Any] | None = None,
) -> CDRGeneratorConfig:
    """Load a YAML config, validate, apply overrides, and return the model.

    Pipeline:
      1. Read and parse YAML
      2. Validate raw dict against JSON Schema (if schema is non-empty)
      3. Apply dot-path overrides
      4. Parse into Pydantic ``CDRGeneratorConfig``

    *overrides* accepts either:
      - ``list[str]`` of ``"key=value"`` strings (CLI style)
      - ``dict[str, Any]`` of ``{dot_path: value}`` pairs

    Raises:
        FileNotFoundError: *path* does not exist.
        ConfigLoadError: YAML parse error, schema violation, or model validation failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")

    # 1. YAML parse
    try:
        config_dict = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        msg = f"YAML parse error in {path}"
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            mark = exc.problem_mark
            msg += f" at line {mark.line + 1}, column {mark.column + 1}"
        raise ConfigLoadError(msg) from exc

    if not isinstance(config_dict, dict):
        raise ConfigLoadError(
            f"Expected a YAML mapping at top level, got {type(config_dict).__name__}"
        )

    # 2. JSON Schema validation (optional -- skipped when schema is empty)
    _validate_json_schema(config_dict)

    # 3. Overrides -- accept both list[str] and dict
    if overrides:
        if isinstance(overrides, list):
            overrides_dict = parse_override_strings(overrides)
        else:
            overrides_dict = overrides
        config_dict = apply_overrides(config_dict, overrides_dict)

    # 4. Pydantic validation
    try:
        return CDRGeneratorConfig.model_validate(config_dict)
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(f"  {loc}: {err['msg']}")
        raise ConfigLoadError(
            "Config validation failed:\n" + "\n".join(errors)
        ) from exc


def validate_config(path: Path) -> list[str]:
    """Load and validate a config file, returning a list of error messages.

    An empty list means the config is valid.
    """
    try:
        load_config(path)
        return []
    except FileNotFoundError as exc:
        return [str(exc)]
    except ConfigLoadError as exc:
        return [str(exc)]


def apply_overrides(
    config_dict: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Apply dot-path overrides to a raw config dict.

    Example:
        ``{"subscribers.total_count": "500"}`` sets
        ``config_dict["subscribers"]["total_count"]`` to ``500`` (coerced to int).

    Returns the mutated *config_dict*.
    """
    for dot_path, value in overrides.items():
        parts = dot_path.split(".")
        target = config_dict
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(
                    f"Override path not found: {dot_path!r} (missing key {part!r})"
                )
            target = target[part]

        final_key = parts[-1]
        if not isinstance(target, dict):
            raise ValueError(
                f"Override path not found: {dot_path!r} "
                f"(intermediate value is not a mapping)"
            )

        # Type coercion: try to match the type of the existing value
        if final_key in target and isinstance(value, str):
            value = _coerce_type(value, type(target[final_key]))

        target[final_key] = value
    return config_dict


def parse_override_strings(
    override_strings: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Parse CLI ``--override key=value`` strings into a dict.

    Each string must be in the form ``dot.path=value``.
    """
    overrides: dict[str, Any] = {}
    for s in override_strings:
        if "=" not in s:
            raise ValueError(f"Invalid override format (expected key=value): {s!r}")
        key, _, value = s.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty key in override: {s!r}")
        overrides[key] = value
    return overrides


class ConfigLoadError(Exception):
    """Raised when config loading fails at any stage."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_json_schema(config_dict: dict[str, Any]) -> None:
    """Validate *config_dict* against the JSON Schema, if available."""
    if not _SCHEMA_PATH.exists():
        return

    try:
        raw = _SCHEMA_PATH.read_text(encoding="utf-8")
        schema = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return

    # Skip if schema is empty (placeholder)
    if not schema:
        return

    try:
        jsonschema.validate(instance=config_dict, schema=schema)
    except jsonschema.ValidationError as exc:
        json_path = (
            "$.%s" % ".".join(str(p) for p in exc.absolute_path)
            if exc.absolute_path
            else "$"
        )
        raise ConfigLoadError(
            f"JSON Schema validation error at {json_path}: {exc.message}"
        ) from exc


def _coerce_type(value: str, target_type: type) -> Any:
    """Best-effort coercion of a string *value* to *target_type*."""
    if target_type is bool:
        lowered = value.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return value
    if target_type is int:
        try:
            return int(value)
        except ValueError:
            return value
    if target_type is float:
        try:
            return float(value)
        except ValueError:
            return value
    return value
