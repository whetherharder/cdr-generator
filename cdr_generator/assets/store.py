"""Save, load, and validate assets on disk with manifest tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from cdr_generator.assets.models import AssetManifest, Cell, NetworkElement, Subscriber

_CELLS_FILE = "cells.json"
_NE_FILE = "network_elements.json"
_SUBSCRIBERS_FILE = "subscribers.json"
_MANIFEST_FILE = "manifest.json"

_CellListAdapter = TypeAdapter(list[Cell])
_NEListAdapter = TypeAdapter(list[NetworkElement])
_SubscriberListAdapter = TypeAdapter(list[Subscriber])


def save_assets(
    assets_dir: Path,
    cells: list[Cell],
    network_elements: list[NetworkElement],
    subscribers: list[Subscriber],
    config_hash: str,
) -> None:
    """Persist all assets and a manifest to *assets_dir*.

    Creates the directory (and parents) if it does not exist.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)

    _write_json(assets_dir / _CELLS_FILE, _CellListAdapter.dump_python(cells, mode="json"))
    _write_json(assets_dir / _NE_FILE, _NEListAdapter.dump_python(network_elements, mode="json"))
    _write_json(assets_dir / _SUBSCRIBERS_FILE, _SubscriberListAdapter.dump_python(subscribers, mode="json"))

    manifest = AssetManifest(
        config_hash=config_hash,
        generated_at=datetime.now(timezone.utc),
        cell_count=len(cells),
        ne_count=len(network_elements),
        subscriber_count=len(subscribers),
    )
    _write_json(assets_dir / _MANIFEST_FILE, manifest.model_dump(mode="json"))


def load_assets(
    assets_dir: Path,
) -> tuple[list[Cell], list[NetworkElement], list[Subscriber], AssetManifest]:
    """Load previously saved assets and manifest from *assets_dir*.

    Raises ``FileNotFoundError`` if any expected file is missing.
    """
    cells = _CellListAdapter.validate_python(_read_json(assets_dir / _CELLS_FILE))
    network_elements = _NEListAdapter.validate_python(_read_json(assets_dir / _NE_FILE))
    subscribers = _SubscriberListAdapter.validate_python(_read_json(assets_dir / _SUBSCRIBERS_FILE))
    manifest = AssetManifest.model_validate(_read_json(assets_dir / _MANIFEST_FILE))
    return cells, network_elements, subscribers, manifest


def validate_assets(
    cells: list[Cell],
    network_elements: list[NetworkElement],
    subscribers: list[Subscriber],
) -> list[str]:
    """Cross-validate referential integrity across all assets.

    Returns a list of error strings.  An empty list means the assets are valid.
    """
    errors: list[str] = []

    cell_ids = {c.cell_id for c in cells}
    ne_ids = {ne.id for ne in network_elements}
    tacs_in_cells = {c.tac for c in cells}

    # Validate cell neighbor references
    for cell in cells:
        for neighbor_id in cell.neighbors:
            if neighbor_id not in cell_ids:
                errors.append(
                    f"Cell {cell.cell_id}: neighbor {neighbor_id} does not exist"
                )

    # Validate NE serves_tacs have at least one cell
    for ne in network_elements:
        for tac in ne.serves_tacs:
            if tac not in tacs_in_cells:
                errors.append(
                    f"NetworkElement {ne.id}: serves TAC {tac} but no cell has that TAC"
                )

    # Validate subscriber references
    for sub in subscribers:
        if sub.home_cell_id not in cell_ids:
            errors.append(
                f"Subscriber {sub.imsi}: home_cell_id {sub.home_cell_id} does not exist"
            )
        if sub.work_cell_id is not None and sub.work_cell_id not in cell_ids:
            errors.append(
                f"Subscriber {sub.imsi}: work_cell_id {sub.work_cell_id} does not exist"
            )
        if sub.serving_ne_id not in ne_ids:
            errors.append(
                f"Subscriber {sub.imsi}: serving_ne_id {sub.serving_ne_id!r} does not exist"
            )

    return errors


def compute_config_hash(config_path: Path) -> str:
    """Return the SHA-256 hex digest of the raw config file content."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
