"""Generate cells, network elements, and subscribers from config."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from cdr_generator.assets.models import Cell, NetworkElement, Subscriber

if TYPE_CHECKING:
    from cdr_generator.config.models import CDRGeneratorConfig


def generate_cells(config: CDRGeneratorConfig, rng: random.Random) -> list[Cell]:
    """Generate cell objects from config.

    Supports inline cell definitions.  File-based and auto-generated modes
    are recognized but raise ``NotImplementedError`` until later phases.
    """
    network = config.network

    if network.auto_generate.enabled:
        raise NotImplementedError("auto_generate cells is not implemented in Phase 1")

    if network.cells.source == "file":
        raise NotImplementedError("file-based cell source is not implemented in Phase 1")

    # inline source
    cells: list[Cell] = []
    for item in network.cells.items:
        cells.append(
            Cell(
                cell_id=item.cell_id,
                tac=item.tac,
                ecgi=item.ecgi,
                lat=item.lat,
                lon=item.lon,
                azimuth=item.azimuth,
                sector=item.sector,
                cell_type=item.type.value,
                capacity=item.capacity.value,
                neighbors=list(item.neighbors),
            )
        )

    _validate_cells(cells)
    return cells


def _validate_cells(cells: list[Cell]) -> None:
    """Ensure all cell_ids are unique and all neighbor references are valid."""
    cell_ids = {c.cell_id for c in cells}

    seen: set[int] = set()
    for cell in cells:
        if cell.cell_id in seen:
            raise ValueError(f"Duplicate cell_id: {cell.cell_id}")
        seen.add(cell.cell_id)

        for neighbor_id in cell.neighbors:
            if neighbor_id not in cell_ids:
                raise ValueError(
                    f"Cell {cell.cell_id} references unknown neighbor {neighbor_id}"
                )


def generate_network_elements(config: CDRGeneratorConfig) -> list[NetworkElement]:
    """Convert config network element entries into NetworkElement objects."""
    elements: list[NetworkElement] = []
    seen_ids: set[str] = set()

    for item in config.network.elements:
        if item.id in seen_ids:
            raise ValueError(f"Duplicate network element id: {item.id}")
        seen_ids.add(item.id)

        elements.append(
            NetworkElement(
                id=item.id,
                ne_type=item.type.value,
                vendor=item.vendor.value,
                address=item.address,
                serves_tacs=list(item.serves_tacs),
                serves_apns=list(item.serves_apns) if item.serves_apns else [],
                extensions_template=dict(item.extensions_template),
                produces=list(item.produces),
            )
        )

    return elements


def generate_subscribers(
    config: CDRGeneratorConfig,
    cells: list[Cell],
    network_elements: list[NetworkElement],
    rng: random.Random,
) -> list[Subscriber]:
    """Generate ``total_count`` subscribers with deterministic assignment.

    Each subscriber gets a profile (weighted random), unique IMSI/MSISDN/IMEI,
    home and work cells based on the profile's mobility strategy, and a serving
    network element derived from the home cell's TAC.
    """
    total = config.subscribers.total_count
    imsi_prefix = config.subscribers.imsi_prefix
    msisdn_prefix = config.subscribers.msisdn_prefix

    # Pre-compute lookup structures
    cells_by_type = _group_cells_by_type(cells)
    tac_to_ne = _build_tac_to_ne_map(network_elements)

    # Build weighted profile list for selection
    profiles = config.subscribers.profiles
    profile_weights = [p.weight for p in profiles]

    # Determine zero-padding width for IMSI/MSISDN suffixes
    suffix_width = len(str(total - 1)) if total > 1 else 1

    subscribers: list[Subscriber] = []
    for i in range(total):
        # Select profile using weighted random choice
        profile = rng.choices(profiles, weights=profile_weights, k=1)[0]

        # Generate identifiers
        suffix = str(i).zfill(suffix_width)
        imsi = imsi_prefix + suffix
        msisdn = msisdn_prefix + suffix
        imei = _generate_imei(profile.imei_tac_pool, rng)

        # Assign cells
        home_cell = _pick_cell_by_strategy(
            profile.mobility.home_cell_strategy, cells, cells_by_type, rng
        )

        work_cell: Cell | None = None
        if profile.mobility.work_cell_strategy is not None:
            work_cell = _pick_cell_by_strategy(
                profile.mobility.work_cell_strategy, cells, cells_by_type, rng
            )

        # Determine serving NE from home cell's TAC
        serving_ne_id = _resolve_serving_ne(home_cell.tac, tac_to_ne, home_cell.cell_id)

        subscribers.append(
            Subscriber(
                imsi=imsi,
                msisdn=msisdn,
                imei=imei,
                profile_name=profile.name,
                home_cell_id=home_cell.cell_id,
                work_cell_id=work_cell.cell_id if work_cell else None,
                serving_ne_id=serving_ne_id,
            )
        )

    return subscribers


def generate_all_assets(
    config: CDRGeneratorConfig,
) -> tuple[list[Cell], list[NetworkElement], list[Subscriber]]:
    """Orchestrate full asset generation using config's seed for determinism."""
    rng = random.Random(config.meta.seed)
    cells = generate_cells(config, rng)
    network_elements = generate_network_elements(config)
    subscribers = generate_subscribers(config, cells, network_elements, rng)
    return cells, network_elements, subscribers


# Alias for backward compatibility with test imports
generate_assets = generate_all_assets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _group_cells_by_type(cells: list[Cell]) -> dict[str, list[Cell]]:
    """Group cells by their cell_type (urban, suburban, rural)."""
    groups: dict[str, list[Cell]] = {}
    for cell in cells:
        groups.setdefault(cell.cell_type, []).append(cell)
    return groups


def _build_tac_to_ne_map(network_elements: list[NetworkElement]) -> dict[int, str]:
    """Map TAC -> first NE id that serves voice (msc) for that TAC.

    For SGW/PGW the mapping is different, but for subscriber assignment
    we use the MSC serving the home cell's TAC.  If no MSC serves the TAC,
    fall back to the first NE that lists it.
    """
    tac_to_msc: dict[int, str] = {}
    tac_to_any: dict[int, str] = {}

    for ne in network_elements:
        for tac in ne.serves_tacs:
            if ne.ne_type == "msc" and tac not in tac_to_msc:
                tac_to_msc[tac] = ne.id
            if tac not in tac_to_any:
                tac_to_any[tac] = ne.id

    # Merge: prefer MSC, fall back to any NE
    merged: dict[int, str] = {}
    for tac in tac_to_any:
        merged[tac] = tac_to_msc.get(tac, tac_to_any[tac])
    return merged


def _pick_cell_by_strategy(
    strategy: str,
    all_cells: list[Cell],
    cells_by_type: dict[str, list[Cell]],
    rng: random.Random,
) -> Cell:
    """Pick a random cell matching the given strategy.

    Strategies:
      - random_urban: pick from urban cells
      - random_suburban: pick from suburban cells
      - random_rural: pick from rural cells
      - random_any: pick from all cells
    """
    if strategy == "random_any":
        candidates = all_cells
    elif strategy.startswith("random_"):
        cell_type = strategy.removeprefix("random_")
        candidates = cells_by_type.get(cell_type, [])
        if not candidates:
            candidates = all_cells
    else:
        raise ValueError(f"Unknown cell strategy: {strategy!r}")

    return rng.choice(candidates)


def _generate_imei(tac_pool: list[str], rng: random.Random) -> str:
    """Generate a 15-digit IMEI: TAC (8 digits) + serial (6 digits) + check (1 digit).

    The check digit uses the Luhn algorithm.
    """
    tac = rng.choice(tac_pool) if tac_pool else "35332508"
    serial = "".join(str(rng.randint(0, 9)) for _ in range(6))
    partial = tac + serial
    check = _luhn_check_digit(partial)
    return partial + str(check)


def _luhn_check_digit(number_str: str) -> int:
    """Compute Luhn check digit for a numeric string."""
    digits = [int(d) for d in number_str]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            doubled = d * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += d
    return (10 - (total % 10)) % 10


def _resolve_serving_ne(
    tac: int, tac_to_ne: dict[int, str], cell_id: int
) -> str:
    """Look up the serving NE for a given TAC, raising on missing mapping."""
    ne_id = tac_to_ne.get(tac)
    if ne_id is None:
        raise ValueError(
            f"No network element serves TAC {tac} (home cell {cell_id})"
        )
    return ne_id
