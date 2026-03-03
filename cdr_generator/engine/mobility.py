"""Subscriber mobility -- position resolution and handover logic.

Determines which cell a subscriber is at during a given time step,
and whether a handover occurs during the event (first_cell != last_cell).

The position depends on time-of-day, profile mobility config, and
the cell topology (neighbors).

Usage::

    first_cell, last_cell = resolve_position(
        home_cell_id, work_cell_id, mobility, timestamp, cells_by_id, rng
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cdr_generator.assets.models import Cell
    from cdr_generator.config.models import MobilityConfig
    from cdr_generator.engine.rng_buffer import _RngBuffer


def resolve_position(
    home_cell_id: int,
    work_cell_id: int | None,
    mobility: MobilityConfig,
    timestamp: datetime,
    cells_by_id: dict[int, Cell],
    rng: np.random.Generator,
    all_cell_ids: list[int] | None = None,
    _buf: _RngBuffer | None = None,
) -> tuple[int, int]:
    """Determine the subscriber's cell position at a given time.

    Returns a tuple of (first_cell_id, last_cell_id). If a handover
    occurs during the event, last_cell_id will differ from first_cell_id.

    Position logic by time of day:
    - Night (22:00--06:00): home cell
    - Commute hours: transit cell (neighbor of home or work)
    - Work hours (remaining daytime): work cell if configured, else home

    Roaming override: with probability roaming_probability, the subscriber
    is placed at a random cell.

    Parameters
    ----------
    home_cell_id:
        The subscriber's home cell ID.
    work_cell_id:
        The subscriber's work cell ID (may be None).
    mobility:
        MobilityConfig Pydantic model with commute_hours,
        roaming_probability, handover_during_call.
    timestamp:
        Current event timestamp.
    cells_by_id:
        Lookup map of all cells.
    rng:
        Numpy random generator.
    all_cell_ids:
        Pre-computed list of all cell IDs (for roaming).  When provided,
        avoids repeated ``list(cells_by_id.keys())`` calls.
    _buf:
        Optional pre-filled RNG buffer.  When provided, uniform float draws
        (roaming and handover checks) are served from the pre-filled batch
        instead of calling rng.random() directly (~8x lower per-call cost).

    Returns
    -------
    tuple[int, int]
        (first_cell_id, last_cell_id)
    """
    # Check roaming first
    roaming_prob = mobility.roaming_probability
    rand_fn = _buf.get_float if _buf is not None else lambda: float(rng.random())
    if roaming_prob > 0 and rand_fn() < roaming_prob:
        if all_cell_ids is None:
            all_cell_ids = list(cells_by_id.keys())
        if all_cell_ids:
            idx = int(rng.integers(0, len(all_cell_ids)))
            first_cell = all_cell_ids[idx]
            last_cell = _maybe_handover(first_cell, mobility, cells_by_id, rng, _buf)
            return first_cell, last_cell

    hour = timestamp.hour
    commute_hours = mobility.commute_hours

    # Night hours: home
    if hour >= 22 or hour < 6:
        first_cell = home_cell_id
    # Commute hours: transit (neighbor of home or work)
    elif commute_hours and hour in commute_hours:
        first_cell = _commute_cell(home_cell_id, work_cell_id, cells_by_id, rng, _buf)
    # Work hours: work cell if available
    elif work_cell_id is not None:
        first_cell = work_cell_id
    else:
        first_cell = home_cell_id

    last_cell = _maybe_handover(first_cell, mobility, cells_by_id, rng, _buf)
    return first_cell, last_cell


def _maybe_handover(
    first_cell_id: int,
    mobility: MobilityConfig,
    cells_by_id: dict[int, Cell],
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int:
    """Possibly return a different (neighbor) cell for handover."""
    handover_prob = mobility.handover_during_call
    rand_val = _buf.get_float() if _buf is not None else float(rng.random())
    if handover_prob <= 0 or rand_val >= handover_prob:
        return first_cell_id

    cell = cells_by_id.get(first_cell_id)
    if cell is None or not cell.neighbors:
        return first_cell_id

    valid_neighbors = [n for n in cell.neighbors if n in cells_by_id]
    if not valid_neighbors:
        return first_cell_id

    idx = int(rng.integers(0, len(valid_neighbors)))
    return valid_neighbors[idx]


def _commute_cell(
    home_cell_id: int,
    work_cell_id: int | None,
    cells_by_id: dict[int, Cell],
    rng: np.random.Generator,
    _buf: _RngBuffer | None = None,
) -> int:
    """Select a transit cell during commute hours."""
    # Pick base: either home or work (50/50 if work exists)
    rand_val = _buf.get_float() if _buf is not None else float(rng.random())
    if work_cell_id is not None and rand_val < 0.5:
        base_id = work_cell_id
    else:
        base_id = home_cell_id

    base_cell = cells_by_id.get(base_id)
    if base_cell and base_cell.neighbors:
        valid_neighbors = [n for n in base_cell.neighbors if n in cells_by_id]
        if valid_neighbors:
            idx = int(rng.integers(0, len(valid_neighbors)))
            return valid_neighbors[idx]

    return base_id
