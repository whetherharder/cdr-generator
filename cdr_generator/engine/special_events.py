"""Special event evaluation engine.

Evaluates which special events are active at a given timestamp and
computes the compound effect across all active events.  Supports both
time-ranged events (e.g., New Year midnight spike) and recurring events
(e.g., weekday lunch hours).

Compound rules when multiple events overlap:
- Rate multipliers: **multiplied** together (e.g., 2.0 * 1.5 = 3.0).
- ``voice_failure_rate_override``: **maximum** is taken (worst case).
- Cell sets (disabled, overflow, congestion): **union** across events.

Usage::

    engine = SpecialEventEngine(config.special_events)
    effects = engine.get_active_effects(timestamp)
    voice_mult = effects.voice_rate_multiplier  # e.g., 5.0
    if effects.is_cell_disabled(cell_id):
        cell_id = effects.pick_overflow_cell(rng)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cdr_generator.assets.models import Cell
    from cdr_generator.config.models import SpecialEventConfig


# ---------------------------------------------------------------------------
# Active effects snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveEffects:
    """Compound effects of all special events active at a given timestamp.

    This is a frozen snapshot -- create a new one for each timestamp.

    Attributes
    ----------
    voice_rate_multiplier:
        Compound multiplier for voice event rates (default 1.0 = no change).
    sms_rate_multiplier:
        Compound multiplier for SMS event rates.
    data_rate_multiplier:
        Compound multiplier for data event rates.
    voice_failure_rate_override:
        If set, overrides the configured voice success_rate.
        When multiple events specify this, the maximum (worst) is used.
        ``None`` means no override.
    disabled_cells:
        Union of all disabled cell IDs across active events.
        Subscribers at these cells should be relocated to overflow cells
        or experience failures.
    overflow_cells:
        Union of all overflow cell IDs.  Subscribers displaced from
        disabled cells are relocated here.
    congestion_cells:
        Union of all congestion cell IDs.  These cells experience
        elevated failure rates (implementation in Phase 5).
    active_event_names:
        Names of all currently active events (for logging/debugging).
    """

    voice_rate_multiplier: float = 1.0
    sms_rate_multiplier: float = 1.0
    data_rate_multiplier: float = 1.0
    voice_failure_rate_override: float | None = None
    disabled_cells: frozenset[int] = field(default_factory=frozenset)
    overflow_cells: frozenset[int] = field(default_factory=frozenset)
    congestion_cells: frozenset[int] = field(default_factory=frozenset)
    active_event_names: tuple[str, ...] = ()

    def is_cell_disabled(self, cell_id: int) -> bool:
        """Check whether a cell is disabled by any active event.

        Parameters
        ----------
        cell_id:
            Cell ID to check.

        Returns
        -------
        bool
            True if the cell is in the disabled set.
        """
        ...

    def has_overflow(self) -> bool:
        """Check whether overflow cells are available.

        Returns
        -------
        bool
            True if at least one overflow cell is configured.
        """
        ...

    def pick_overflow_cell(self, rng: np.random.Generator) -> int | None:
        """Select a random overflow cell.

        Parameters
        ----------
        rng:
            Numpy RNG for deterministic selection.

        Returns
        -------
        int | None
            An overflow cell ID, or ``None`` if no overflow cells exist.
        """
        ...

    @property
    def is_active(self) -> bool:
        """True if any special event is currently active."""
        ...


# ---------------------------------------------------------------------------
# Special event engine
# ---------------------------------------------------------------------------

# Day name to weekday index mapping (Monday=0 ... Sunday=6)
_DAY_NAME_TO_INDEX: dict[str, int] = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


class SpecialEventEngine:
    """Evaluates special events and computes compound effects.

    Pre-processes the event list at initialization for efficient
    per-timestamp lookups during generation.

    Parameters
    ----------
    events:
        List of special event configurations from the CDR generator config.
    """

    def __init__(self, events: list[SpecialEventConfig]) -> None:
        ...

    def get_active_effects(self, timestamp: datetime) -> ActiveEffects:
        """Compute the compound effects of all events active at *timestamp*.

        Algorithm
        ---------
        1. Check each event for activity:
           - Time-ranged: ``start <= timestamp < end``.
           - Recurring: day-of-week and hour match.
        2. For all active events, compound their effects:
           - Rate multipliers: multiply together.
           - ``voice_failure_rate_override``: take maximum.
           - Cell sets: union.

        Parameters
        ----------
        timestamp:
            The current simulation timestamp (UTC).

        Returns
        -------
        ActiveEffects
            Compound effects snapshot.  If no events are active,
            returns a default ``ActiveEffects`` (all multipliers = 1.0,
            no overrides, empty cell sets).
        """
        ...

    def _is_event_active(
        self,
        event: SpecialEventConfig,
        timestamp: datetime,
    ) -> bool:
        """Check whether a single event is active at the given timestamp.

        Parameters
        ----------
        event:
            Special event configuration.
        timestamp:
            Current simulation timestamp.

        Returns
        -------
        bool
            True if the event is active.
        """
        ...

    def _is_recurrence_active(
        self,
        days: list[str],
        hours: list[int],
        timestamp: datetime,
    ) -> bool:
        """Check whether a recurring event matches the given timestamp.

        Parameters
        ----------
        days:
            Day-of-week names (e.g., ``["mon", "tue"]``).
        hours:
            Hours during which the event is active (e.g., ``[12, 13]``).
        timestamp:
            Current simulation timestamp.

        Returns
        -------
        bool
            True if the day and hour both match.
        """
        ...


# ---------------------------------------------------------------------------
# No-op singleton for when special events are disabled
# ---------------------------------------------------------------------------

#: Default effects when no special events are active.
NO_EFFECTS = ActiveEffects()
