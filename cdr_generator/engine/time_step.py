"""Time-step iteration engine for CDR generation.

Iterates over every time step in the configured time range, yielding
``(timestamp, active_subscribers)`` pairs. In Phase 2 all subscribers
are active at every step (no mobility or roaming filtering).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

from cdr_generator.assets.models import Subscriber
from cdr_generator.config.models import CDRGeneratorConfig


class Assets:
    """Container for pre-loaded runtime assets."""

    def __init__(
        self,
        subscribers: list[Subscriber],
        cells_by_id: dict[int, object],
        nes_by_id: dict[str, object],
        tac_to_msc: dict[int, str],
        tac_to_sgw: dict[int, str],
        tac_to_pgw: dict[int, str],
        tac_to_smsc: dict[int, str],
    ) -> None:
        self.subscribers = subscribers
        self.cells_by_id = cells_by_id
        self.nes_by_id = nes_by_id
        self.tac_to_msc = tac_to_msc
        self.tac_to_sgw = tac_to_sgw
        self.tac_to_pgw = tac_to_pgw
        self.tac_to_smsc = tac_to_smsc


def build_assets(
    cells: list,
    network_elements: list,
    subscribers: list[Subscriber],
) -> Assets:
    """Build lookup structures from raw asset lists."""
    cells_by_id = {c.cell_id: c for c in cells}
    nes_by_id = {ne.id: ne for ne in network_elements}

    tac_to_msc: dict[int, str] = {}
    tac_to_sgw: dict[int, str] = {}
    tac_to_pgw: dict[int, str] = {}
    tac_to_smsc: dict[int, str] = {}

    for ne in network_elements:
        target_map: dict[int, str] | None = None
        if ne.ne_type == "msc":
            target_map = tac_to_msc
        elif ne.ne_type == "sgw":
            target_map = tac_to_sgw
        elif ne.ne_type == "pgw":
            target_map = tac_to_pgw
        elif ne.ne_type == "smsc":
            target_map = tac_to_smsc

        if target_map is not None:
            for tac in ne.serves_tacs:
                target_map.setdefault(tac, ne.id)

    return Assets(
        subscribers=subscribers,
        cells_by_id=cells_by_id,
        nes_by_id=nes_by_id,
        tac_to_msc=tac_to_msc,
        tac_to_sgw=tac_to_sgw,
        tac_to_pgw=tac_to_pgw,
        tac_to_smsc=tac_to_smsc,
    )


class TimeStepEngine:
    """Iterates over time steps within the configured time range.

    Each iteration yields a ``(timestamp, subscribers)`` tuple where
    *timestamp* is the beginning of the step and *subscribers* is the
    list of subscribers active during that step.
    """

    def __init__(self, config: CDRGeneratorConfig, assets: Assets) -> None:
        self._start = config.meta.time_range.start.replace(tzinfo=timezone.utc)
        self._end = config.meta.time_range.end.replace(tzinfo=timezone.utc)
        self._step = timedelta(seconds=config.meta.time_step_seconds)
        self._subscribers = assets.subscribers

    def iter_steps(self) -> Iterator[tuple[datetime, list[Subscriber]]]:
        """Yield ``(timestamp, active_subscribers)`` for each time step.

        In Phase 2 all subscribers are active at every step.
        """
        current = self._start
        while current <= self._end:
            yield current, self._subscribers
            current += self._step
