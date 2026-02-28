"""Tests for engine/mobility.py -- Subscriber position and movement.

Phase 3 acceptance criteria:
- Position: home at night, work during day, commute at configured hours
- Handover: first_cell != last_cell in ~5% of calls (per profile)
- Roaming: ~1% for office_worker, ~15% for heavy_traveler
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from cdr_generator.assets.models import Cell, Subscriber
from cdr_generator.config.models import MobilityConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def home_cell() -> Cell:
    return Cell(
        cell_id=20011,
        tac=2001,
        ecgi="25001-20011",
        lat=55.7558,
        lon=37.6173,
        azimuth=0,
        sector=1,
        cell_type="urban",
        capacity="high",
        neighbors=[20012, 20013],
    )


@pytest.fixture
def work_cell() -> Cell:
    return Cell(
        cell_id=20021,
        tac=2002,
        ecgi="25001-20021",
        lat=55.7700,
        lon=37.6400,
        azimuth=120,
        sector=1,
        cell_type="urban",
        capacity="high",
        neighbors=[20022, 20023],
    )


@pytest.fixture
def cells() -> list[Cell]:
    """A set of cells for mobility tests."""
    return [
        Cell(
            cell_id=20011,
            tac=2001,
            ecgi="25001-20011",
            lat=55.7558,
            lon=37.6173,
            azimuth=0,
            sector=1,
            cell_type="urban",
            capacity="high",
            neighbors=[20012, 20013],
        ),
        Cell(
            cell_id=20012,
            tac=2001,
            ecgi="25001-20012",
            lat=55.7560,
            lon=37.6200,
            azimuth=120,
            sector=2,
            cell_type="urban",
            capacity="medium",
            neighbors=[20011, 20013],
        ),
        Cell(
            cell_id=20013,
            tac=2001,
            ecgi="25001-20013",
            lat=55.7555,
            lon=37.6150,
            azimuth=240,
            sector=3,
            cell_type="urban",
            capacity="medium",
            neighbors=[20011, 20012],
        ),
        Cell(
            cell_id=20021,
            tac=2002,
            ecgi="25001-20021",
            lat=55.7700,
            lon=37.6400,
            azimuth=0,
            sector=1,
            cell_type="urban",
            capacity="high",
            neighbors=[20022, 20023],
        ),
        Cell(
            cell_id=20022,
            tac=2002,
            ecgi="25001-20022",
            lat=55.7705,
            lon=37.6420,
            azimuth=120,
            sector=2,
            cell_type="suburban",
            capacity="medium",
            neighbors=[20021, 20023],
        ),
        Cell(
            cell_id=20023,
            tac=2002,
            ecgi="25001-20023",
            lat=55.7695,
            lon=37.6380,
            azimuth=240,
            sector=3,
            cell_type="suburban",
            capacity="low",
            neighbors=[20021, 20022],
        ),
    ]


@pytest.fixture
def cells_by_id(cells) -> dict[int, Cell]:
    return {c.cell_id: c for c in cells}


@pytest.fixture
def office_worker_sub() -> Subscriber:
    return Subscriber(
        imsi="250010000000001",
        msisdn="+79160000001",
        imei="353325080000001",
        profile_name="office_worker",
        home_cell_id=20011,
        work_cell_id=20021,
        serving_ne_id="msc-01",
    )


@pytest.fixture
def mobility_config_office() -> MobilityConfig:
    """Mobility config for office_worker profile."""
    return MobilityConfig(
        home_cell_strategy="random_urban",
        work_cell_strategy="random_urban",
        commute_hours=[8, 9, 17, 18],
        roaming_probability=0.01,
        handover_during_call=0.05,
    )


@pytest.fixture
def mobility_config_traveler() -> MobilityConfig:
    """Mobility config for heavy_traveler profile."""
    return MobilityConfig(
        home_cell_strategy="random_urban",
        work_cell_strategy="random_urban",
        commute_hours=[7, 8, 18, 19],
        roaming_probability=0.15,
        handover_during_call=0.10,
    )


# ===================================================================
# Position at different times of day
# ===================================================================


class TestPositionByTimeOfDay:
    """resolve_position returns home cell at night, work cell during day."""

    def test_night_returns_home_cell(self, office_worker_sub, cells_by_id) -> None:
        from cdr_generator.engine.mobility import resolve_position

        night_time = datetime(2025, 1, 15, 3, 0, 0, tzinfo=timezone.utc)

        no_roaming = MobilityConfig(
            home_cell_strategy="random_urban",
            commute_hours=[8, 9, 17, 18],
            roaming_probability=0.0,
            handover_during_call=0.0,
        )
        rng = np.random.default_rng(42)

        first_cell, last_cell = resolve_position(
            home_cell_id=office_worker_sub.home_cell_id,
            work_cell_id=office_worker_sub.work_cell_id,
            mobility=no_roaming,
            timestamp=night_time,
            cells_by_id=cells_by_id,
            rng=rng,
        )
        assert first_cell == office_worker_sub.home_cell_id, (
            f"At 3 AM, office_worker should be at home cell {office_worker_sub.home_cell_id}, "
            f"got {first_cell}"
        )

    def test_workday_returns_work_cell(
        self, office_worker_sub, cells_by_id, mobility_config_office
    ) -> None:
        from cdr_generator.engine.mobility import resolve_position

        work_time = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        work_cell_count = 0
        n_trials = 100
        for i in range(n_trials):
            rng = np.random.default_rng(i)
            first_cell, _ = resolve_position(
                home_cell_id=office_worker_sub.home_cell_id,
                work_cell_id=office_worker_sub.work_cell_id,
                mobility=mobility_config_office,
                timestamp=work_time,
                cells_by_id=cells_by_id,
                rng=rng,
            )
            if first_cell == office_worker_sub.work_cell_id:
                work_cell_count += 1

        assert work_cell_count > n_trials * 0.90, (
            f"At 2 PM, office_worker should mostly be at work cell, "
            f"got work cell {work_cell_count}/{n_trials} times"
        )

    def test_commute_hour_returns_valid_cell(
        self, office_worker_sub, cells_by_id
    ) -> None:
        """During commute hours, subscriber should be at a valid cell."""
        from cdr_generator.engine.mobility import resolve_position

        commute_time = datetime(2025, 1, 15, 8, 30, 0, tzinfo=timezone.utc)
        valid_cells = set(cells_by_id.keys())

        no_roaming = MobilityConfig(
            home_cell_strategy="random_urban",
            commute_hours=[8, 9, 17, 18],
            roaming_probability=0.0,
            handover_during_call=0.0,
        )
        rng = np.random.default_rng(42)

        first_cell, last_cell = resolve_position(
            home_cell_id=office_worker_sub.home_cell_id,
            work_cell_id=office_worker_sub.work_cell_id,
            mobility=no_roaming,
            timestamp=commute_time,
            cells_by_id=cells_by_id,
            rng=rng,
        )
        assert first_cell in valid_cells, (
            f"During commute, first_cell {first_cell} must be a valid cell"
        )
        assert last_cell in valid_cells, (
            f"During commute, last_cell {last_cell} must be a valid cell"
        )


class TestPositionDeterministic:
    """resolve_position must be deterministic given the same seed."""

    def test_same_inputs_same_output(
        self, office_worker_sub, cells_by_id, mobility_config_office
    ) -> None:
        from cdr_generator.engine.mobility import resolve_position

        t = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        rng1 = np.random.default_rng(77)
        rng2 = np.random.default_rng(77)

        result1 = resolve_position(
            home_cell_id=office_worker_sub.home_cell_id,
            work_cell_id=office_worker_sub.work_cell_id,
            mobility=mobility_config_office,
            timestamp=t,
            cells_by_id=cells_by_id,
            rng=rng1,
        )
        result2 = resolve_position(
            home_cell_id=office_worker_sub.home_cell_id,
            work_cell_id=office_worker_sub.work_cell_id,
            mobility=mobility_config_office,
            timestamp=t,
            cells_by_id=cells_by_id,
            rng=rng2,
        )
        assert result1 == result2


# ===================================================================
# Handover: first_cell != last_cell
# ===================================================================


class TestHandover:
    """~5% of events should have first_cell != last_cell (handover)."""

    def test_handover_occurs_at_configured_rate(
        self, office_worker_sub, cells_by_id, mobility_config_office
    ) -> None:
        from cdr_generator.engine.mobility import resolve_position

        handover_count = 0
        n_trials = 2000
        t = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)

        for i in range(n_trials):
            rng = np.random.default_rng(i)
            first_cell, last_cell = resolve_position(
                home_cell_id=office_worker_sub.home_cell_id,
                work_cell_id=office_worker_sub.work_cell_id,
                mobility=mobility_config_office,
                timestamp=t,
                cells_by_id=cells_by_id,
                rng=rng,
            )
            if first_cell != last_cell:
                handover_count += 1

        handover_rate = handover_count / n_trials
        expected = mobility_config_office.handover_during_call
        assert handover_rate < expected * 3.0, (
            f"Handover rate {handover_rate:.3f} much higher than expected {expected}"
        )

    def test_handover_cell_is_neighbor(self, office_worker_sub, cells_by_id) -> None:
        """When handover occurs, last_cell should be a neighbor of first_cell."""
        from cdr_generator.engine.mobility import resolve_position

        high_handover = MobilityConfig(
            home_cell_strategy="random_urban",
            commute_hours=[8, 9, 17, 18],
            roaming_probability=0.0,
            handover_during_call=0.5,
        )
        t = datetime(2025, 1, 15, 3, 0, 0, tzinfo=timezone.utc)

        handovers_found = 0
        for i in range(200):
            rng = np.random.default_rng(i)
            first_cell, last_cell = resolve_position(
                home_cell_id=office_worker_sub.home_cell_id,
                work_cell_id=office_worker_sub.work_cell_id,
                mobility=high_handover,
                timestamp=t,
                cells_by_id=cells_by_id,
                rng=rng,
            )
            if first_cell != last_cell:
                handovers_found += 1
                first = cells_by_id[first_cell]
                valid_neighbors = [n for n in first.neighbors if n in cells_by_id]
                assert last_cell in valid_neighbors, (
                    f"Handover cell {last_cell} not a neighbor of {first_cell}. "
                    f"Neighbors: {valid_neighbors}"
                )

        assert handovers_found > 0, "Expected some handovers with 50% probability"


# ===================================================================
# Roaming
# ===================================================================


class TestRoaming:
    """Roaming probability differs by profile."""

    def test_office_worker_low_roaming(
        self, office_worker_sub, cells_by_id, mobility_config_office
    ) -> None:
        """Office worker roaming ~1%."""
        from cdr_generator.engine.mobility import resolve_position

        t = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        roaming_count = 0
        n_trials = 5000

        for i in range(n_trials):
            rng = np.random.default_rng(i)
            first_cell, _ = resolve_position(
                home_cell_id=office_worker_sub.home_cell_id,
                work_cell_id=office_worker_sub.work_cell_id,
                mobility=mobility_config_office,
                timestamp=t,
                cells_by_id=cells_by_id,
                rng=rng,
            )
            if first_cell != office_worker_sub.work_cell_id:
                roaming_count += 1

        roaming_rate = roaming_count / n_trials
        expected = mobility_config_office.roaming_probability
        assert roaming_rate < expected * 3.0, (
            f"Office worker roaming rate {roaming_rate:.3f} too high "
            f"(expected ~{expected})"
        )

    def test_heavy_traveler_high_roaming(
        self, office_worker_sub, cells_by_id, mobility_config_traveler
    ) -> None:
        """Heavy traveler roaming ~15%."""
        from cdr_generator.engine.mobility import resolve_position

        t = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        roaming_count = 0
        n_trials = 5000

        for i in range(n_trials):
            rng = np.random.default_rng(i + 10000)
            first_cell, _ = resolve_position(
                home_cell_id=office_worker_sub.home_cell_id,
                work_cell_id=office_worker_sub.work_cell_id,
                mobility=mobility_config_traveler,
                timestamp=t,
                cells_by_id=cells_by_id,
                rng=rng,
            )
            if first_cell != office_worker_sub.work_cell_id:
                roaming_count += 1

        roaming_rate = roaming_count / n_trials
        expected = mobility_config_traveler.roaming_probability
        assert expected * 0.5 < roaming_rate < expected * 1.5, (
            f"Heavy traveler roaming rate {roaming_rate:.3f} not near "
            f"expected {expected}. Range: [{expected * 0.5:.3f}, {expected * 1.5:.3f}]"
        )

    def test_roaming_rate_difference(
        self,
        office_worker_sub,
        cells_by_id,
        mobility_config_office,
        mobility_config_traveler,
    ) -> None:
        """Heavy traveler should roam significantly more than office worker."""
        from cdr_generator.engine.mobility import resolve_position

        t = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        n_trials = 3000

        office_roaming = 0
        for i in range(n_trials):
            rng = np.random.default_rng(i)
            first_cell, _ = resolve_position(
                home_cell_id=office_worker_sub.home_cell_id,
                work_cell_id=office_worker_sub.work_cell_id,
                mobility=mobility_config_office,
                timestamp=t,
                cells_by_id=cells_by_id,
                rng=rng,
            )
            if first_cell != office_worker_sub.work_cell_id:
                office_roaming += 1

        traveler_roaming = 0
        for i in range(n_trials):
            rng = np.random.default_rng(i + 100000)
            first_cell, _ = resolve_position(
                home_cell_id=office_worker_sub.home_cell_id,
                work_cell_id=office_worker_sub.work_cell_id,
                mobility=mobility_config_traveler,
                timestamp=t,
                cells_by_id=cells_by_id,
                rng=rng,
            )
            if first_cell != office_worker_sub.work_cell_id:
                traveler_roaming += 1

        assert traveler_roaming > office_roaming * 3, (
            f"Heavy traveler roaming ({traveler_roaming}) should be much higher "
            f"than office worker ({office_roaming})"
        )


# ===================================================================
# Subscriber without work cell
# ===================================================================


class TestNoWorkCell:
    """Subscribers without work_cell_id (e.g., retiree) stay at home."""

    def test_no_work_cell_stays_home(self, cells_by_id) -> None:
        from cdr_generator.engine.mobility import resolve_position

        retiree = Subscriber(
            imsi="250010000000099",
            msisdn="+79160000099",
            imei="353325080000099",
            profile_name="retiree",
            home_cell_id=20011,
            work_cell_id=None,
            serving_ne_id="msc-01",
        )

        no_roaming = MobilityConfig(
            home_cell_strategy="random_urban",
            commute_hours=[8, 9, 17, 18],
            roaming_probability=0.0,
            handover_during_call=0.0,
        )

        work_time = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        rng = np.random.default_rng(42)

        first_cell, last_cell = resolve_position(
            home_cell_id=retiree.home_cell_id,
            work_cell_id=retiree.work_cell_id,
            mobility=no_roaming,
            timestamp=work_time,
            cells_by_id=cells_by_id,
            rng=rng,
        )
        assert first_cell == retiree.home_cell_id, (
            f"Subscriber without work_cell should be at home cell, got {first_cell}"
        )
