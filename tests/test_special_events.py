"""Tests for the special event evaluation engine.

Tests SpecialEventEngine and ActiveEffects:
  - Time-ranged events (start <= timestamp < end)
  - Recurring events (day-of-week + hour match)
  - Compound effects when multiple events overlap
  - ActiveEffects dataclass methods (is_cell_disabled, has_overflow, pick_overflow_cell, is_active)
  - NO_EFFECTS singleton
  - Edge cases: no events, outside all ranges, midnight boundary
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from cdr_generator.config.models import (
    Recurrence,
    SpecialEventConfig,
    SpecialEventEffect,
    TimeRange,
)
from cdr_generator.engine.special_events import (
    NO_EFFECTS,
    ActiveEffects,
    SpecialEventEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _utc(year=2025, month=1, day=1, hour=0, minute=0, second=0) -> datetime:
    """Helper to build a UTC datetime."""
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


@pytest.fixture
def new_year_event() -> SpecialEventConfig:
    """New Year midnight spike: 2025-01-01 00:00 to 00:30."""
    return SpecialEventConfig(
        name="new_year_midnight",
        time_range=TimeRange(
            start=_utc(2025, 1, 1, 0, 0, 0),
            end=_utc(2025, 1, 1, 0, 30, 0),
        ),
        effect=SpecialEventEffect(
            sms_rate_multiplier=20.0,
            voice_rate_multiplier=5.0,
            voice_failure_rate_override=0.50,
            congestion_cells=[20011, 20012],
        ),
    )


@pytest.fixture
def lunch_event() -> SpecialEventConfig:
    """Working day lunch: Mon-Fri, hours 12-13."""
    return SpecialEventConfig(
        name="working_day_lunch",
        recurrence=Recurrence(
            days=["mon", "tue", "wed", "thu", "fri"],
            hours=[12, 13],
        ),
        effect=SpecialEventEffect(
            data_rate_multiplier=1.5,
            voice_rate_multiplier=0.7,
        ),
    )


@pytest.fixture
def cell_outage_event() -> SpecialEventConfig:
    """Cell outage: 2025-01-15 03:00 to 06:00."""
    return SpecialEventConfig(
        name="cell_outage",
        time_range=TimeRange(
            start=_utc(2025, 1, 15, 3, 0, 0),
            end=_utc(2025, 1, 15, 6, 0, 0),
        ),
        effect=SpecialEventEffect(
            disabled_cells=[20021, 20022],
            overflow_cells=[20011, 20012],
            voice_failure_rate_override=0.30,
        ),
    )


# ---------------------------------------------------------------------------
# ActiveEffects dataclass tests
# ---------------------------------------------------------------------------


class TestActiveEffects:
    """Tests for the ActiveEffects frozen dataclass."""

    def test_default_values(self):
        effects = ActiveEffects()
        assert effects.voice_rate_multiplier == 1.0
        assert effects.sms_rate_multiplier == 1.0
        assert effects.data_rate_multiplier == 1.0
        assert effects.voice_failure_rate_override is None
        assert effects.disabled_cells == frozenset()
        assert effects.overflow_cells == frozenset()
        assert effects.congestion_cells == frozenset()
        assert effects.active_event_names == ()

    def test_is_cell_disabled_true(self):
        effects = ActiveEffects(disabled_cells=frozenset({100, 200, 300}))
        assert effects.is_cell_disabled(100) is True
        assert effects.is_cell_disabled(200) is True

    def test_is_cell_disabled_false(self):
        effects = ActiveEffects(disabled_cells=frozenset({100, 200}))
        assert effects.is_cell_disabled(999) is False

    def test_is_cell_disabled_empty(self):
        effects = ActiveEffects()
        assert effects.is_cell_disabled(100) is False

    def test_has_overflow_true(self):
        effects = ActiveEffects(overflow_cells=frozenset({100}))
        assert effects.has_overflow() is True

    def test_has_overflow_false(self):
        effects = ActiveEffects(overflow_cells=frozenset())
        assert effects.has_overflow() is False

    def test_pick_overflow_cell_returns_cell(self, rng):
        cells = frozenset({100, 200, 300})
        effects = ActiveEffects(overflow_cells=cells)
        cell = effects.pick_overflow_cell(rng)
        assert cell in cells

    def test_pick_overflow_cell_no_overflow(self, rng):
        effects = ActiveEffects(overflow_cells=frozenset())
        cell = effects.pick_overflow_cell(rng)
        assert cell is None

    def test_pick_overflow_cell_single(self, rng):
        effects = ActiveEffects(overflow_cells=frozenset({42}))
        cell = effects.pick_overflow_cell(rng)
        assert cell == 42

    def test_pick_overflow_cell_deterministic(self):
        cells = frozenset({10, 20, 30, 40, 50})
        effects = ActiveEffects(overflow_cells=cells)

        rng1 = np.random.default_rng(123)
        cell1 = effects.pick_overflow_cell(rng1)

        rng2 = np.random.default_rng(123)
        cell2 = effects.pick_overflow_cell(rng2)

        assert cell1 == cell2

    def test_is_active_true(self):
        effects = ActiveEffects(active_event_names=("event_a",))
        assert effects.is_active is True

    def test_is_active_false(self):
        effects = ActiveEffects(active_event_names=())
        assert effects.is_active is False

    def test_is_active_multiple_events(self):
        effects = ActiveEffects(active_event_names=("a", "b", "c"))
        assert effects.is_active is True

    def test_frozen_immutable(self):
        effects = ActiveEffects()
        with pytest.raises(AttributeError):
            effects.voice_rate_multiplier = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NO_EFFECTS singleton tests
# ---------------------------------------------------------------------------


class TestNoEffects:
    def test_no_effects_defaults(self):
        assert NO_EFFECTS.voice_rate_multiplier == 1.0
        assert NO_EFFECTS.sms_rate_multiplier == 1.0
        assert NO_EFFECTS.data_rate_multiplier == 1.0
        assert NO_EFFECTS.voice_failure_rate_override is None
        assert NO_EFFECTS.disabled_cells == frozenset()
        assert NO_EFFECTS.overflow_cells == frozenset()
        assert NO_EFFECTS.congestion_cells == frozenset()
        assert NO_EFFECTS.active_event_names == ()

    def test_no_effects_not_active(self):
        assert NO_EFFECTS.is_active is False

    def test_no_effects_no_disabled_cells(self):
        assert NO_EFFECTS.is_cell_disabled(20011) is False

    def test_no_effects_no_overflow(self, rng):
        assert NO_EFFECTS.has_overflow() is False
        assert NO_EFFECTS.pick_overflow_cell(rng) is None


# ---------------------------------------------------------------------------
# SpecialEventEngine — time-ranged events
# ---------------------------------------------------------------------------


class TestTimeRangedEvents:
    """Tests for time-ranged events: start <= timestamp < end."""

    def test_within_range(self, new_year_event):
        engine = SpecialEventEngine([new_year_event])
        # 2025-01-01 00:15 is within [00:00, 00:30)
        ts = _utc(2025, 1, 1, 0, 15, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True
        assert "new_year_midnight" in effects.active_event_names
        assert effects.voice_rate_multiplier == 5.0
        assert effects.sms_rate_multiplier == 20.0

    def test_at_start_boundary(self, new_year_event):
        engine = SpecialEventEngine([new_year_event])
        # Exactly at start: should be active
        ts = _utc(2025, 1, 1, 0, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True

    def test_at_end_boundary(self, new_year_event):
        engine = SpecialEventEngine([new_year_event])
        # Exactly at end: should NOT be active (half-open interval)
        ts = _utc(2025, 1, 1, 0, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is False

    def test_before_range(self, new_year_event):
        engine = SpecialEventEngine([new_year_event])
        ts = _utc(2024, 12, 31, 23, 59, 59)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is False

    def test_after_range(self, new_year_event):
        engine = SpecialEventEngine([new_year_event])
        ts = _utc(2025, 1, 1, 1, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is False

    def test_voice_failure_rate_override(self, new_year_event):
        engine = SpecialEventEngine([new_year_event])
        ts = _utc(2025, 1, 1, 0, 15, 0)
        effects = engine.get_active_effects(ts)
        assert effects.voice_failure_rate_override == 0.50

    def test_congestion_cells(self, new_year_event):
        engine = SpecialEventEngine([new_year_event])
        ts = _utc(2025, 1, 1, 0, 15, 0)
        effects = engine.get_active_effects(ts)
        assert 20011 in effects.congestion_cells
        assert 20012 in effects.congestion_cells

    def test_cell_outage_disabled_cells(self, cell_outage_event):
        engine = SpecialEventEngine([cell_outage_event])
        ts = _utc(2025, 1, 15, 4, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_cell_disabled(20021) is True
        assert effects.is_cell_disabled(20022) is True
        assert effects.is_cell_disabled(20011) is False

    def test_cell_outage_overflow_cells(self, cell_outage_event, rng):
        engine = SpecialEventEngine([cell_outage_event])
        ts = _utc(2025, 1, 15, 4, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.has_overflow() is True
        cell = effects.pick_overflow_cell(rng)
        assert cell in {20011, 20012}


# ---------------------------------------------------------------------------
# SpecialEventEngine — recurring events
# ---------------------------------------------------------------------------


class TestRecurringEvents:
    """Tests for recurring events: day-of-week + hour match."""

    def test_matching_day_and_hour(self, lunch_event):
        engine = SpecialEventEngine([lunch_event])
        # 2025-01-06 is a Monday, hour 12
        ts = _utc(2025, 1, 6, 12, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True
        assert "working_day_lunch" in effects.active_event_names

    def test_matching_day_different_hour(self, lunch_event):
        engine = SpecialEventEngine([lunch_event])
        # 2025-01-06 Monday, hour 10 (not 12 or 13)
        ts = _utc(2025, 1, 6, 10, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is False

    def test_matching_hour_wrong_day(self, lunch_event):
        engine = SpecialEventEngine([lunch_event])
        # 2025-01-04 is a Saturday, hour 12
        ts = _utc(2025, 1, 4, 12, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is False

    def test_all_weekdays(self, lunch_event):
        """Lunch event active on all weekdays (Mon-Fri) at hour 12."""
        engine = SpecialEventEngine([lunch_event])
        # 2025-01-06 Mon, 07 Tue, 08 Wed, 09 Thu, 10 Fri
        for day in [6, 7, 8, 9, 10]:
            ts = _utc(2025, 1, day, 12, 15, 0)
            effects = engine.get_active_effects(ts)
            assert effects.is_active is True, f"Day={day} should be active"

    def test_weekend_inactive(self, lunch_event):
        """Lunch event NOT active on Sat/Sun."""
        engine = SpecialEventEngine([lunch_event])
        # 2025-01-04 Sat, 2025-01-05 Sun
        for day in [4, 5]:
            ts = _utc(2025, 1, day, 12, 15, 0)
            effects = engine.get_active_effects(ts)
            assert effects.is_active is False, f"Day={day} should be inactive"

    def test_hour_13_also_active(self, lunch_event):
        engine = SpecialEventEngine([lunch_event])
        # 2025-01-06 Monday, hour 13
        ts = _utc(2025, 1, 6, 13, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True

    def test_hour_14_inactive(self, lunch_event):
        engine = SpecialEventEngine([lunch_event])
        # 2025-01-06 Monday, hour 14 (outside [12, 13])
        ts = _utc(2025, 1, 6, 14, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is False

    def test_rate_multipliers(self, lunch_event):
        engine = SpecialEventEngine([lunch_event])
        ts = _utc(2025, 1, 6, 12, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.data_rate_multiplier == 1.5
        assert effects.voice_rate_multiplier == 0.7
        # SMS not specified, should default to 1.0
        assert effects.sms_rate_multiplier == 1.0


# ---------------------------------------------------------------------------
# Compound effects — multiple overlapping events
# ---------------------------------------------------------------------------


class TestCompoundEffects:
    """Tests for compound effects when multiple events overlap."""

    def test_rate_multipliers_multiplied(self):
        """Rate multipliers from multiple events are multiplied together."""
        event_a = SpecialEventConfig(
            name="event_a",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(
                voice_rate_multiplier=2.0,
                sms_rate_multiplier=3.0,
            ),
        )
        event_b = SpecialEventConfig(
            name="event_b",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(
                voice_rate_multiplier=1.5,
                sms_rate_multiplier=2.0,
            ),
        )
        engine = SpecialEventEngine([event_a, event_b])
        ts = _utc(2025, 1, 1, 0, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.voice_rate_multiplier == pytest.approx(3.0)  # 2.0 * 1.5
        assert effects.sms_rate_multiplier == pytest.approx(6.0)  # 3.0 * 2.0

    def test_voice_failure_rate_override_maximum(self):
        """voice_failure_rate_override: maximum is taken across events."""
        event_a = SpecialEventConfig(
            name="event_a",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(voice_failure_rate_override=0.30),
        )
        event_b = SpecialEventConfig(
            name="event_b",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(voice_failure_rate_override=0.50),
        )
        engine = SpecialEventEngine([event_a, event_b])
        ts = _utc(2025, 1, 1, 0, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.voice_failure_rate_override == 0.50

    def test_cell_sets_union(self):
        """Cell sets (disabled, overflow, congestion) are unioned across events."""
        event_a = SpecialEventConfig(
            name="event_a",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(
                disabled_cells=[100, 200],
                overflow_cells=[300],
                congestion_cells=[400],
            ),
        )
        event_b = SpecialEventConfig(
            name="event_b",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(
                disabled_cells=[200, 500],
                overflow_cells=[600],
                congestion_cells=[400, 700],
            ),
        )
        engine = SpecialEventEngine([event_a, event_b])
        ts = _utc(2025, 1, 1, 0, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.disabled_cells == frozenset({100, 200, 500})
        assert effects.overflow_cells == frozenset({300, 600})
        assert effects.congestion_cells == frozenset({400, 700})

    def test_active_event_names_collected(self):
        """All active event names should appear in active_event_names."""
        event_a = SpecialEventConfig(
            name="alpha",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(),
        )
        event_b = SpecialEventConfig(
            name="beta",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(),
        )
        engine = SpecialEventEngine([event_a, event_b])
        ts = _utc(2025, 1, 1, 0, 30, 0)
        effects = engine.get_active_effects(ts)
        assert "alpha" in effects.active_event_names
        assert "beta" in effects.active_event_names

    def test_one_active_one_inactive(self, new_year_event, lunch_event):
        """When only one event is active, only its effects apply."""
        engine = SpecialEventEngine([new_year_event, lunch_event])
        # New year at midnight (active), lunch not active at hour 0
        ts = _utc(2025, 1, 1, 0, 15, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True
        assert "new_year_midnight" in effects.active_event_names
        assert "working_day_lunch" not in effects.active_event_names
        assert effects.voice_rate_multiplier == 5.0  # only new_year

    def test_none_failure_override_ignored(self):
        """If one event has None override and another has 0.3, result should be 0.3."""
        event_a = SpecialEventConfig(
            name="a",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(voice_failure_rate_override=None),
        )
        event_b = SpecialEventConfig(
            name="b",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(voice_failure_rate_override=0.30),
        )
        engine = SpecialEventEngine([event_a, event_b])
        ts = _utc(2025, 1, 1, 0, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.voice_failure_rate_override == 0.30

    def test_default_multiplier_when_not_specified(self):
        """Events without explicit multipliers should contribute 1.0 (neutral)."""
        event = SpecialEventConfig(
            name="minimal",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 0, 0, 0),
                end=_utc(2025, 1, 1, 1, 0, 0),
            ),
            effect=SpecialEventEffect(),
        )
        engine = SpecialEventEngine([event])
        ts = _utc(2025, 1, 1, 0, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.voice_rate_multiplier == 1.0
        assert effects.sms_rate_multiplier == 1.0
        assert effects.data_rate_multiplier == 1.0

    def test_three_overlapping_events_multiplied(self):
        """Three events overlapping: multipliers should all multiply."""
        events = []
        for i, mult in enumerate([2.0, 3.0, 1.5]):
            events.append(
                SpecialEventConfig(
                    name=f"event_{i}",
                    time_range=TimeRange(
                        start=_utc(2025, 1, 1, 0, 0, 0),
                        end=_utc(2025, 1, 1, 1, 0, 0),
                    ),
                    effect=SpecialEventEffect(voice_rate_multiplier=mult),
                )
            )
        engine = SpecialEventEngine(events)
        ts = _utc(2025, 1, 1, 0, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.voice_rate_multiplier == pytest.approx(9.0)  # 2.0 * 3.0 * 1.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for SpecialEventEngine."""

    def test_no_events_configured(self):
        engine = SpecialEventEngine([])
        ts = _utc(2025, 1, 1, 12, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is False
        assert effects.voice_rate_multiplier == 1.0
        assert effects.sms_rate_multiplier == 1.0
        assert effects.data_rate_multiplier == 1.0

    def test_timestamp_outside_all_ranges(self, new_year_event, cell_outage_event):
        engine = SpecialEventEngine([new_year_event, cell_outage_event])
        # 2025-06-15 is far from both events
        ts = _utc(2025, 6, 15, 12, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is False

    def test_midnight_boundary_crossing(self):
        """Event that spans midnight: check both sides."""
        event = SpecialEventConfig(
            name="midnight_span",
            time_range=TimeRange(
                start=_utc(2025, 1, 1, 23, 30, 0),
                end=_utc(2025, 1, 2, 0, 30, 0),
            ),
            effect=SpecialEventEffect(voice_rate_multiplier=3.0),
        )
        engine = SpecialEventEngine([event])
        # Before midnight
        ts_before = _utc(2025, 1, 1, 23, 45, 0)
        assert engine.get_active_effects(ts_before).is_active is True
        # After midnight
        ts_after = _utc(2025, 1, 2, 0, 15, 0)
        assert engine.get_active_effects(ts_after).is_active is True
        # After end
        ts_past = _utc(2025, 1, 2, 0, 30, 0)
        assert engine.get_active_effects(ts_past).is_active is False

    def test_recurrence_hour_boundary(self):
        """Recurrence at exact hour boundary (start of hour)."""
        event = SpecialEventConfig(
            name="exact_hour",
            recurrence=Recurrence(days=["mon"], hours=[12]),
            effect=SpecialEventEffect(data_rate_multiplier=2.0),
        )
        engine = SpecialEventEngine([event])
        # 2025-01-06 is Monday, 12:00:00 exactly
        ts = _utc(2025, 1, 6, 12, 0, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True

    def test_recurrence_last_second_of_hour(self):
        """Recurrence still active at 12:59:59."""
        event = SpecialEventConfig(
            name="last_second",
            recurrence=Recurrence(days=["mon"], hours=[12]),
            effect=SpecialEventEffect(data_rate_multiplier=2.0),
        )
        engine = SpecialEventEngine([event])
        ts = _utc(2025, 1, 6, 12, 59, 59)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True

    def test_mixed_time_range_and_recurrence_overlap(self, new_year_event):
        """A time-ranged event and a recurring event both active at same time."""
        # 2025-01-01 is a Wednesday. Create a recurring event for Wed hour 0.
        recurring = SpecialEventConfig(
            name="wed_midnight",
            recurrence=Recurrence(days=["wed"], hours=[0]),
            effect=SpecialEventEffect(
                voice_rate_multiplier=2.0,
                data_rate_multiplier=1.5,
            ),
        )
        engine = SpecialEventEngine([new_year_event, recurring])
        # 2025-01-01 00:15 - both should be active
        ts = _utc(2025, 1, 1, 0, 15, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True
        assert len(effects.active_event_names) == 2
        # Voice: 5.0 (new_year) * 2.0 (recurring) = 10.0
        assert effects.voice_rate_multiplier == pytest.approx(10.0)
        # Data: 1.0 (new_year, not specified) * 1.5 (recurring) = 1.5
        assert effects.data_rate_multiplier == pytest.approx(1.5)

    def test_event_with_only_recurrence_no_time_range(self):
        """Event with recurrence but no time_range."""
        event = SpecialEventConfig(
            name="recurring_only",
            recurrence=Recurrence(days=["tue"], hours=[9, 10]),
            effect=SpecialEventEffect(sms_rate_multiplier=1.2),
        )
        engine = SpecialEventEngine([event])
        # 2025-01-07 is Tuesday, hour 9
        ts = _utc(2025, 1, 7, 9, 30, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True

    def test_event_with_only_time_range_no_recurrence(self, new_year_event):
        """Event with time_range but no recurrence."""
        engine = SpecialEventEngine([new_year_event])
        ts = _utc(2025, 1, 1, 0, 10, 0)
        effects = engine.get_active_effects(ts)
        assert effects.is_active is True

    def test_engine_initialization_stores_events(
        self, new_year_event, lunch_event, cell_outage_event
    ):
        """Engine can be initialized with multiple events."""
        engine = SpecialEventEngine([new_year_event, lunch_event, cell_outage_event])
        # Just verify it does not raise
        assert engine is not None
