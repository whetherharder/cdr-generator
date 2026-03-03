"""Effective rate calculation for event generation.

The effective rate for a given (hour, day_of_week, time_step) is the expected
number of events **per time step**, computed as:

    hourly_rate = base_lambda * (weight[h] / sum(weights)) * dow_mult[dow]
    step_rate   = hourly_rate * (time_step_seconds / 3600)

This distributes the daily total across hours according to the weight
profile, scales by the day-of-week multiplier, then converts to per-step
rate suitable for Poisson sampling.
"""

from __future__ import annotations


def effective_rate(
    base_lambda: float,
    hourly_weights: list[float],
    dow_multipliers: list[float],
    hour: int,
    dow: int,
    time_step_seconds: int = 60,
    weight_sum: float = 0.0,
) -> float:
    """Compute the effective Poisson rate for a single time step.

    Parameters
    ----------
    base_lambda:
        Daily average event count (from profile daily_rates).
    hourly_weights:
        24-element array of relative weights (index 0 = midnight).
        Normalized internally -- does not need to sum to 1.
    dow_multipliers:
        7-element array (Mon=0 ... Sun=6) scaling the daily rate.
    hour:
        Hour of day (0-23).
    dow:
        Day of week (0=Monday ... 6=Sunday).
    time_step_seconds:
        Duration of a single time step in seconds (default 60).
    weight_sum:
        Pre-computed sum of *hourly_weights*.  When positive the function
        skips calling ``sum()`` internally, saving ~40% of its runtime.
        Pass ``0.0`` (the default) to compute on the fly.

    Returns
    -------
    float
        The effective rate (expected events per time step) for this slot.
    """
    if weight_sum <= 0.0:
        weight_sum = sum(hourly_weights)
    if weight_sum <= 0.0 or base_lambda <= 0.0:
        return 0.0

    # hourly_weights[h] / weight_sum = fraction of daily events in hour h
    # base_lambda * fraction = expected events in that hour
    # dow_mult scales the whole day
    hourly_fraction = hourly_weights[hour] / weight_sum
    dow_mult = dow_multipliers[dow]
    hourly_rate = base_lambda * hourly_fraction * dow_mult
    return hourly_rate * time_step_seconds / 3600.0
