"""Concurrency enforcement for per-subscriber event counts.

Caps the number of voice, SMS, and data events that a single subscriber
can generate in one time step, preventing unrealistic bursts.
"""

from __future__ import annotations

from cdr_generator.config.models import ConcurrencyConfig


def enforce_concurrency(
    n_voice: int,
    n_sms: int,
    n_data: int,
    config: ConcurrencyConfig,
) -> tuple[int, int, int]:
    """Cap event counts per subscriber per time step.

    Parameters
    ----------
    n_voice:
        Sampled voice event count.
    n_sms:
        Sampled SMS event count.
    n_data:
        Sampled data session count.
    config:
        Concurrency limits. ``None`` for a field means unlimited.

    Returns
    -------
    tuple[int, int, int]
        Capped (n_voice, n_sms, n_data).
    """
    if config.max_voice is not None:
        n_voice = min(n_voice, config.max_voice)
    if config.max_data is not None:
        n_data = min(n_data, config.max_data)
    if config.max_sms is not None:
        n_sms = min(n_sms, config.max_sms)
    return n_voice, n_sms, n_data
