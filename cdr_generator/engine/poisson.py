"""Poisson event count sampling."""

from __future__ import annotations

import numpy as np


def sample_count(rate: float, rng: np.random.Generator) -> int:
    """Sample the number of events from a Poisson distribution.

    Parameters
    ----------
    rate:
        The expected number of events (lambda). Must be >= 0.
    rng:
        A numpy random Generator for deterministic sampling.

    Returns
    -------
    int
        Number of events (non-negative).
    """
    if rate <= 0.0:
        return 0
    return int(rng.poisson(rate))
