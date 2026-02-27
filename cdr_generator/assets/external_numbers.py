"""External number pool generation.

Generates a fixed-size pool of external phone numbers (MSISDNs) drawn
from weighted prefix groups.  These represent numbers outside the
operator's subscriber base -- landlines, other mobile operators,
international numbers.

Usage::

    numbers = generate_external_numbers(config, rng)
    msisdn = numbers[0].msisdn  # or str(numbers[0])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExternalNumber:
    """A single external phone number."""

    msisdn: str
    imsi: str = "external"

    def __str__(self) -> str:
        return self.msisdn


def generate_external_numbers(
    config: dict,
    rng: np.random.Generator,
) -> list[ExternalNumber]:
    """Generate the external number pool from config.

    Parameters
    ----------
    config:
        External numbers configuration dict with keys: count, prefixes
        (list of {prefix, weight, label}).
    rng:
        Numpy random generator for deterministic generation.

    Returns
    -------
    list[ExternalNumber]
        Pool of external numbers with unique MSISDNs.
    """
    count = config.get("count", 0)
    if count <= 0:
        return []

    prefix_entries = config.get("prefixes", [])
    if not prefix_entries:
        prefix_entries = [{"prefix": "+7495", "weight": 1.0}]

    prefixes = [e["prefix"] for e in prefix_entries]
    weights = [e.get("weight", 1.0) for e in prefix_entries]

    total_w = sum(weights)
    if total_w <= 0:
        probs = np.ones(len(prefixes)) / len(prefixes)
    else:
        probs = np.array(weights) / total_w

    prefix_indices = rng.choice(len(prefixes), size=count, p=probs)

    seen: set[str] = set()
    numbers: list[ExternalNumber] = []

    for idx in prefix_indices:
        prefix = prefixes[idx]
        # Generate unique MSISDN
        for _ in range(100):
            digits = rng.integers(0, 10, size=7)
            suffix = "".join(str(d) for d in digits)
            msisdn = prefix + suffix
            if msisdn not in seen:
                seen.add(msisdn)
                numbers.append(ExternalNumber(msisdn=msisdn))
                break
        else:
            # If we can't generate unique after 100 tries, append anyway
            numbers.append(ExternalNumber(msisdn=msisdn))

    return numbers
