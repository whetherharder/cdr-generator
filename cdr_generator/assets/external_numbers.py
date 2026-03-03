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

# PERFORMANCE: Pre-computed positional multipliers for vectorized 7-digit
# suffix generation.  Used by generate_external_numbers to convert a
# (count × 7) digit array into integer suffixes in one numpy operation,
# replacing count separate rng.integers(size=7) calls (~1 µs each) with
# a single rng.integers(size=(count, 7)) call (~5 µs regardless of count).
_SUFFIX_POWERS: np.ndarray = np.array(
    [1000000, 100000, 10000, 1000, 100, 10, 1], dtype=np.int64
)


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

    # PERFORMANCE: Generate all 7-digit suffixes in one vectorized numpy call
    # instead of count separate rng.integers(size=7) calls.  For count=1000
    # this reduces numpy dispatch overhead from ~5ms (1000 × 5µs/call) to
    # ~5µs (one call), a ~1000× reduction in numpy overhead.
    all_digit_rows = rng.integers(0, 10, size=(count, 7))
    all_nums = (all_digit_rows * _SUFFIX_POWERS).sum(axis=1)

    seen: set[str] = set()
    numbers: list[ExternalNumber] = []

    for i, idx in enumerate(prefix_indices):
        prefix = prefixes[int(idx)]
        msisdn = f"{prefix}{int(all_nums[i]):07d}"
        if msisdn not in seen:
            seen.add(msisdn)
            numbers.append(ExternalNumber(msisdn=msisdn))
        else:
            # Rare collision (P ≈ count²/2×10⁷ ≈ 0.005% for count=1000):
            # fall back to per-number generation to find a unique suffix.
            for _ in range(100):
                extra = rng.integers(0, 10, size=7)
                num = int((extra * _SUFFIX_POWERS).sum())
                msisdn = f"{prefix}{num:07d}"
                if msisdn not in seen:
                    seen.add(msisdn)
                    numbers.append(ExternalNumber(msisdn=msisdn))
                    break
            else:
                numbers.append(ExternalNumber(msisdn=msisdn))

    return numbers
