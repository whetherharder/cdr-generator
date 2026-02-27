"""B-party selection for CDR generation.

Selects the B-party (callee/recipient) for each event using a three-tier
strategy:

1. **External number** (~15%): pick from the pre-generated external
   number pool.
2. **Contact book** (~60%): pick from the subscriber's contact graph.
3. **Random subscriber** (~25%): pick a random subscriber from the
   network.

The exact ratios are controlled by ``repeat_call_probability`` and
``external_call_ratio`` in the contact book config.

Usage::

    result = select_b_party(a_party, contact_book, subscribers, ext_numbers, config, rng)
    callee_msisdn = result.msisdn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cdr_generator.assets.external_numbers import ExternalNumber
    from cdr_generator.assets.models import Subscriber


@dataclass(frozen=True)
class BPartyResult:
    """Result of B-party selection."""

    msisdn: str
    imsi: str
    is_external: bool
    source: str
    subscriber: Subscriber | None = None


def select_b_party(
    a_party: Subscriber,
    contact_book: dict[str, list[str]],
    subscribers: list[Subscriber],
    external_numbers: list[ExternalNumber],
    config: dict,
    rng: np.random.Generator,
) -> BPartyResult:
    """Select a B-party for the given A-party.

    Decision tree:
    1. With probability ``external_call_ratio``, select an external number.
    2. With probability ``repeat_call_probability`` (of remaining),
       select from the A-party's contact book.
    3. Otherwise, select a random subscriber.

    Parameters
    ----------
    a_party:
        The A-party subscriber (caller/sender).
    contact_book:
        Contact book dict mapping IMSI -> list of contact IMSIs.
    subscribers:
        Full subscriber list.
    external_numbers:
        Pre-generated external number pool (list of ExternalNumber).
    config:
        Contact book config dict with repeat_call_probability and
        external_call_ratio.
    rng:
        Numpy random generator.

    Returns
    -------
    BPartyResult
        The selected B-party.
    """
    external_ratio = config.get("external_call_ratio", 0.15)
    repeat_prob = config.get("repeat_call_probability", 0.6)

    sub_by_imsi = {s.imsi: s for s in subscribers}

    roll = float(rng.random())

    # Tier 1: external number
    if roll < external_ratio and external_numbers:
        idx = int(rng.integers(0, len(external_numbers)))
        ext = external_numbers[idx]
        msisdn = ext.msisdn if hasattr(ext, "msisdn") else str(ext)
        return BPartyResult(
            msisdn=msisdn,
            imsi="external",
            is_external=True,
            source="external",
            subscriber=None,
        )

    # Tier 2: contact book
    contact_threshold = external_ratio + (1.0 - external_ratio) * repeat_prob
    if roll < contact_threshold:
        contacts = contact_book.get(a_party.imsi, [])
        if contacts:
            c_idx = int(rng.integers(0, len(contacts)))
            contact_imsi = contacts[c_idx]
            sub = sub_by_imsi.get(contact_imsi)
            if sub is not None:
                return BPartyResult(
                    msisdn=sub.msisdn,
                    imsi=sub.imsi,
                    is_external=False,
                    source="contact_book",
                    subscriber=sub,
                )

    # Tier 3: random subscriber
    return _select_random_subscriber(a_party, subscribers, rng)


def _select_random_subscriber(
    a_party: Subscriber,
    subscribers: list[Subscriber],
    rng: np.random.Generator,
) -> BPartyResult:
    """Select a random subscriber different from A-party."""
    if len(subscribers) <= 1:
        sub = subscribers[0]
        return BPartyResult(
            msisdn=sub.msisdn,
            imsi=sub.imsi,
            is_external=False,
            source="random",
            subscriber=sub,
        )

    for _ in range(10):
        idx = int(rng.integers(0, len(subscribers)))
        candidate = subscribers[idx]
        if candidate.imsi != a_party.imsi:
            return BPartyResult(
                msisdn=candidate.msisdn,
                imsi=candidate.imsi,
                is_external=False,
                source="random",
                subscriber=candidate,
            )

    # Fallback
    for sub in subscribers:
        if sub.imsi != a_party.imsi:
            return BPartyResult(
                msisdn=sub.msisdn,
                imsi=sub.imsi,
                is_external=False,
                source="random",
                subscriber=sub,
            )

    sub = subscribers[0]
    return BPartyResult(
        msisdn=sub.msisdn,
        imsi=sub.imsi,
        is_external=False,
        source="random",
        subscriber=sub,
    )
