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
    sub_by_imsi: dict[str, Subscriber] | None = None,
    a_party_contacts: list[str] | None = None,
    ext_ratio: float | None = None,
    contact_threshold: float | None = None,
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
    sub_by_imsi:
        Pre-built IMSI-to-subscriber lookup.  When provided, avoids
        rebuilding the dict on every call (significant when called
        thousands of times in the main generation loop).
    a_party_contacts:
        Pre-fetched contact list for ``a_party``.  When provided,
        avoids the ``contact_book.get()`` dict lookup per call.
    ext_ratio:
        Pre-extracted ``external_call_ratio`` value.  When provided
        together with ``contact_threshold``, avoids two ``config.get()``
        calls per invocation.
    contact_threshold:
        Pre-computed ``ext_ratio + (1 - ext_ratio) * repeat_prob``.

    Returns
    -------
    BPartyResult
        The selected B-party.
    """
    if ext_ratio is None:
        ext_ratio = config.get("external_call_ratio", 0.15)
        repeat_prob = config.get("repeat_call_probability", 0.6)
        contact_threshold = ext_ratio + (1.0 - ext_ratio) * repeat_prob

    if sub_by_imsi is None:
        sub_by_imsi = {s.imsi: s for s in subscribers}

    roll = float(rng.random())

    # Tier 1: external number
    if roll < ext_ratio and external_numbers:
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
    if roll < contact_threshold:  # type: ignore[operator]
        contacts = (
            a_party_contacts
            if a_party_contacts is not None
            else contact_book.get(a_party.imsi, [])
        )
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
