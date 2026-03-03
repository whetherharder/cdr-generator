"""Zipf-distributed contact graph generation.

Generates an asymmetric contact book where each subscriber has a set of
contacts drawn from the subscriber pool. The number of contacts per
subscriber follows a Zipf distribution, and same-profile subscribers are
biased towards each other by ``intra_profile_bias``.

Usage::

    book = build_contact_book(subscribers, contact_book_config, rng)
    contacts = book["250010000001"]  # list of contact IMSIs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cdr_generator.assets.models import Subscriber


def build_contact_book(
    subscribers: list[Subscriber],
    config: dict,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Generate the full contact book for all subscribers.

    Parameters
    ----------
    subscribers:
        Full list of generated subscribers.
    config:
        Contact book configuration dict with keys: degree_distribution,
        asymmetric, intra_profile_bias.
    rng:
        Numpy random generator for deterministic generation.

    Returns
    -------
    dict[str, list[str]]
        Mapping from subscriber IMSI to a list of contact IMSIs.
    """
    if len(subscribers) < 2:
        return {s.imsi: [] for s in subscribers}

    dist_params = config.get("degree_distribution", {}).get("params", {})
    zipf_a = dist_params.get("a", 2.0)
    min_contacts = dist_params.get("min", 3)
    max_contacts = dist_params.get("max", 100)
    intra_bias = config.get("intra_profile_bias", 1.0)
    asymmetric = config.get("asymmetric", True)

    # Effective max cannot exceed (n_subscribers - 1)
    effective_max = min(max_contacts, len(subscribers) - 1)
    effective_min = min(min_contacts, effective_max)

    all_imsis = [s.imsi for s in subscribers]
    profile_by_imsi = {s.imsi: s.profile_name for s in subscribers}

    # PERFORMANCE: Pre-compute integer profile IDs so the weighted-sampling
    # inner loop can compare ints instead of calling profile_by_imsi.get(c)
    # per candidate (eliminates N×(N-1) dict.get calls, ~9900 for N=100).
    _unique_profiles = sorted(set(profile_by_imsi.values()))
    _profile_to_int: dict[str, int] = {p: i for i, p in enumerate(_unique_profiles)}
    _n_profiles = len(_unique_profiles)
    _sub_prof_ints: list[int] = [
        _profile_to_int[profile_by_imsi[imsi]] for imsi in all_imsis
    ]

    graph: dict[str, list[str]] = {}

    for _sub_pos, sub in enumerate(subscribers):
        degree = _sample_zipf_degree(zipf_a, effective_min, effective_max, rng)
        degree = min(degree, len(subscribers) - 1)
        if degree <= 0:
            graph[sub.imsi] = []
            continue

        candidates: list[str] = [i for i in all_imsis if i != sub.imsi]
        if not candidates:
            graph[sub.imsi] = []
            continue

        degree = min(degree, len(candidates))

        if intra_bias == 1.0 or _n_profiles <= 1:
            # Uniform sampling without replacement
            indices = rng.choice(len(candidates), size=degree, replace=False)
            contacts = [candidates[i] for i in indices]
        else:
            # PERFORMANCE: integer comparison instead of dict.get per candidate
            sub_prof_int = _sub_prof_ints[_sub_pos]
            weights = [
                intra_bias if _sub_prof_ints[j] == sub_prof_int else 1.0
                for j in range(len(all_imsis))
                if j != _sub_pos
            ]
            contacts = _weighted_sample_without_replacement(
                candidates, weights, degree, rng
            )

        graph[sub.imsi] = contacts

    if not asymmetric:
        _make_symmetric(graph)

    return graph


def _sample_zipf_degree(
    a: float,
    min_val: int,
    max_val: int,
    rng: np.random.Generator,
) -> int:
    """Sample a contact count from a Zipf distribution with clamping."""
    if max_val <= min_val:
        return min_val

    if a <= 1.0:
        a = 1.01

    for _ in range(100):
        u = float(rng.random())
        if u == 0.0:
            u = 1e-10
        raw = (1.0 - u) ** (-1.0 / (a - 1.0))
        degree = min_val + int(raw) - 1
        if degree < min_val:
            degree = min_val
        if degree <= max_val:
            return degree

    return min_val


def _weighted_sample_without_replacement(
    items: list[str],
    weights: list[float],
    k: int,
    rng: np.random.Generator,
) -> list[str]:
    """Weighted sampling without replacement using Efraimidis-Spirakis.

    Vectorized with numpy: replaces an O(n) Python loop + sort with
    numpy random generation + argpartition, ~10-15x faster for typical
    contact book sizes (k=5-10, n=50-1000 candidates).
    """
    n = len(items)
    if k >= n:
        result = list(items)
        rng.shuffle(result)
        return result

    w_arr = np.array(weights, dtype=np.float64)
    w_arr = np.maximum(w_arr, 1e-10)
    u_arr = rng.random(n)
    np.clip(u_arr, 1e-10, None, out=u_arr)
    key_arr = u_arr ** (1.0 / w_arr)

    # argpartition is O(n) vs O(n log n) for full sort
    top_k_idx = np.argpartition(-key_arr, k)[:k]
    # Sort only the top-k elements
    top_k_sorted = top_k_idx[np.argsort(-key_arr[top_k_idx])]
    return [items[int(i)] for i in top_k_sorted]


def _make_symmetric(graph: dict[str, list[str]]) -> None:
    """Ensure all contacts are bidirectional (modifies in place)."""
    edges: set[tuple[str, str]] = set()
    for imsi, contacts in graph.items():
        for c in contacts:
            edges.add((imsi, c))
            edges.add((c, imsi))

    for imsi in list(graph.keys()):
        graph[imsi] = []

    for a, b in edges:
        if a in graph and b not in graph[a]:
            graph[a].append(b)
