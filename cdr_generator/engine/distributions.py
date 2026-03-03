"""Distribution sampling helpers for CDR generation.

Samples values from configured distributions (lognormal, constant, etc.)
with optional min/max clamping.
"""

from __future__ import annotations


import numpy as np


def sample_distribution(
    dist_type: str,
    params: dict,
    rng: np.random.Generator,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Sample a single value from the specified distribution.

    Parameters
    ----------
    dist_type:
        Distribution name: lognormal, normal, uniform, exponential,
        poisson, constant, categorical.
    params:
        Distribution parameters (varies by type).
    rng:
        Numpy random generator.
    min_val:
        Optional lower clamp.
    max_val:
        Optional upper clamp.

    Returns
    -------
    float
        The sampled value, clamped to [min_val, max_val] if provided.
    """
    value: float

    if dist_type == "lognormal":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        value = float(rng.lognormal(mu, sigma))
    elif dist_type == "normal":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)
        value = float(rng.normal(mu, sigma))
    elif dist_type == "uniform":
        low = params.get("low", 0.0)
        high = params.get("high", 1.0)
        value = float(rng.uniform(low, high))
    elif dist_type == "exponential":
        scale = params.get("scale", 1.0)
        value = float(rng.exponential(scale))
    elif dist_type == "constant":
        value = float(params.get("value", 0.0))
    elif dist_type == "poisson":
        lam = params.get("lambda", 1.0)
        value = float(rng.poisson(lam))
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type!r}")

    if min_val is not None:
        value = max(value, min_val)
    if max_val is not None:
        value = min(value, max_val)

    return value


def sample_from_config(
    config: dict,
    rng: np.random.Generator,
) -> float:
    """Sample from a DistributionWithBounds-style config dict.

    Expects keys: distribution.type, distribution.params, and optional
    min_seconds/max_seconds or min_bytes.
    """
    dist = config["distribution"]
    return sample_distribution(
        dist_type=dist["type"],
        params=dist.get("params", {}),
        rng=rng,
        min_val=config.get("min_seconds") or config.get("min_bytes"),
        max_val=config.get("max_seconds") or config.get("max_bytes"),
    )


def weighted_choice(
    items: list[dict],
    weight_key: str,
    rng: np.random.Generator,
) -> dict:
    """Select one item from a weighted list.

    Parameters
    ----------
    items:
        List of dicts, each having a weight field.
    weight_key:
        Key name for the weight value in each dict.
    rng:
        Numpy random generator.

    Returns
    -------
    dict
        The selected item.
    """
    weights = [item[weight_key] for item in items]
    total = sum(weights)
    if total <= 0:
        return items[0]
    probs = [w / total for w in weights]
    idx = int(rng.choice(len(items), p=probs))
    return items[idx]


def weighted_choice_str(
    mapping: dict[str, float],
    rng: np.random.Generator,
) -> str:
    """Select a key from a {key: weight} mapping.

    Parameters
    ----------
    mapping:
        Dict of {name: weight}.
    rng:
        Numpy random generator.

    Returns
    -------
    str
        The selected key.
    """
    keys = list(mapping.keys())
    weights = [mapping[k] for k in keys]
    total = sum(weights)
    if total <= 0:
        return keys[0]
    probs = [w / total for w in weights]
    idx = int(rng.choice(len(keys), p=probs))
    return keys[idx]
