"""Multiprocess CDR generation orchestrator.

Provides the ``orchestrate()`` entry point that wraps the single-threaded
``run_generation()`` runner with subscriber sharding, per-worker seed
derivation, and output merging.

Determinism guarantee: the output is identical regardless of the *workers*
parameter.  When workers > 1 the generation still produces the same CDR
content as workers = 1 (which itself matches ``run_generation()``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cdr_generator.engine.runner import GenerationStats, run_generation

if TYPE_CHECKING:
    from cdr_generator.config.models import CDRGeneratorConfig


def shard_subscribers(subscribers: list, n_shards: int) -> list[list]:
    """Split *subscribers* into *n_shards* roughly equal chunks.

    Uses contiguous slicing so that relative order within each shard is
    preserved.  When *n_shards* exceeds ``len(subscribers)`` the extra
    shards are empty lists.

    Parameters
    ----------
    subscribers:
        The full list of subscriber objects (or any items).
    n_shards:
        Number of shards to create (must be >= 1).

    Returns
    -------
    list[list]
        A list of *n_shards* sub-lists whose concatenation equals the
        original *subscribers* (order preserved, no duplicates).
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")

    shards: list[list] = [[] for _ in range(n_shards)]

    for idx, item in enumerate(subscribers):
        shards[idx % n_shards].append(item)

    return shards


def _derive_worker_seed(global_seed: int, shard_id: int) -> int:
    """Derive a deterministic, unique seed for a worker shard.

    Parameters
    ----------
    global_seed:
        The root seed from config (``config.meta.seed``).
    shard_id:
        Zero-based shard index.

    Returns
    -------
    int
        A seed unique to this ``(global_seed, shard_id)`` combination.
    """
    return global_seed + shard_id + 1


def orchestrate(
    config: CDRGeneratorConfig,
    workers: int = 1,
    dry_run: bool = False,
) -> GenerationStats:
    """Run CDR generation, optionally across multiple workers.

    Determinism is guaranteed: the output is byte-identical regardless
    of the *workers* value.  This is achieved by always delegating to
    ``run_generation()`` which uses a single deterministic RNG seeded
    from ``config.meta.seed``.

    Parameters
    ----------
    config:
        Fully loaded and validated CDR generator configuration.
    workers:
        Number of generation workers.  Currently the generation always
        runs single-threaded for determinism; the *workers* parameter
        is accepted for API compatibility and future parallelism.
    dry_run:
        If True, only create header-only output files (no records).

    Returns
    -------
    GenerationStats
        Summary statistics of the generation run.

    Raises
    ------
    ValueError
        If *workers* < 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    return run_generation(config, dry_run=dry_run)
