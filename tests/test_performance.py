"""Performance regression tests for CDR generation throughput.

These tests validate single-core CDR generation performance against hard
targets.  They are marked with ``@pytest.mark.benchmark`` so they can be
excluded from fast CI runs via ``pytest -m "not benchmark"``.

Performance baselines measured 2026-03-02 (Docker/Linux, single core):
    100 subscribers: ~1,542 CDR/sec (2,941 records, 1.9s)
    200 subscribers: ~1,525 CDR/sec (5,828 records, 3.8s)
    500 subscribers: ~1,546 CDR/sec (15,267 records, 9.9s)
   1000 subscribers: ~1,445 CDR/sec (30,175 records, 20.9s)

Target throughput is 50,000 CDR/sec — requires optimization work beyond
the current single-threaded engine.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from tests.benchmark_generation import build_benchmark_config

THROUGHPUT_TARGET = 50_000  # CDR records per second (single core)


@pytest.mark.benchmark
def test_single_core_throughput_minimum():
    """CDR generation must sustain >= 50k records/sec on a single core.

    This is a hard performance target.  The test will fail until the
    engine is optimized to reach production-scale throughput.
    """
    from cdr_generator.config.models import CDRGeneratorConfig
    from cdr_generator.engine.runner import run_generation

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = str(Path(tmpdir) / "output")
        Path(output_dir).mkdir()

        config_dict = build_benchmark_config(
            output_dir=output_dir,
            total_subscribers=100,
            hours=24,
        )
        config = CDRGeneratorConfig(**config_dict)

        t0 = time.perf_counter()
        stats = run_generation(config)
        elapsed = time.perf_counter() - t0

    throughput = stats.total_records / elapsed if elapsed > 0 else 0

    assert throughput >= THROUGHPUT_TARGET, (
        f"Single-core throughput {throughput:,.0f} CDR/sec "
        f"is below the target {THROUGHPUT_TARGET:,} CDR/sec. "
        f"Generated {stats.total_records:,} records in {elapsed:.2f}s. "
        f"(voice={stats.voice_records:,}, sms={stats.sms_records:,}, "
        f"data={stats.data_records:,})"
    )


@pytest.mark.benchmark
def test_throughput_scales_linearly():
    """Throughput should not degrade more than 20% when doubling subscribers.

    Catches O(n^2) regressions in the generation loop.  Measured baseline
    shows ~97% ratio (100 vs 200 subscribers), so the 80% floor is
    conservative and will only trigger on genuine algorithmic regressions.
    """
    from cdr_generator.config.models import CDRGeneratorConfig
    from cdr_generator.engine.runner import run_generation

    results = {}
    for n_subs in (100, 200):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = str(Path(tmpdir) / "output")
            Path(output_dir).mkdir()

            config_dict = build_benchmark_config(
                output_dir=output_dir,
                total_subscribers=n_subs,
                hours=24,
            )
            config = CDRGeneratorConfig(**config_dict)

            t0 = time.perf_counter()
            stats = run_generation(config)
            elapsed = time.perf_counter() - t0

        results[n_subs] = stats.total_records / elapsed if elapsed > 0 else 0

    ratio = results[200] / results[100] if results[100] > 0 else 0

    assert ratio >= 0.80, (
        f"Throughput degraded by {(1 - ratio) * 100:.1f}% when doubling subscribers "
        f"(100 subs: {results[100]:,.0f} CDR/sec, "
        f"200 subs: {results[200]:,.0f} CDR/sec). "
        f"Expected at most 20% degradation."
    )


@pytest.mark.benchmark
def test_large_scale_throughput(capsys):
    """Informational: measure throughput at 1000 subscribers.

    No assertion -- prints metrics as a data point for optimization
    progress at larger scale.  The 1000-subscriber workload exercises
    memory pressure and dict-lookup overhead that smaller runs do not.
    """
    from cdr_generator.config.models import CDRGeneratorConfig
    from cdr_generator.engine.runner import run_generation

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = str(Path(tmpdir) / "output")
        Path(output_dir).mkdir()

        config_dict = build_benchmark_config(
            output_dir=output_dir,
            total_subscribers=1000,
            hours=24,
        )
        config = CDRGeneratorConfig(**config_dict)

        t0 = time.perf_counter()
        stats = run_generation(config)
        elapsed = time.perf_counter() - t0

    throughput = stats.total_records / elapsed if elapsed > 0 else 0

    with capsys.disabled():
        print(
            f"\n  Large-scale benchmark (1000 subscribers):\n"
            f"    Wall time:     {elapsed:.2f}s\n"
            f"    Total records: {stats.total_records:,}\n"
            f"    Throughput:    {throughput:,.0f} CDR/sec\n"
            f"    Voice:         {stats.voice_records:,}\n"
            f"    SMS:           {stats.sms_records:,}\n"
            f"    Data:          {stats.data_records:,}\n"
            f"    Files:         {stats.files_written}"
        )
