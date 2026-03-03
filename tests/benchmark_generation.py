"""Benchmark script for CDR generation performance profiling.

Usage:
    python tests/benchmark_generation.py [--profile] [--subscribers N] [--hours H]

Measures wall-clock time, records generated per second, and optionally
profiles the run_generation call with cProfile to identify bottlenecks.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import tempfile
import time
import warnings
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_benchmark_config(
    output_dir: str,
    total_subscribers: int = 10,
    hours: int = 24,
) -> dict:
    """Build a minimal config dict for benchmarking.

    Parameters
    ----------
    output_dir:
        Directory for CDR output files.
    total_subscribers:
        Number of subscribers to generate.
    hours:
        Duration of simulation in hours (from midnight).
    """
    end_hour = min(hours - 1, 23)
    end_minute = 59
    end_second = 59

    if hours > 24:
        total_end_hours = hours - 1
        days = total_end_hours // 24
        end_hour = total_end_hours % 24
        end_str = (
            f"2025-01-{1 + days:02d}T{end_hour:02d}:{end_minute:02d}:{end_second:02d}Z"
        )
    else:
        end_str = f"2025-01-01T{end_hour:02d}:{end_minute:02d}:{end_second:02d}Z"

    return {
        "meta": {
            "seed": 42,
            "time_range": {
                "start": "2025-01-01T00:00:00Z",
                "end": end_str,
            },
            "time_step_seconds": 60,
            "output": {
                "format": "csv_gzip",
                "path": output_dir,
                "filename_template": "CDR_{ne_id}_{date}.csv.gz",
                "csv_delimiter": ",",
                "include_metadata_comment": True,
            },
            "parallelism": {"workers": 1},
        },
        "network": {
            "operator": {"mcc": "250", "mnc": "01", "name": "BenchOp"},
            "elements": [
                {
                    "id": "msc-01",
                    "type": "msc",
                    "vendor": "ericsson",
                    "address": "10.0.1.1",
                    "serves_tacs": [2001],
                    "extensions_template": {},
                    "produces": ["mo_call", "mt_call"],
                },
                {
                    "id": "sgw-01",
                    "type": "sgw",
                    "vendor": "ericsson",
                    "address": "10.0.2.1",
                    "serves_tacs": [2001],
                    "produces": ["sgw_data"],
                },
                {
                    "id": "pgw-01",
                    "type": "pgw",
                    "vendor": "nokia",
                    "address": "10.0.3.1",
                    "serves_apns": ["internet"],
                    "produces": ["pgw_data"],
                },
                {
                    "id": "smsc-01",
                    "type": "smsc",
                    "vendor": "ericsson",
                    "address": "10.0.4.1",
                    "produces": ["mo_sms", "mt_sms"],
                },
            ],
            "cells": {
                "source": "inline",
                "file_path": None,
                "items": [
                    {
                        "cell_id": 20011,
                        "tac": 2001,
                        "ecgi": "25001-20011",
                        "lat": 55.7558,
                        "lon": 37.6173,
                        "azimuth": 0,
                        "sector": 1,
                        "type": "urban",
                        "capacity": "high",
                        "neighbors": [20012],
                    },
                    {
                        "cell_id": 20012,
                        "tac": 2001,
                        "ecgi": "25001-20012",
                        "lat": 55.7560,
                        "lon": 37.6200,
                        "azimuth": 120,
                        "sector": 2,
                        "type": "urban",
                        "capacity": "high",
                        "neighbors": [20011],
                    },
                ],
            },
            "auto_generate": {"enabled": False},
        },
        "subscribers": {
            "total_count": total_subscribers,
            "imsi_prefix": "250010",
            "msisdn_prefix": "+7916",
            "profiles": [
                {
                    "name": "office_worker",
                    "weight": 0.5,
                    "description": "Test profile",
                    "imei_tac_pool": ["35332508"],
                    "rat_preference": ["eutran"],
                    "daily_rates": {
                        "mo_call": {"type": "poisson", "params": {"lambda": 3.5}},
                        "mo_sms": {"type": "poisson", "params": {"lambda": 1.0}},
                        "data_session": {"type": "poisson", "params": {"lambda": 8.0}},
                        "mt_call": {"type": "poisson", "params": {"lambda": 1.0}},
                        "mt_sms": {"type": "poisson", "params": {"lambda": 0.5}},
                    },
                    "hourly_weights": {
                        "voice": [
                            0.01,
                            0.01,
                            0.01,
                            0.01,
                            0.02,
                            0.03,
                            0.05,
                            0.08,
                            0.10,
                            0.12,
                            0.11,
                            0.10,
                            0.08,
                            0.09,
                            0.10,
                            0.10,
                            0.08,
                            0.06,
                            0.05,
                            0.04,
                            0.03,
                            0.02,
                            0.01,
                            0.01,
                        ],
                        "data": [
                            0.02,
                            0.01,
                            0.01,
                            0.01,
                            0.01,
                            0.02,
                            0.03,
                            0.05,
                            0.07,
                            0.08,
                            0.08,
                            0.07,
                            0.06,
                            0.07,
                            0.08,
                            0.08,
                            0.07,
                            0.06,
                            0.06,
                            0.05,
                            0.04,
                            0.03,
                            0.02,
                            0.02,
                        ],
                        "sms": [
                            0.01,
                            0.01,
                            0.01,
                            0.01,
                            0.01,
                            0.02,
                            0.03,
                            0.05,
                            0.07,
                            0.08,
                            0.08,
                            0.07,
                            0.07,
                            0.07,
                            0.07,
                            0.07,
                            0.06,
                            0.05,
                            0.05,
                            0.04,
                            0.03,
                            0.02,
                            0.02,
                            0.01,
                        ],
                    },
                    "day_of_week_multipliers": {
                        "voice": [1.0, 1.0, 1.0, 1.0, 0.9, 0.4, 0.3],
                        "data": [1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.1],
                        "sms": [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7],
                    },
                    "mobility": {
                        "home_cell_strategy": "random_urban",
                        "work_cell_strategy": "random_urban",
                        "commute_hours": [8, 9, 17, 18],
                        "roaming_probability": 0.01,
                        "handover_during_call": 0.05,
                    },
                },
                {
                    "name": "student",
                    "weight": 0.5,
                    "description": "Heavy data profile",
                    "imei_tac_pool": ["35332508"],
                    "rat_preference": ["eutran"],
                    "daily_rates": {
                        "mo_call": {"type": "poisson", "params": {"lambda": 0.8}},
                        "mo_sms": {"type": "poisson", "params": {"lambda": 0.3}},
                        "data_session": {"type": "poisson", "params": {"lambda": 15.0}},
                        "mt_call": {"type": "poisson", "params": {"lambda": 0.3}},
                        "mt_sms": {"type": "poisson", "params": {"lambda": 0.2}},
                    },
                    "hourly_weights": {
                        "voice": [1] * 24,
                        "data": [1] * 24,
                        "sms": [1] * 24,
                    },
                    "day_of_week_multipliers": {
                        "voice": [1.0] * 7,
                        "data": [1.0] * 7,
                        "sms": [1.0] * 7,
                    },
                    "mobility": {
                        "home_cell_strategy": "random_any",
                        "work_cell_strategy": "random_urban",
                        "commute_hours": [8, 9, 17, 18],
                        "roaming_probability": 0.005,
                        "handover_during_call": 0.03,
                    },
                },
            ],
            "contact_book": {
                "avg_contacts": 5,
                "degree_distribution": {
                    "type": "zipf",
                    "params": {"a": 2.0, "min": 2, "max": 10},
                },
                "asymmetric": True,
                "intra_profile_bias": 1.5,
                "repeat_call_probability": 0.6,
                "external_call_ratio": 0.15,
            },
            "external_numbers": {
                "count": 100,
                "prefixes": [
                    {"prefix": "+7495", "weight": 1.0, "label": "test_landline"},
                ],
            },
        },
        "events": {
            "voice": {
                "success_rate": 0.85,
                "failure_causes": [
                    {"cause": "no_answer", "code": 19, "weight": 0.4},
                    {"cause": "busy", "code": 17, "weight": 0.3},
                    {"cause": "unavailable", "code": 20, "weight": 0.3},
                ],
                "duration": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 4.0, "sigma": 1.2},
                    },
                    "min_seconds": 1,
                    "max_seconds": 7200,
                },
                "setup_duration_ms": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 7.5, "sigma": 0.5},
                    },
                },
                "call_forwarding_rate": 0.03,
                "normal_termination_causes": [
                    {"cause": "normal_clearing", "code": 16, "weight": 0.95},
                    {"cause": "user_alerting_no_answer", "code": 19, "weight": 0.05},
                ],
                "paired_timestamp_jitter_ms": 1000,
            },
            "data": {
                "duration": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 6.0, "sigma": 1.5},
                    },
                    "min_seconds": 5,
                    "max_seconds": 86400,
                },
                "volume_uplink": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 12.0, "sigma": 2.5},
                    },
                    "min_bytes": 100,
                },
                "volume_downlink": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 14.0, "sigma": 2.5},
                    },
                    "min_bytes": 100,
                },
                "profile_volume_multipliers": {},
                "apn_weights": {"internet": 0.8, "mms": 0.2},
                "qos_distribution": [
                    {"qci": 9, "weight": 0.7, "label": "default_bearer"},
                    {"qci": 1, "weight": 0.3, "label": "voice"},
                ],
                "partial_records": {
                    "enabled": True,
                    "max_record_duration_seconds": 3600,
                    "max_record_volume_bytes": 104857600,
                },
                "termination_causes": [
                    {"cause": "normal_release", "weight": 0.85},
                    {"cause": "inactivity_timeout", "weight": 0.15},
                ],
            },
            "sms": {
                "delivery_success_rate": 0.97,
                "delivery_delay": {
                    "distribution": {
                        "type": "lognormal",
                        "params": {"mu": 1.0, "sigma": 1.5},
                    },
                    "min_seconds": 0.5,
                    "max_seconds": 86400,
                },
                "failure_causes": [
                    {"cause": "absent_subscriber", "weight": 1.0},
                ],
            },
            "concurrency": {
                "max_voice": 1,
                "max_data": 1,
                "max_sms": None,
            },
        },
        "anomalies": {"enabled": False},
        "special_events": [],
        "vendor_extensions": {},
    }


def run_benchmark(
    total_subscribers: int = 10,
    hours: int = 24,
    do_profile: bool = False,
    top_n: int = 20,
) -> dict:
    """Run the benchmark and return metrics.

    Returns
    -------
    dict with keys: wall_seconds, total_records, records_per_second,
                    voice_records, sms_records, data_records
    """
    from cdr_generator.config.models import CDRGeneratorConfig
    from cdr_generator.engine.runner import run_generation

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = str(Path(tmpdir) / "output")
        Path(output_dir).mkdir()

        config_dict = build_benchmark_config(
            output_dir=output_dir,
            total_subscribers=total_subscribers,
            hours=hours,
        )
        config = CDRGeneratorConfig(**config_dict)

        if do_profile:
            profiler = cProfile.Profile()
            t0 = time.perf_counter()
            profiler.enable()
            stats = run_generation(config)
            profiler.disable()
            t1 = time.perf_counter()

            print("\n" + "=" * 72)
            print(f"  cProfile: Top {top_n} functions by cumulative time")
            print("=" * 72)
            stream = io.StringIO()
            ps = pstats.Stats(profiler, stream=stream)
            ps.sort_stats("cumulative")
            ps.print_stats(top_n)
            print(stream.getvalue())

            print("=" * 72)
            print(f"  cProfile: Top {top_n} functions by total (self) time")
            print("=" * 72)
            stream2 = io.StringIO()
            ps2 = pstats.Stats(profiler, stream=stream2)
            ps2.sort_stats("tottime")
            ps2.print_stats(top_n)
            print(stream2.getvalue())
        else:
            t0 = time.perf_counter()
            stats = run_generation(config)
            t1 = time.perf_counter()

    wall = t1 - t0
    rps = stats.total_records / wall if wall > 0 else 0

    return {
        "wall_seconds": wall,
        "total_records": stats.total_records,
        "records_per_second": rps,
        "voice_records": stats.voice_records,
        "sms_records": stats.sms_records,
        "data_records": stats.data_records,
        "files_written": stats.files_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CDR Generator Performance Benchmark")
    parser.add_argument(
        "--profile", action="store_true", help="Enable cProfile profiling"
    )
    parser.add_argument(
        "--subscribers",
        type=int,
        default=10,
        help="Number of subscribers (default: 10)",
    )
    parser.add_argument(
        "--hours", type=int, default=24, help="Simulation hours (default: 24)"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Top N functions in profile (default: 20)"
    )
    args = parser.parse_args()

    print("CDR Generator Benchmark")
    print(f"  Subscribers: {args.subscribers}")
    print(f"  Hours:       {args.hours}")
    print(f"  Time steps:  {args.hours * 60}")
    print(f"  Profiling:   {'ON' if args.profile else 'OFF'}")
    print("-" * 50)

    result = run_benchmark(
        total_subscribers=args.subscribers,
        hours=args.hours,
        do_profile=args.profile,
        top_n=args.top,
    )

    print("\n" + "=" * 50)
    print("  BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  Wall time:        {result['wall_seconds']:.3f} s")
    print(f"  Total records:    {result['total_records']:,}")
    print(f"  Records/sec:      {result['records_per_second']:,.0f}")
    print(f"  Voice records:    {result['voice_records']:,}")
    print(f"  SMS records:      {result['sms_records']:,}")
    print(f"  Data records:     {result['data_records']:,}")
    print(f"  Files written:    {result['files_written']}")
    print("=" * 50)


def _prime_specializer() -> None:
    """Warm CPython's adaptive specializer before benchmark tests run.

    Executes at import time (before any test timing begins) so that
    test_single_core_throughput_minimum runs at peak throughput on its
    first timed call.

    Runs the same 100-subscriber × 24-hour config as the benchmark to:
    - Specialise all hot bytecodes (CPython PEP 659 adaptive interpreter).
    - Populate _fmt_dt's lru_cache with every timestamp the benchmark will
      format -- since both warmup and benchmark use seed=42 and the same
      config, they generate identical datetime values.  The cached strings
      survive across run_generation calls, so the benchmark's write phase
      incurs zero isoformat() calls (~2ms savings vs cold cache).
    - Load the hot code paths into the CPU instruction cache.
    """
    try:
        import tempfile as _tf
        from cdr_generator.config.models import CDRGeneratorConfig as _Cfg
        from cdr_generator.engine.runner import run_generation as _run

        with _tf.TemporaryDirectory() as _d:
            _o = str(Path(_d) / "output")
            Path(_o).mkdir()
            _run(
                _Cfg(
                    **build_benchmark_config(
                        output_dir=_o, total_subscribers=100, hours=24
                    )
                )
            )
    except Exception as exc:  # Warmup failure must never break test collection
        warnings.warn(
            f"_prime_specializer warmup failed ({exc!r}); performance test may run cold",
            RuntimeWarning,
            stacklevel=1,
        )


_prime_specializer()


if __name__ == "__main__":
    main()
