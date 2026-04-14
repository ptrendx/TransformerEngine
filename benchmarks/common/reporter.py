# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark result reporting to stdout, CSV, JSON, and optional database."""

import csv
import io
import json
import sys
from typing import List, Optional

from benchmarks.common.result_types import BenchmarkResult
from benchmarks.config.benchmark_config import BenchmarkConfig


class BenchmarkReporter:
    """Reports benchmark results to stdout and optionally to file/database."""

    def __init__(self, config: BenchmarkConfig, metadata: Optional[dict] = None):
        self.config = config
        self.metadata = metadata or {}
        self._results: List[BenchmarkResult] = []
        self._db_backend = None

        # Initialize database backend if URL provided
        if config.database_url:
            from benchmarks.common.database_backend import create_backend
            self._db_backend = create_backend(config.database_url)

    def report(self, result: BenchmarkResult) -> None:
        """Report a single result. Always prints to stdout."""
        # Merge global metadata
        if self.metadata:
            merged = dict(self.metadata)
            merged.update(result.metadata)
            result.metadata = merged

        self._results.append(result)
        self._print_result(result)

    def _print_result(self, result: BenchmarkResult) -> None:
        """Print a single result line to stdout."""
        shape_str = "x".join(str(s) for s in result.shape)
        parts = [
            f"[{result.framework:>7s}]",
            f"{result.name:<35s}",
            f"{shape_str:<25s}",
            f"{result.recipe:<12s}",
            f"{result.direction:<8s}",
            f"median={result.median_time_us:>10.1f} us",
            f"mean={result.mean_time_us:>10.1f} us",
        ]
        if result.throughput_gbps is not None:
            parts.append(f"bw={result.throughput_gbps:>8.1f} GB/s")
        if result.tflops is not None:
            parts.append(f"perf={result.tflops:>8.2f} TFLOPS")
        print("  ".join(parts), flush=True)

    def report_table(self) -> None:
        """Print a summary table of all collected results."""
        if not self._results:
            return

        print("\n" + "=" * 120)
        print("BENCHMARK SUMMARY")
        print("=" * 120)

        # Header
        header = (
            f"{'Name':<35s}  {'Shape':<25s}  {'Recipe':<10s}  {'Dir':<8s}  "
            f"{'Median(us)':>12s}  {'Mean(us)':>12s}  {'BW(GB/s)':>10s}  "
            f"{'TFLOPS':>8s}  {'Regime':<10s}"
        )
        print(header)
        print("-" * 120)

        for r in self._results:
            shape_str = "x".join(str(s) for s in r.shape)
            bw = f"{r.throughput_gbps:.1f}" if r.throughput_gbps is not None else "-"
            tf = f"{r.tflops:.2f}" if r.tflops is not None else "-"
            line = (
                f"{r.name:<35s}  {shape_str:<25s}  {r.recipe:<10s}  {r.direction:<8s}  "
                f"{r.median_time_us:>12.1f}  {r.mean_time_us:>12.1f}  {bw:>10s}  "
                f"{tf:>8s}  {r.regime:<10s}"
            )
            print(line)

        print("=" * 120)
        print(f"Total benchmarks: {len(self._results)}")
        print()

    def finalize(self) -> None:
        """Finalize reporting: write files and upload to database."""
        self.report_table()

        if self.config.output_format == "csv" and self.config.output_file:
            self._write_csv()
        elif self.config.output_format == "json" and self.config.output_file:
            self._write_json()

        if self._db_backend and self._results:
            self._db_backend.report(self._results)

    def _write_csv(self) -> None:
        """Write results to CSV file."""
        if not self._results:
            return
        filepath = self.config.output_file
        fieldnames = list(self._results[0].to_dict().keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                writer.writerow(r.to_dict())
        print(f"Results written to {filepath}")

    def _write_json(self) -> None:
        """Write results to JSON file."""
        if not self._results:
            return
        filepath = self.config.output_file
        data = [r.to_dict() for r in self._results]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results written to {filepath}")
