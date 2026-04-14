# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pluggable database backends for benchmark result reporting."""

import json
from abc import ABC, abstractmethod
from typing import List
from urllib.parse import urlparse

from benchmarks.common.result_types import BenchmarkResult


class DatabaseBackend(ABC):
    """Abstract database backend for storing benchmark results."""

    @abstractmethod
    def report(self, results: List[BenchmarkResult]) -> None:
        """Store benchmark results in the database."""
        ...


class HTTPBackend(DatabaseBackend):
    """POST results as JSON to a REST endpoint."""

    def __init__(self, url: str):
        self.url = url

    def report(self, results: List[BenchmarkResult]) -> None:
        import urllib.request

        data = json.dumps([r.to_dict() for r in results]).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"Database: uploaded {len(results)} results (HTTP {resp.status})")
        except Exception as e:
            print(f"Database: upload failed: {e}")


class PostgreSQLBackend(DatabaseBackend):
    """Insert results into a PostgreSQL table."""

    def __init__(self, url: str):
        self.url = url

    def report(self, results: List[BenchmarkResult]) -> None:
        try:
            import psycopg2
        except ImportError:
            print("Database: psycopg2 not installed, skipping PostgreSQL upload")
            return

        try:
            conn = psycopg2.connect(self.url)
            cur = conn.cursor()

            for r in results:
                d = r.to_dict()
                cur.execute(
                    """INSERT INTO benchmark_results
                       (benchmark_name, category, framework, shape, dtype, recipe,
                        direction, regime, median_time_us, mean_time_us,
                        min_time_us, max_time_us, std_time_us,
                        throughput_gbps, tflops, num_iterations, num_gpus,
                        git_commit, te_version, hostname, gpu_model)
                       VALUES (%(name)s, %(category)s, %(framework)s, %(shape)s,
                               %(dtype)s, %(recipe)s, %(direction)s, %(regime)s,
                               %(median_time_us)s, %(mean_time_us)s,
                               %(min_time_us)s, %(max_time_us)s, %(std_time_us)s,
                               %(throughput_gbps)s, %(tflops)s,
                               %(num_iterations)s, %(num_gpus)s,
                               %(git_commit)s, %(te_version)s,
                               %(hostname)s, %(gpu_model)s)""",
                    {
                        **d,
                        "throughput_gbps": d.get("throughput_gbps"),
                        "tflops": d.get("tflops"),
                        "git_commit": d.get("git_commit", "unknown"),
                        "te_version": d.get("te_version", "unknown"),
                        "hostname": d.get("hostname", "unknown"),
                        "gpu_model": d.get("gpu_model", "unknown"),
                    },
                )

            conn.commit()
            cur.close()
            conn.close()
            print(f"Database: uploaded {len(results)} results to PostgreSQL")
        except Exception as e:
            print(f"Database: PostgreSQL upload failed: {e}")


class InfluxDBBackend(DatabaseBackend):
    """Write results using InfluxDB line protocol over HTTP."""

    def __init__(self, url: str):
        # Convert influxdb://host:port/db to http://host:port/write?db=db
        parsed = urlparse(url)
        db_name = parsed.path.strip("/") or "benchmarks"
        self.write_url = f"http://{parsed.hostname}:{parsed.port or 8086}/write?db={db_name}"
        self.auth = None
        if parsed.username:
            self.auth = f"{parsed.username}:{parsed.password or ''}"

    def report(self, results: List[BenchmarkResult]) -> None:
        import urllib.request

        lines = []
        for r in results:
            # InfluxDB line protocol:
            # measurement,tag=val,tag=val field=val,field=val timestamp
            tags = (
                f"name={_escape(r.name)},"
                f"category={_escape(r.category)},"
                f"framework={r.framework},"
                f"recipe={r.recipe},"
                f"direction={r.direction},"
                f"regime={r.regime},"
                f"dtype={r.dtype}"
            )
            fields = (
                f"median_time_us={r.median_time_us},"
                f"mean_time_us={r.mean_time_us},"
                f"min_time_us={r.min_time_us},"
                f"max_time_us={r.max_time_us},"
                f"std_time_us={r.std_time_us},"
                f"num_iterations={r.num_iterations}i,"
                f"num_gpus={r.num_gpus}i"
            )
            if r.throughput_gbps is not None:
                fields += f",throughput_gbps={r.throughput_gbps}"
            if r.tflops is not None:
                fields += f",tflops={r.tflops}"
            lines.append(f"te_benchmark,{tags} {fields}")

        data = "\n".join(lines).encode("utf-8")
        req = urllib.request.Request(self.write_url, data=data, method="POST")
        if self.auth:
            import base64
            cred = base64.b64encode(self.auth.encode()).decode()
            req.add_header("Authorization", f"Basic {cred}")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"Database: uploaded {len(results)} results to InfluxDB (HTTP {resp.status})")
        except Exception as e:
            print(f"Database: InfluxDB upload failed: {e}")


def _escape(s: str) -> str:
    """Escape special characters for InfluxDB line protocol tags."""
    return s.replace(",", r"\,").replace("=", r"\=").replace(" ", r"\ ")


def create_backend(url: str) -> DatabaseBackend:
    """Create a database backend based on the URL scheme."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme in ("http", "https"):
        return HTTPBackend(url)
    elif scheme in ("postgresql", "postgres"):
        return PostgreSQLBackend(url)
    elif scheme == "influxdb":
        return InfluxDBBackend(url)
    else:
        raise ValueError(
            f"Unsupported database URL scheme: {scheme}. "
            f"Supported: http, https, postgresql, postgres, influxdb"
        )
