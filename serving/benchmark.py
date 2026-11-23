"""Empirical API latency benchmark with warm-up and concurrent load."""
from __future__ import annotations

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from src.data_pipeline import FEATURES


def percentile_summary(latencies_ms: list[float], elapsed_seconds: float, errors: int) -> dict:
    values = np.asarray(latencies_ms, dtype=float)
    return {
        "requests": int(len(values)),
        "errors": int(errors),
        "error_rate": float(errors / max(1, len(values))),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
        "max_ms": float(np.max(values)),
        "wall_seconds": float(elapsed_seconds),
        "throughput_requests_per_second": float(len(values) / elapsed_seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/api_benchmark.json"))
    args = parser.parse_args()
    if args.requests < 100:
        raise ValueError("Use at least 100 measured requests")

    frame = pd.read_csv(args.data, nrows=max(args.requests, 200), usecols=FEATURES)
    payloads = frame.astype(float).to_dict(orient="records")
    timeout = httpx.Timeout(10.0)
    limits = httpx.Limits(max_connections=max(16, args.concurrency * 2), max_keepalive_connections=max(8, args.concurrency))

    with httpx.Client(base_url=args.base_url, timeout=timeout, limits=limits) as client:
        health = client.get("/health")
        health.raise_for_status()
        if not health.json().get("artifact_loaded"):
            raise RuntimeError("API health endpoint reports no loaded artifact")
        for i in range(args.warmup):
            response = client.post("/predict", json=payloads[i % len(payloads)])
            response.raise_for_status()

        def one_request(i: int) -> tuple[float, int]:
            start = time.perf_counter_ns()
            response = client.post("/predict", json=payloads[i % len(payloads)])
            latency_ms = (time.perf_counter_ns() - start) / 1_000_000
            return latency_ms, response.status_code

        start_wall = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(one_request, range(args.requests)))
        elapsed = time.perf_counter() - start_wall

    latencies = [row[0] for row in results]
    errors = sum(status != 200 for _, status in results)
    report = {
        "benchmark": percentile_summary(latencies, elapsed, errors),
        "configuration": {
            "base_url": args.base_url,
            "warmup_requests": args.warmup,
            "measured_requests": args.requests,
            "concurrency": args.concurrency,
            "protocol_path": "HTTP POST /predict through running Uvicorn",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
