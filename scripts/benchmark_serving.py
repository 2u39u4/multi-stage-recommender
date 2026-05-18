"""Small latency benchmark for the local FastAPI recommender.

This is intentionally simpler than locust/wrk so it works in constrained
environments and can be copied into README numbers.
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _call(base_url: str, user_id: int, k: int, diversity: float) -> float:
    t0 = time.perf_counter()
    r = requests.get(
        f"{base_url.rstrip('/')}/recommend/{user_id}",
        params={"k": k, "diversity": diversity},
        timeout=10,
    )
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000.0


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(int(round((len(xs) - 1) * q)), len(xs) - 1)
    return xs[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--users", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--diversity", type=float, default=0.7)
    args = parser.parse_args()

    users = [int(x) for x in args.users.split(",") if x.strip()]
    latencies: list[float] = []
    errors = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_call, args.url, users[i % len(users)], args.k, args.diversity)
            for i in range(args.requests)
        ]
        for fut in as_completed(futures):
            try:
                latencies.append(float(fut.result()))
            except Exception as exc:
                errors += 1
                print(f"request failed: {exc}")
    elapsed = time.perf_counter() - t0

    ok = len(latencies)
    print(f"requests_ok={ok} errors={errors} elapsed_s={elapsed:.2f}")
    if latencies:
        print(f"qps={ok / elapsed:.2f}")
        print(f"p50_ms={statistics.median(latencies):.2f}")
        print(f"p95_ms={_percentile(latencies, 0.95):.2f}")
        print(f"p99_ms={_percentile(latencies, 0.99):.2f}")


if __name__ == "__main__":
    main()
