"""Measure warm request times against your real model and data.

Run from the project root:
    python -m backend.benchmark

Times are for in-process requests (no network, no browser), so they show the cost of
the API and the similarity calculation themselves.
"""
import random
import statistics
import time

from fastapi.testclient import TestClient

from analysis.cageatlas.similarity import load_atlas

from .main import app


def timed(fn, runs):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "runs": runs,
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 1),
        "max_ms": round(times[-1], 1),
    }


def main():
    t0 = time.perf_counter()
    load_atlas()
    print(f"Cold load of model + profiles: {(time.perf_counter() - t0) * 1000:.0f} ms (once per server start)\n")

    with TestClient(app) as client:
        if client.get("/api/status").status_code != 200:
            print("API is not ready (model or profiles missing). Build them first.")
            return
        points = client.get("/api/map").json()["points"]
        sample = random.Random(7).sample(points, min(40, len(points)))
        for p in sample[:5]:                                   # warm-up
            client.get(f"/api/fighters/{p['fighter_id']}/similar")
            client.get("/api/fighters/search", params={"q": p["fighter_name"][:4]})

        cases = {
            "search (4-letter prefix)": lambda p: client.get(
                "/api/fighters/search", params={"q": p["fighter_name"][:4]}),
            "similar: top 10, no filters": lambda p: client.get(
                f"/api/fighters/{p['fighter_id']}/similar"),
            "similar: top 10, division + category filters": lambda p: client.get(
                f"/api/fighters/{p['fighter_id']}/similar",
                params={"division": "same", "category": "same"}),
            "compare 3 fighters": lambda p: client.get(
                "/api/compare", params={"ids": [s["fighter_id"] for s in sample[:3]]}),
            "style map (cached)": lambda p: client.get("/api/map"),
        }
        print(f"{'request':<48}{'median ms':>10}{'p95 ms':>9}{'max ms':>9}")
        for name, call in cases.items():
            idx = iter(range(10_000))
            stats = timed(lambda: call(sample[next(idx) % len(sample)]), runs=len(sample))
            print(f"{name:<48}{stats['median_ms']:>10}{stats['p95_ms']:>9}{stats['max_ms']:>9}")


if __name__ == "__main__":
    main()
