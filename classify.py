#!/usr/bin/env python3
"""Classify every paper in data/papers.json into one of the 24 topics in topics.json, one Jev choice per paper.

Papers go out in batches (default 4 per HTTP request), a bounded number of requests in flight.
The run is resumable: papers already in data/classified.json are skipped, progress is saved as it goes.
Latency and cost come from each API response, nothing is estimated.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jev

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "classified.json"


def batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def summarize(run: dict) -> dict:
    reqs, papers = run["requests"], run["papers"]
    per_paper = [p["ms"] for p in papers]
    per_request = [r["ms"] for r in reqs]
    cost = sum(r["cost"] for r in reqs)
    return {
        "papers": len(papers),
        "requests": len(reqs),
        "batch": run["batch"],
        "model": run.get("resolved_model") or run["model"],
        "total_cost": round(cost, 6),
        "cost_per_paper": round(cost / max(1, len(papers)), 8),
        "input_tokens": sum(r["input_tokens"] for r in reqs),
        "median_ms_per_paper": round(statistics.median(per_paper), 1) if per_paper else 0,
        "p95_ms_per_paper": round(jev.percentile(per_paper, 0.95), 1),
        "median_ms_per_request": round(statistics.median(per_request), 1) if per_request else 0,
        "p95_ms_per_request": round(jev.percentile(per_request, 0.95), 1),
    }


def save(run: dict) -> None:
    run["summary"] = summarize(run)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=1))
    os.replace(tmp, OUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file")
    ap.add_argument("--batch", type=int, default=4, help="papers per request")
    ap.add_argument("--workers", type=int, default=6, help="requests in flight")
    ap.add_argument("--limit", type=int, default=0, help="only the first N papers")
    ap.add_argument("--fresh", action="store_true", help="ignore data/classified.json and start over")
    a = ap.parse_args()

    key = jev.load_key(a.key_file)
    topics = jev.load_topics()
    papers = json.loads((ROOT / "data" / "papers.json").read_text())
    if a.limit:
        papers = papers[: a.limit]

    run = {"model": jev.MODEL, "batch": a.batch, "requests": [], "papers": []}
    if OUT.is_file() and not a.fresh:
        run = json.loads(OUT.read_text())
        if run.get("batch") != a.batch:
            sys.exit(f"data/classified.json was made with --batch {run.get('batch')}. Use the same value or --fresh.")
    done = {p["id"] for p in run["papers"]}
    todo = [p for p in papers if p["id"] not in done]
    print(f"{len(papers)} papers, {len(done)} already classified, {len(todo)} to go, batch {a.batch}, {a.workers} workers")

    failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(jev.classify_batch, key, b, topics): b for b in batches(todo, a.batch)}
        for n, fut in enumerate(as_completed(futures), 1):
            batch = futures[fut]
            try:
                rows, usage, took = fut.result()
            except Exception as err:  # one bad batch must not lose the rest; the next run retries it
                failed += len(batch)
                print(f"  batch failed ({len(batch)} papers): {err}", file=sys.stderr)
                continue
            run["resolved_model"] = usage.get("model") or run.get("resolved_model")
            ms, cost = took * 1000, float(usage.get("cost") or 0)
            run["requests"].append({"ids": [p["id"] for p in batch], "ms": round(ms, 1), "cost": cost,
                                    "input_tokens": int(usage.get("input_tokens") or 0)})
            for row in rows:
                run["papers"].append({**row, "ms": round(ms / len(batch), 1), "cost": cost / len(batch)})
            if n % 25 == 0:
                save(run)
                print(f"  {len(run['papers'])} classified")

    order = {p["id"]: i for i, p in enumerate(papers)}
    run["papers"].sort(key=lambda r: order.get(r["id"], len(order)))
    save(run)
    s = run["summary"]
    print(f"papers: {s['papers']}   requests: {s['requests']} ({s['batch']} per request)   model: {s['model']}")
    print(f"total cost: ${s['total_cost']:.4f}   per paper: ${s['cost_per_paper']:.6f}   input tokens: {s['input_tokens']}")
    print(f"per paper:   median {s['median_ms_per_paper']:.1f} ms   p95 {s['p95_ms_per_paper']:.1f} ms   (request time divided by papers in the request)")
    print(f"per request: median {s['median_ms_per_request']:.1f} ms   p95 {s['p95_ms_per_request']:.1f} ms")
    print("wrote", OUT, "and", jev.write_site_data())
    if failed:
        print(f"{failed} papers failed. Run the same command again to pick them up.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
