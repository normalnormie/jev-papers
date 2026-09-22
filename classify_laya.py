#!/usr/bin/env python3
"""Classify the committed paper corpus with the local Laya English checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import jev
import laya_bench

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "laya_classified.json"


def batches(items: list, size: int) -> list[list]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def summarize(run: dict) -> dict:
    requests, papers = run["requests"], run["papers"]
    per_paper = [p["ms"] for p in papers]
    per_request = [r["ms"] for r in requests]
    return {
        "papers": len(papers),
        "requests": len(requests),
        "batch": run["batch"],
        "model": run["model"],
        "resolved_revision": run.get("resolved_revision"),
        "api_cost_usd": 0,
        "input_tokens": sum(r["input_tokens"] for r in requests),
        "model_load_ms": run.get("model_load_ms"),
        "warmup_ms": run.get("warmup_ms"),
        "median_ms_per_paper": round(statistics.median(per_paper), 1) if per_paper else 0,
        "p95_ms_per_paper": round(jev.percentile(per_paper, 0.95), 1),
        "median_ms_per_request": round(statistics.median(per_request), 1) if per_request else 0,
        "p95_ms_per_request": round(jev.percentile(per_request, 0.95), 1),
        **run.get("runtime", {}),
    }


def save(run: dict) -> None:
    run["summary"] = summarize(run)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=1))
    os.replace(tmp, OUT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=laya_bench.MODEL)
    parser.add_argument("--revision", help="Hugging Face commit or tag; latest main is resolved and recorded by default")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch", type=int, default=1, help="independent paper sequences per forward pass")
    parser.add_argument("--limit", type=int, default=0, help="only the first N papers")
    parser.add_argument("--fresh", action="store_true", help="ignore data/laya_classified.json and start over")
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    topics = jev.load_topics()
    papers = json.loads((ROOT / "data" / "papers.json").read_text())
    if args.limit:
        papers = papers[:args.limit]

    run = {
        "model": args.model,
        "batch": args.batch,
        "rubric_sha256": laya_bench.rubric_sha256(topics),
        "requests": [],
        "papers": [],
    }
    if OUT.is_file() and not args.fresh:
        run = json.loads(OUT.read_text())
        if run.get("model") != args.model:
            sys.exit(f"{OUT} was made with {run.get('model')!r}; use that model or --fresh")
        if run.get("batch") != args.batch:
            sys.exit(f"{OUT} was made with --batch {run.get('batch')}; use the same value or --fresh")
        if run.get("rubric_sha256") != laya_bench.rubric_sha256(topics):
            sys.exit(f"{OUT} was made with a different topic rubric; use --fresh")
    done = {p["id"] for p in run["papers"]}
    todo = [p for p in papers if p["id"] not in done]
    print(f"{len(papers)} papers, {len(done)} already classified, {len(todo)} to go; loading {args.model}")

    agent, resolved_revision, model_load_ms = laya_bench.load_agent(args.model, args.revision, args.device)
    runtime = laya_bench.runtime_metadata(agent)
    if run.get("resolved_revision") not in (None, resolved_revision):
        sys.exit(
            f"{OUT} uses model revision {run['resolved_revision']}; resolved {resolved_revision}. "
            "Pin the old --revision or use --fresh."
        )
    run["resolved_revision"] = resolved_revision
    run.setdefault("model_load_ms", model_load_ms)
    run["runtime"] = runtime
    print(
        f"loaded revision {resolved_revision[:12]} in {model_load_ms / 1000:.1f}s on {runtime['device']}"
    )

    # Compile kernels and fill allocator caches before measured rows.
    if todo:
        _, _, warmup = laya_bench.classify_batch(agent, todo[:args.batch], topics)
        run["warmup_ms"] = round(warmup * 1000, 1)
        print(f"warmup: {run['warmup_ms']:.1f} ms")

    order = {p["id"]: i for i, p in enumerate(papers)}
    for n, paper_batch in enumerate(batches(todo, args.batch), 1):
        try:
            rows, usage, took = laya_bench.classify_batch(agent, paper_batch, topics)
        except Exception as error:
            save(run)
            print(f"batch starting at {paper_batch[0]['id']} failed: {error}", file=sys.stderr)
            return 1
        ms = round(took * 1000, 1)
        run["requests"].append({
            "ids": [paper["id"] for paper in paper_batch],
            "ms": ms,
            "input_tokens": int(usage.get("input_tokens") or 0),
        })
        for row in rows:
            run["papers"].append({**row, "ms": round(ms / len(paper_batch), 1), "cost": 0})
        if n % max(1, args.save_every) == 0:
            run["papers"].sort(key=lambda result: order.get(result["id"], len(order)))
            save(run)
            print(f"  {len(run['papers'])} classified")

    run["papers"].sort(key=lambda result: order.get(result["id"], len(order)))
    save(run)
    summary = run["summary"]
    print(
        f"papers: {summary['papers']}   requests: {summary['requests']} "
        f"({summary['batch']} per request)   model: {summary['model']}"
    )
    print(f"API cost: $0 (local inference)   input tokens processed: {summary['input_tokens']}")
    print(
        f"per paper: median {summary['median_ms_per_paper']:.1f} ms   "
        f"p95 {summary['p95_ms_per_paper']:.1f} ms"
    )
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
