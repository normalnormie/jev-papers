#!/usr/bin/env python3
"""Real Jev call on the eight fixture papers. Prints each label, the confidence, and what the call cost.

    OPENROUTER_API_KEY=... python3 test/jev_check.py
    python3 test/jev_check.py --key-file path/to/file.env
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jev  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file")
    a = ap.parse_args()
    key = jev.load_key(a.key_file)
    topics = jev.load_topics()
    papers = json.loads((ROOT / "test/fixtures/papers.json").read_text())["papers"]

    # batched, four per request: the path classify.py uses
    total, labels, log = 0.0, {}, []
    for start in range(0, len(papers), 4):
        chunk = papers[start:start + 4]
        rows, usage, took = jev.classify_batch(key, chunk, topics)
        cost = float(usage.get("cost") or 0)
        total += cost
        log.append({"papers": len(chunk), "ms": round(took * 1000, 1), "cost": cost, "input_tokens": usage.get("input_tokens")})
        print(f"request: {len(chunk)} papers, {usage.get('input_tokens')} input tokens, cost ${cost:.6f}, {took * 1000:.0f} ms")
        for p, r in zip(chunk, rows):
            labels[p["id"]] = r["topic"]
            print(f"  {r['topic']:<22} conf {r['confidence']:.2f}  next {r['runner_up'] or '-'} {r['runner_up_p'] or 0:.2f}  {p['title'][:70]!r}")
    print(f"total cost ${total:.6f} for {len(papers)} papers (${total / len(papers):.6f} per paper)")

    # one paper alone, so both request shapes are proven against the real API
    rows, usage, took = jev.classify_batch(key, papers[:1], topics)
    same = "same label as batched" if rows[0]["topic"] == labels[papers[0]["id"]] else "DIFFERENT label from batched"
    print(f"single-paper request: {usage.get('input_tokens')} tokens, cost ${float(usage.get('cost') or 0):.6f}, {took * 1000:.0f} ms -> {rows[0]['topic']} ({same})")

    log.append({"papers": 1, "ms": round(took * 1000, 1), "cost": float(usage.get("cost") or 0), "input_tokens": usage.get("input_tokens")})
    out = ROOT / "data" / "jev_check.json"
    out.write_text(json.dumps({"model": usage.get("model"), "requests": log, "labels": labels, "total_cost_batched": round(total, 8)}, indent=1))
    print("wrote", out)

    ids = {t["id"] for t in topics["topics"]}
    assert set(labels.values()) <= ids, "Jev returned a label that is not in topics.json"
    assert len(labels) == len(papers)
    print("jev_check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
