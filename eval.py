#!/usr/bin/env python3
"""Check Jev against an LLM judge on a fixed random sample of the classified papers.

The judge gets the same 24 topics, the same criteria and the same title and abstract through
OpenRouter chat completions, and answers with one topic id. Writes data/eval.json.
The run is resumable: papers the same judge model already answered are not asked again.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jev

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "eval.json"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE = "anthropic/claude-opus-5"


def judge_messages(paper: dict, topics: dict) -> list[dict]:
    lines = "\n".join(f"- {t['id']}: {t['criterion']}" for t in topics["topics"])
    system = (
        "You label AI research papers. " + topics["instructions"] + "\n\nTopics:\n" + lines
        + "\n\nAnswer with the topic id only, exactly as written above. No other text."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"}]


def parse_label(text: str, ids: list[str]) -> str | None:
    """The judge is asked for a bare id. Accept it with stray quotes or prose around it, longest id first."""
    cleaned = (text or "").strip().strip("`'\". ").lower()
    if cleaned in ids:
        return cleaned
    hits = [i for i in sorted(ids, key=len, reverse=True) if i in cleaned]
    return hits[0] if hits else None


def ask_judge(key: str, model: str, paper: dict, topics: dict) -> dict:
    body = {"model": model, "messages": judge_messages(paper, topics), "temperature": 0, "max_tokens": 20,
            "reasoning": {"enabled": False},  # a bare label is wanted; thinking tokens would eat max_tokens
            "usage": {"include": True}}
    out, took = jev.post(CHAT_URL, key, body)
    if "choices" not in out:
        raise RuntimeError(f"judge returned no choices: {json.dumps(out)[:300]}")
    text = out["choices"][0]["message"].get("content") or ""
    usage = out.get("usage") or {}
    return {
        "id": paper["id"],
        "judge": parse_label(text, [t["id"] for t in topics["topics"]]),
        "judge_raw": text[:80],
        "judge_ms": round(took * 1000, 1),
        "judge_cost": float(usage.get("cost") or 0),
        "judge_tokens": int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0),
    }


def summarize(sample: list[dict], jev_rows: dict, model: str, seed: int) -> dict:
    rows = [r for r in sample if r.get("judge")]
    agree = [r for r in rows if r["judge"] == jev_rows[r["id"]]["topic"]]
    top2 = [r for r in rows if r["judge"] in (jev_rows[r["id"]]["topic"], jev_rows[r["id"]]["runner_up"])]
    sure = [r for r in rows if jev_rows[r["id"]]["confidence"] >= 0.9]
    sure_agree = [r for r in sure if r["judge"] == jev_rows[r["id"]]["topic"]]
    pairs = collections.Counter((jev_rows[r["id"]]["topic"], r["judge"]) for r in rows if r not in agree)
    jev_ms = [jev_rows[r["id"]]["ms"] for r in rows]
    n = max(1, len(rows))
    return {
        "judge_model": model,
        "seed": seed,
        "sample": len(sample),
        "judged": len(rows),
        "agreement": round(len(agree) / n, 4),
        "agreement_top2": round(len(top2) / n, 4),
        "jev_confident": len(sure),
        "agreement_when_jev_confident": round(len(sure_agree) / max(1, len(sure)), 4),
        "agreement_when_jev_unsure": round((len(agree) - len(sure_agree)) / max(1, len(rows) - len(sure)), 4),
        "jev_cost_per_paper": round(sum(jev_rows[r["id"]]["cost"] for r in rows) / n, 8),
        "judge_cost_per_paper": round(sum(r["judge_cost"] for r in rows) / n, 6),
        "judge_total_cost": round(sum(r["judge_cost"] for r in sample), 4),
        "jev_median_ms": round(statistics.median(jev_ms), 1) if jev_ms else 0,
        "judge_median_ms": round(statistics.median(r["judge_ms"] for r in rows), 1) if rows else 0,
        "judge_p95_ms": round(jev.percentile([r["judge_ms"] for r in rows], 0.95), 1),
        "top_disagreements": [{"jev": a, "judge": b, "count": c} for (a, b), c in pairs.most_common(8)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file")
    ap.add_argument("--judge", default=JUDGE, help="any chat model id on OpenRouter")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    key = jev.load_key(a.key_file)
    topics = jev.load_topics()
    papers = {p["id"]: p for p in json.loads((ROOT / "data" / "papers.json").read_text())}
    jev_rows = {r["id"]: r for r in json.loads((ROOT / "data" / "classified.json").read_text())["papers"]}
    ids = sorted(i for i in jev_rows if i in papers)
    picked = random.Random(a.seed).sample(ids, min(a.sample, len(ids)))

    prior = {}
    if OUT.is_file():
        old = json.loads(OUT.read_text())
        if old.get("summary", {}).get("judge_model") == a.judge:
            prior = {r["id"]: r for r in old.get("sample", []) if r.get("judge")}
    sample = [prior[i] for i in picked if i in prior]
    todo = [i for i in picked if i not in prior]
    print(f"sample {len(picked)} (seed {a.seed}), judge {a.judge}, {len(sample)} already judged, {len(todo)} to ask")

    errors = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(ask_judge, key, a.judge, papers[i], topics): i for i in todo}
        for fut in as_completed(futures):
            try:
                sample.append(fut.result())
            except Exception as err:  # keep what was judged; the next run asks the rest
                errors.append(str(err))
    if errors:
        print(f"{len(errors)} judge calls failed. First error: {errors[0]}", file=sys.stderr)
    if not sample:
        return 1

    order = {i: n for n, i in enumerate(picked)}
    sample.sort(key=lambda r: order[r["id"]])
    for r in sample:
        r["jev"], r["jev_confidence"] = jev_rows[r["id"]]["topic"], jev_rows[r["id"]]["confidence"]
        r["title"] = papers[r["id"]]["title"]
    summary = summarize(sample, jev_rows, a.judge, a.seed)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"summary": summary, "sample": sample}, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)

    s = summary
    print(f"judged: {s['judged']} of {s['sample']}")
    print(f"agreement: {s['agreement']:.0%}   judge pick in Jev's top two: {s['agreement_top2']:.0%}")
    print(f"when Jev confidence >= 0.9 ({s['jev_confident']} papers): {s['agreement_when_jev_confident']:.0%}   below 0.9: {s['agreement_when_jev_unsure']:.0%}")
    print(f"cost per paper:  Jev ${s['jev_cost_per_paper']:.6f}   judge ${s['judge_cost_per_paper']:.6f}   (judge total ${s['judge_total_cost']:.4f})")
    print(f"latency per paper: Jev median {s['jev_median_ms']:.0f} ms   judge median {s['judge_median_ms']:.0f} ms, p95 {s['judge_p95_ms']:.0f} ms")
    print("top disagreements (Jev -> judge):")
    for d in s["top_disagreements"]:
        print(f"  {d['count']:2d}  {d['jev']} -> {d['judge']}")
    print("wrote", OUT, "and", jev.write_site_data())
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
