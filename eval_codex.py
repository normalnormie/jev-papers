#!/usr/bin/env python3
"""Judge a seeded Laya sample with one low-effort Codex CLI batch."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jev
import laya_bench

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "laya_eval.json"
MODEL = "gpt-5.6-sol"
EFFORT = "low"


def build_prompt(papers: list[dict], topics: dict) -> str:
    rubric = "\n".join(f"- {topic['id']}: {topic['criterion']}" for topic in topics["topics"])
    inputs = [{"id": paper["id"], "title": paper["title"], "abstract": paper["abstract"]} for paper in papers]
    return (
        "Act as an independent judge for an AI-paper topic-classification benchmark. "
        + topics["instructions"]
        + "\nClassify every paper independently. Use exactly one topic id from the rubric for each paper. "
          "Return every input id exactly once through the required JSON schema. Do not discuss the answers, "
          "use tools, inspect files, or consider how any other model might label them.\n\nTopics:\n"
        + rubric
        + "\n\nPapers (JSON):\n"
        + json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
    )


def output_schema(paper_ids: list[str], topic_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "minItems": len(paper_ids),
                "maxItems": len(paper_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": paper_ids},
                        "topic": {"type": "string", "enum": topic_ids},
                    },
                    "required": ["id", "topic"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def parse_usage(stdout: str) -> dict:
    usage = {}
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or usage
    return {key: int(value or 0) for key, value in usage.items()}


def run_judge(papers: list[dict], topics: dict, model: str, effort: str, timeout: int) -> tuple[list[dict], dict]:
    paper_ids = [paper["id"] for paper in papers]
    topic_ids = [topic["id"] for topic in topics["topics"]]
    prompt = build_prompt(papers, topics)
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="laya-codex-judge-", dir=cache_dir) as temp_dir:
        temp = Path(temp_dir)
        schema_path = temp / "schema.json"
        response_path = temp / "response.json"
        schema_path.write_text(json.dumps(output_schema(paper_ids, topic_ids)))
        command = [
            "codex", "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--model", model,
            "--config", f'model_reasoning_effort="{effort}"',
            "--output-schema", str(schema_path),
            "--output-last-message", str(response_path),
            "--color", "never",
            "--json",
            "-",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        if completed.returncode:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"Codex CLI exited {completed.returncode}: {detail}")
        if not response_path.is_file():
            raise RuntimeError("Codex CLI produced no final response file")
        response = json.loads(response_path.read_text())
        labels = response.get("labels") or []

    by_id = {}
    for label in labels:
        paper_id, topic = label.get("id"), label.get("topic")
        if paper_id in by_id:
            raise ValueError(f"judge returned duplicate id {paper_id}")
        if paper_id not in paper_ids or topic not in topic_ids:
            raise ValueError(f"judge returned invalid label {label}")
        by_id[paper_id] = topic
    missing = [paper_id for paper_id in paper_ids if paper_id not in by_id]
    if missing:
        raise ValueError(f"judge omitted {len(missing)} ids: {missing[:5]}")
    rows = [{"id": paper_id, "judge": by_id[paper_id]} for paper_id in paper_ids]
    run = {
        "paper_ids": paper_ids,
        "ms": elapsed_ms,
        "usage": parse_usage(completed.stdout),
    }
    return rows, run


def summarize(
    sample: list[dict],
    laya_rows: dict,
    model: str,
    effort: str,
    seed: int,
    judge_runs: list[dict],
    jev_rows: dict | None = None,
    previous_judges: dict | None = None,
) -> dict:
    rows = [row for row in sample if row.get("judge")]
    agree = [row for row in rows if row["judge"] == laya_rows[row["id"]]["topic"]]
    top2 = [
        row for row in rows
        if row["judge"] in (laya_rows[row["id"]]["topic"], laya_rows[row["id"]]["runner_up"])
    ]
    confident = [row for row in rows if laya_rows[row["id"]]["confidence"] >= 0.9]
    confident_agree = [row for row in confident if row["judge"] == laya_rows[row["id"]]["topic"]]
    pairs = collections.Counter(
        (laya_rows[row["id"]]["topic"], row["judge"])
        for row in rows if row not in agree
    )
    laya_ms = [laya_rows[row["id"]]["ms"] for row in rows]
    usage = collections.Counter()
    for run in judge_runs:
        usage.update(run.get("usage") or {})
    count = max(1, len(rows))
    summary = {
        "judge_model": model,
        "judge_reasoning_effort": effort,
        "judge_transport": "codex-cli",
        "judge_batch_calls": len(judge_runs),
        "seed": seed,
        "sample": len(sample),
        "judged": len(rows),
        "agreement": round(len(agree) / count, 4),
        "agreement_top2": round(len(top2) / count, 4),
        "laya_confident": len(confident),
        "agreement_when_laya_confident": round(len(confident_agree) / max(1, len(confident)), 4),
        "agreement_when_laya_unsure": round(
            (len(agree) - len(confident_agree)) / max(1, len(rows) - len(confident)), 4
        ),
        "laya_api_cost_usd": 0,
        "laya_median_ms": round(statistics.median(laya_ms), 1) if laya_ms else 0,
        "judge_total_ms": round(sum(run["ms"] for run in judge_runs), 1),
        "judge_usage": dict(usage),
        "top_disagreements": [
            {"laya": laya_topic, "judge": judge_topic, "count": occurrences}
            for (laya_topic, judge_topic), occurrences in pairs.most_common(8)
        ],
    }
    if jev_rows:
        jev_comparable = [row for row in rows if row["id"] in jev_rows]
        jev_agree = [row for row in jev_comparable if row["judge"] == jev_rows[row["id"]]["topic"]]
        jev_top2 = [
            row for row in jev_comparable
            if row["judge"] in (jev_rows[row["id"]]["topic"], jev_rows[row["id"]]["runner_up"])
        ]
        baseline_count = max(1, len(jev_comparable))
        summary.update({
            "jev_comparable": len(jev_comparable),
            "jev_agreement_same_judge": round(len(jev_agree) / baseline_count, 4),
            "jev_agreement_top2_same_judge": round(len(jev_top2) / baseline_count, 4),
        })
        all_comparable = sorted(set(laya_rows) & set(jev_rows))
        summary.update({
            "laya_jev_comparable_all": len(all_comparable),
            "laya_jev_agreement_all": round(
                sum(laya_rows[paper_id]["topic"] == jev_rows[paper_id]["topic"] for paper_id in all_comparable)
                / max(1, len(all_comparable)), 4
            ),
            "jev_pick_in_laya_top2_all": round(
                sum(
                    jev_rows[paper_id]["topic"]
                    in (laya_rows[paper_id]["topic"], laya_rows[paper_id]["runner_up"])
                    for paper_id in all_comparable
                ) / max(1, len(all_comparable)), 4
            ),
        })
    if previous_judges:
        comparable = [row for row in rows if row["id"] in previous_judges]
        same = [row for row in comparable if row["judge"] == previous_judges[row["id"]]]
        laya_same = [
            row for row in comparable
            if laya_rows[row["id"]]["topic"] == previous_judges[row["id"]]
        ]
        laya_top2 = [
            row for row in comparable
            if previous_judges[row["id"]]
            in (laya_rows[row["id"]]["topic"], laya_rows[row["id"]]["runner_up"])
        ]
        summary.update({
            "previous_judge_comparable": len(comparable),
            "judge_agreement_with_previous": round(len(same) / max(1, len(comparable)), 4),
            "laya_agreement_previous_judge": round(len(laya_same) / max(1, len(comparable)), 4),
            "laya_agreement_top2_previous_judge": round(len(laya_top2) / max(1, len(comparable)), 4),
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default=MODEL)
    parser.add_argument("--effort", default=EFFORT, choices=("low", "medium", "high", "xhigh"))
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print prompt size without calling Codex")
    args = parser.parse_args()

    topics = jev.load_topics()
    papers = {paper["id"]: paper for paper in json.loads((ROOT / "data" / "papers.json").read_text())}
    laya_run = json.loads((ROOT / "data" / "laya_classified.json").read_text())
    laya_rows = {row["id"]: row for row in laya_run["papers"]}
    jev_path = ROOT / "data" / "classified.json"
    jev_rows = {
        row["id"]: row for row in json.loads(jev_path.read_text())["papers"]
    } if jev_path.is_file() else {}
    previous_path = ROOT / "data" / "eval.json"
    previous_eval = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    previous_judges = {
        row["id"]: row["judge"] for row in previous_eval.get("sample", []) if row.get("judge")
    }
    ids = sorted(paper_id for paper_id in laya_rows if paper_id in papers)
    picked = random.Random(args.seed).sample(ids, min(args.sample, len(ids)))

    prior, judge_runs = {}, []
    if OUT.is_file() and not args.fresh:
        old = json.loads(OUT.read_text())
        summary = old.get("summary") or {}
        compatible = (
            summary.get("judge_model") == args.judge
            and summary.get("judge_reasoning_effort") == args.effort
            and summary.get("seed") == args.seed
            and summary.get("sample") == len(picked)
        )
        if compatible:
            prior = {row["id"]: row for row in old.get("sample", []) if row.get("judge")}
            judge_runs = old.get("judge_runs") or []
    todo = [paper_id for paper_id in picked if paper_id not in prior]
    print(
        f"sample {len(picked)} (seed {args.seed}), judge {args.judge} at {args.effort} effort, "
        f"{len(prior)} already judged, {len(todo)} to ask"
    )
    if args.dry_run:
        prompt = build_prompt([papers[paper_id] for paper_id in todo], topics)
        print(f"prompt characters: {len(prompt):,}; paper ids valid: {len(set(picked)) == len(picked)}")
        return 0

    sample = [prior[paper_id] for paper_id in picked if paper_id in prior]
    if todo:
        judged, judge_run = run_judge(
            [papers[paper_id] for paper_id in todo], topics, args.judge, args.effort, args.timeout
        )
        sample.extend(judged)
        judge_runs.append(judge_run)

    order = {paper_id: index for index, paper_id in enumerate(picked)}
    sample.sort(key=lambda row: order[row["id"]])
    for row in sample:
        prediction = laya_rows[row["id"]]
        row["laya"] = prediction["topic"]
        row["laya_confidence"] = prediction["confidence"]
        if row["id"] in jev_rows:
            row["jev"] = jev_rows[row["id"]]["topic"]
            row["jev_confidence"] = jev_rows[row["id"]]["confidence"]
        row["title"] = papers[row["id"]]["title"]
    summary = summarize(
        sample, laya_rows, args.judge, args.effort, args.seed, judge_runs, jev_rows, previous_judges
    )
    if previous_eval.get("summary", {}).get("judge_model"):
        summary["previous_judge_model"] = previous_eval["summary"]["judge_model"]
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"summary": summary, "judge_runs": judge_runs, "sample": sample}, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)

    print(f"judged: {summary['judged']} of {summary['sample']}")
    print(f"agreement: {summary['agreement']:.0%}   judge pick in Laya's top two: {summary['agreement_top2']:.0%}")
    if "jev_agreement_same_judge" in summary:
        print(
            f"same GPT judge vs committed Jev labels: {summary['jev_agreement_same_judge']:.0%} "
            f"(top two {summary['jev_agreement_top2_same_judge']:.0%})"
        )
    print(
        f"when Laya confidence >= 0.9 ({summary['laya_confident']} papers): "
        f"{summary['agreement_when_laya_confident']:.0%}; below 0.9: {summary['agreement_when_laya_unsure']:.0%}"
    )
    print(f"Laya median: {summary['laya_median_ms']:.1f} ms; judge batch total: {summary['judge_total_ms'] / 1000:.1f}s")
    print("top disagreements (Laya -> judge):")
    for disagreement in summary["top_disagreements"]:
        print(f"  {disagreement['count']:2d}  {disagreement['laya']} -> {disagreement['judge']}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
