#!/usr/bin/env python3
"""Offline self-check: no network, no key. Feed parsing, batching, request shape, answer parsing, topic file sanity."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import classify  # noqa: E402
import classify_laya  # noqa: E402
import eval as judge_eval  # noqa: E402
import eval_codex  # noqa: E402
import fetch  # noqa: E402
import jev  # noqa: E402
import laya_bench  # noqa: E402

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2609.01234v2</id>
    <published>2026-09-17T17:59:01Z</published>
    <title>A Title
      Broken Over Two Lines</title>
    <summary>  First line.
      Second line.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
  <entry><id>http://arxiv.org/api/errors#bad</id><title>Error</title><summary>bad</summary></entry>
</feed>"""


def main() -> int:
    # fetch: Atom -> paper, whitespace collapsed, version stripped from the id, error entries skipped
    papers = fetch.parse_feed(FEED)
    assert len(papers) == 1, papers
    p = papers[0]
    assert p["id"] == "2609.01234" and p["url"] == "https://arxiv.org/abs/2609.01234v2", p
    assert p["title"] == "A Title Broken Over Two Lines" and p["abstract"] == "First line. Second line.", p
    assert p["authors"] == ["Ada Lovelace", "Alan Turing"] and p["date"] == "2026-09-17" and p["category"] == "cs.LG", p

    # topics.json: 24 unique ids, a name and a real criterion each
    topics = jev.load_topics()
    ids = [t["id"] for t in topics["topics"]]
    assert len(ids) == 24 and len(set(ids)) == 24, len(ids)
    for t in topics["topics"]:
        assert t["id"].replace("_", "").isalnum() and t["id"] == t["id"].lower(), t["id"]
        assert t["name"].strip() and len(t["criterion"]) > 30, t
    # no id may be a substring of another, the judge parser relies on it
    assert not [(a, b) for a in ids for b in ids if a != b and a in b], "one topic id contains another"

    # batching
    assert classify.batches(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert classify.batches([], 4) == []
    assert classify_laya.batches(list(range(5)), 4) == [[0, 1, 2, 3], [4]]

    # request shape: one choice question per paper, all 24 criteria on each
    fixtures = json.loads((ROOT / "test/fixtures/papers.json").read_text())["papers"]
    assert len(fixtures) == 8
    req = jev.build_request(fixtures[:4], topics)
    assert set(req) == {"model", "state", "questions"}
    assert list(req["state"]["papers"]) == ["p0", "p1", "p2", "p3"]
    assert list(req["questions"]) == ["p0_topic", "p1_topic", "p2_topic", "p3_topic"]
    q = req["questions"]["p2_topic"]
    assert q["type"] == "choice" and q["instructions"].startswith("Paper p2.") and list(q["criteria"]) == ids
    assert req["state"]["papers"]["p2"] == {"title": fixtures[2]["title"], "abstract": fixtures[2]["abstract"]}

    # answer parsing: choice, confidence, runner-up, order kept
    answers = {
        "p0_topic": {"type": "choice", "choice": "agents", "confidence": 0.7, "probabilities": {"agents": 0.7, "code": 0.2, "nlp": 0.1}},
        "p1_topic": {"type": "choice", "choice": "theory", "confidence": 1, "probabilities": {"theory": 1, "science": 0}},
    }
    rows = jev.parse_answers(answers, fixtures[:2])
    assert [r["id"] for r in rows] == [fixtures[0]["id"], fixtures[1]["id"]]
    assert rows[0] == {"id": fixtures[0]["id"], "topic": "agents", "confidence": 0.7, "runner_up": "code", "runner_up_p": 0.2}, rows[0]
    assert rows[1]["topic"] == "theory" and rows[1]["confidence"] == 1.0 and rows[1]["runner_up"] is None

    # judge label parsing
    assert judge_eval.parse_label("robotics", ids) == "robotics"
    assert judge_eval.parse_label(' "Safety_Security". ', ids) == "safety_security"
    assert judge_eval.parse_label("The best fit is vision_3d.", ids) == "vision_3d"
    assert judge_eval.parse_label("no idea", ids) is None and judge_eval.parse_label("", ids) is None
    messages = judge_eval.judge_messages(fixtures[0], topics)
    assert all(i in messages[0]["content"] for i in ids) and fixtures[0]["title"] in messages[1]["content"]

    # Local Laya shape: one paper in state, same instructions and criteria, stable rubric fingerprint.
    state, questions = laya_bench.build_input(fixtures[0], topics)
    assert state == {"title": fixtures[0]["title"], "abstract": fixtures[0]["abstract"]}
    assert list(questions) == ["topic"]
    assert questions["topic"]["instructions"] == topics["instructions"]
    assert list(questions["topic"]["criteria"]) == ids
    assert len(laya_bench.rubric_sha256(topics)) == 64
    laya_result = {
        "answers": {"topic": {"choice": "agents", "confidence": 0.7,
                               "probabilities": {"agents": 0.7, "code": 0.2, "nlp": 0.1}}}
    }
    assert laya_bench.parse_answer(laya_result, fixtures[0]) == {
        "id": fixtures[0]["id"], "topic": "agents", "confidence": 0.7,
        "runner_up": "code", "runner_up_p": 0.2,
    }

    # Codex judge prompt and schema include every input and constrain all outputs to known ids.
    prompt = eval_codex.build_prompt(fixtures[:2], topics)
    assert all(topic_id in prompt for topic_id in ids)
    assert all(paper["id"] in prompt and paper["title"] in prompt for paper in fixtures[:2])
    schema = eval_codex.output_schema([paper["id"] for paper in fixtures[:2]], ids)
    labels = schema["properties"]["labels"]
    assert labels["minItems"] == labels["maxItems"] == 2
    assert labels["items"]["properties"]["topic"]["enum"] == ids
    usage = eval_codex.parse_usage(
        '{"type":"turn.started"}\n'
        '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}\n'
    )
    assert usage == {"input_tokens": 10, "output_tokens": 2}

    # stats
    assert jev.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10 and jev.percentile([], 0.5) == 0.0

    # published data, if present: every label is a known topic
    out = ROOT / "data" / "classified.json"
    if out.is_file():
        run = json.loads(out.read_text())
        assert all(r["topic"] in ids for r in run["papers"]), "unknown topic in data/classified.json"
        assert len({r["id"] for r in run["papers"]}) == len(run["papers"]) == run["summary"]["papers"]
        print(f"data/classified.json: {len(run['papers'])} papers, all labels valid")

    laya_out = ROOT / "data" / "laya_classified.json"
    if laya_out.is_file():
        run = json.loads(laya_out.read_text())
        result_ids = [row["id"] for row in run["papers"]]
        request_ids = [paper_id for request in run["requests"] for paper_id in request["ids"]]
        assert run["rubric_sha256"] == laya_bench.rubric_sha256(topics)
        assert all(row["topic"] in ids and row["runner_up"] in ids for row in run["papers"])
        assert len(set(result_ids)) == len(result_ids) == run["summary"]["papers"]
        assert request_ids == result_ids
        assert run["summary"]["requests"] == len(run["requests"])
        print(f"data/laya_classified.json: {len(result_ids)} papers, all labels valid")

    laya_eval_out = ROOT / "data" / "laya_eval.json"
    if laya_eval_out.is_file():
        evaluation = json.loads(laya_eval_out.read_text())
        summary = evaluation["summary"]
        assert summary["judged"] == summary["sample"] == len(evaluation["sample"])
        assert all(row["judge"] in ids and row["laya"] in ids for row in evaluation["sample"])
        sample_ids = [row["id"] for row in evaluation["sample"]]
        assert len(set(sample_ids)) == len(sample_ids)
        classified_ids = sorted(row["id"] for row in run["papers"])
        expected = random.Random(summary["seed"]).sample(classified_ids, summary["sample"])
        assert sample_ids == expected
        judged_ids = [paper_id for judge_run in evaluation["judge_runs"] for paper_id in judge_run["paper_ids"]]
        assert sorted(judged_ids) == sorted(sample_ids)
        print(f"data/laya_eval.json: {len(evaluation['sample'])} judgments, all labels valid")

    print("self_check: all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
