"""The Jev client shared by classify.py, eval.py and the tests. Standard library only.

Jev is a decision model: you send a state and typed questions, it sends back a choice and a
probability per label. No text comes back.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
URL = os.environ.get("JEV_URL", "https://openrouter.ai/api/alpha/decisions")
MODEL = os.environ.get("JEV_MODEL", "~typesafe/jev-latest")
KEY_NAMES = ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY")


def load_key(path: str | None = None) -> str:
    """Key from the environment, then --key-file, then ./.env. It is never printed."""
    for name in KEY_NAMES:
        if os.environ.get(name):
            return os.environ[name]
    for candidate in (path, ROOT / ".env"):
        if not candidate or not Path(candidate).is_file():
            continue
        for line in Path(candidate).read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in KEY_NAMES and v.strip():
                    return v.strip().strip("'\"")
    raise SystemExit("no API key: set OPENROUTER_API_KEY (or TYPESAFE_API_KEY), put it in .env, or pass --key-file")


def load_topics(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "topics.json").read_text())


def build_request(papers: list[dict], topics: dict) -> dict:
    """One request for a batch of papers: one 24-way choice question per paper."""
    criteria = {t["id"]: t["criterion"] for t in topics["topics"]}
    state, questions = {"papers": {}}, {}
    for i, p in enumerate(papers):
        key = f"p{i}"
        state["papers"][key] = {"title": p["title"], "abstract": p["abstract"]}
        questions[f"{key}_topic"] = {
            "type": "choice",
            "instructions": f"Paper {key}. " + topics["instructions"],
            "criteria": criteria,
        }
    return {"model": MODEL, "state": state, "questions": questions}


def parse_answers(answers: dict, papers: list[dict]) -> list[dict]:
    """Answers for one batch -> one row per paper, in order."""
    rows = []
    for i, p in enumerate(papers):
        a = answers[f"p{i}_topic"]
        probs = a.get("probabilities") or {}
        top = [kv for kv in sorted(probs.items(), key=lambda kv: -kv[1]) if kv[1] > 0]  # a 0.00 runner-up is a tie, not a second opinion
        rows.append({
            "id": p["id"],
            "topic": a["choice"],
            "confidence": round(float(a.get("confidence") or probs.get(a["choice"]) or 0), 4),
            "runner_up": top[1][0] if len(top) > 1 else None,
            "runner_up_p": round(top[1][1], 4) if len(top) > 1 else None,
        })
    return rows


_local = threading.local()  # one warm HTTPS connection per thread: a TLS handshake costs as much as a decision


def post(url: str, key: str, body: dict, timeout: int = 120) -> tuple[dict, float]:
    """POST json, return (response, seconds). HTTP errors are never retried: callers are resumable,
    a failed batch is picked up by the next run. Only a keep-alive socket the server already closed
    is reopened, once."""
    u = urlparse(url)
    data = json.dumps(body).encode()
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "X-Title": "jev-papers"}
    for attempt in range(2):
        conn = getattr(_local, "conn", None)
        fresh = conn is None
        if fresh:
            conn = _local.conn = http.client.HTTPSConnection(u.netloc, timeout=timeout)
        started = time.perf_counter()
        try:
            conn.request("POST", u.path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
        except (http.client.HTTPException, ConnectionError, BrokenPipeError):
            _local.conn = None
            if fresh or attempt:
                raise
            continue
        except OSError:
            _local.conn = None
            raise
        took = time.perf_counter() - started
        if resp.status >= 400:
            raise RuntimeError(f"http {resp.status} from {url}: {raw.decode('utf-8', 'replace')[:400]}")
        return json.loads(raw), took
    raise RuntimeError("unreachable")


def classify_batch(key: str, papers: list[dict], topics: dict) -> tuple[list[dict], dict, float]:
    out, took = post(URL, key, build_request(papers, topics))
    usage = {**(out.get("usage") or {}), "model": out.get("model")}  # the dated model behind the alias
    return parse_answers(out["answers"], papers), usage, took


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else 0.0


def write_site_data() -> Path:
    """Join papers, labels and the eval summary into site/data.js so the page opens from file:// with no server."""
    data = ROOT / "data"
    papers = {p["id"]: p for p in json.loads((data / "papers.json").read_text())}
    run = json.loads((data / "classified.json").read_text())
    ev = json.loads((data / "eval.json").read_text()) if (data / "eval.json").is_file() else {}
    judged = {r["id"]: r["judge"] for r in ev.get("sample", [])}
    rows = []
    for r in run["papers"]:
        p = papers.get(r["id"])
        if not p:
            continue
        rows.append({
            "id": p["id"], "url": p["url"], "title": p["title"], "date": p["date"], "cat": p["category"],
            "authors": p["authors"][:3] + ([f"+{len(p['authors']) - 3}"] if len(p["authors"]) > 3 else []),
            "abstract": p["abstract"][:420] + ("..." if len(p["abstract"]) > 420 else ""),
            "topic": r["topic"], "conf": r["confidence"], "alt": r["runner_up"], "altp": r["runner_up_p"],
            **({"judge": judged[p["id"]]} if p["id"] in judged else {}),
        })
    payload = {
        "topics": [{"id": t["id"], "name": t["name"], "group": t.get("group", ""), "criterion": t["criterion"]} for t in load_topics()["topics"]],
        "summary": run["summary"],
        "eval": ev.get("summary"),
        "papers": rows,
    }
    out = ROOT / "site" / "data.js"
    out.write_text("window.JEV_PAPERS = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    return out
