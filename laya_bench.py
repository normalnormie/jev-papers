"""Local Laya adapter shared by the classifier, evaluator, and offline checks."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
QUESTION_ID = "topic"


def build_input(paper: dict, topics: dict) -> tuple[dict, dict]:
    """Build one Laya state and one 24-way choice question for one paper.

    Laya applies every question in a call to the same state and has a 512-token
    context at the English checkpoint's defaults. Keeping one paper per state
    avoids truncating three papers out of the four-paper Jev transport batch.
    """
    state = {"title": paper["title"], "abstract": paper["abstract"]}
    question = {
        "type": "choice",
        "instructions": topics["instructions"],
        "criteria": {t["id"]: t["criterion"] for t in topics["topics"]},
    }
    return state, {QUESTION_ID: question}


def rubric_sha256(topics: dict) -> str:
    """Stable fingerprint for the instructions, topic ids, and criteria."""
    _, questions = build_input({"title": "", "abstract": ""}, topics)
    encoded = json.dumps(questions[QUESTION_ID], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_answer(result: dict, paper: dict) -> dict:
    """Convert Laya's choice response to the row shape used by the benchmark."""
    answer = result["answers"][QUESTION_ID]
    probabilities = answer.get("probabilities") or {}
    ranked = sorted(probabilities.items(), key=lambda item: (-float(item[1]), item[0]))
    runner_up = ranked[1] if len(ranked) > 1 else (None, None)
    return {
        "id": paper["id"],
        "topic": answer["choice"],
        "confidence": round(float(answer.get("confidence") or 0), 4),
        "runner_up": runner_up[0],
        "runner_up_p": round(float(runner_up[1]), 4) if runner_up[1] is not None else None,
    }


def snapshot_path(model: str, revision: str | None = None) -> tuple[Path, str]:
    """Download only the English checkpoint files and return its resolved revision."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        model,
        revision=revision,
        allow_patterns=[
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/*",
            "encoder/*",
        ],
    )).resolve()
    resolved = path.name if path.parent.name == "snapshots" else (revision or "local")
    return path, resolved


def load_agent(model: str = MODEL, revision: str | None = None, device: str | None = None):
    """Resolve an exact Hub snapshot, then construct the local Laya agent."""
    import laya

    path, resolved = snapshot_path(model, revision)
    started = time.perf_counter()
    agent = laya.load(str(path), device=device)
    load_ms = round((time.perf_counter() - started) * 1000, 1)
    return agent, resolved, load_ms


def synchronize(agent: Any) -> None:
    if getattr(getattr(agent, "device", None), "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(agent.device)


def classify_one(agent: Any, paper: dict, topics: dict) -> tuple[dict, dict, float]:
    rows, usage, took = classify_batch(agent, [paper], topics)
    return rows[0], usage, took


def classify_batch(agent: Any, papers: list[dict], topics: dict) -> tuple[list[dict], dict, float]:
    """Run independent paper states together through Laya's native batch dimension."""
    import numpy as np
    import torch
    from laya.common import (
        QTYPES,
        build_sequence,
        collate_items,
        confidence_from_probs,
        render_options,
        temp_bucket,
    )

    synchronize(agent)
    started = time.perf_counter()
    items = []
    internal = []
    for paper in papers:
        state, questions = build_input(paper, topics)
        question = agent._to_internal(questions[QUESTION_ID])
        sequence, markers = build_sequence(
            agent.tok,
            state,
            question,
            int(agent.cfg.get("max_len", 512)),
            int(agent.cfg.get("head_max_len", 192)),
        )
        if len(markers) != len(render_options(question)):
            raise ValueError("topic options exceed head_max_len")
        items.append({"ids": sequence, "markers": markers, "qtype": QTYPES[question["t"]]})
        internal.append(question)

    batch = collate_items([items], agent.tok.pad_token_id)
    with torch.no_grad(), torch.autocast(
        device_type=agent.device.type,
        dtype=agent.dtype,
        enabled=agent.device.type == "cuda",
    ):
        logits, actions = agent.model(
            batch["input_ids"].to(agent.device),
            batch["attention_mask"].to(agent.device),
            batch["marker_pos"].to(agent.device),
            batch["marker_mask"].to(agent.device),
            batch["qtype"].to(agent.device),
        )
    logits = logits.float().cpu().numpy()
    actions = torch.softmax(actions.float(), -1).cpu().numpy()
    synchronize(agent)
    took = time.perf_counter() - started

    rows = []
    for index, (paper, question) in enumerate(zip(papers, internal)):
        option_count = len(items[index]["markers"])
        question_type = QTYPES[question["t"]]
        temperature = agent.temperature_by_options.get(
            temp_bucket(question_type, option_count), agent.temperature[question_type]
        )
        values = logits[index, :option_count] / temperature
        probabilities = np.exp(values - values.max())
        probabilities = probabilities / probabilities.sum()
        keys = list(question["crit"])
        answer = {
            "type": "choice",
            "choice": keys[int(probabilities.argmax())],
            "probabilities": {key: round(float(value), 4) for key, value in zip(keys, probabilities)},
            "confidence": round(confidence_from_probs(probabilities, option_count), 4),
            "action": {"act_probability": round(float(actions[index, 0]), 4)},
        }
        rows.append(parse_answer({"answers": {QUESTION_ID: answer}}, paper))
    usage = {"input_tokens": int(batch["attention_mask"].sum()), "output_tokens": 0}
    return rows, usage, took


def runtime_metadata(agent: Any) -> dict:
    import laya
    import torch
    import transformers
    from laya.common import QTYPES, temp_bucket

    device = str(agent.device)
    bucket = temp_bucket(QTYPES["choice"], 24)
    raw_temperature = agent.temperature_by_options_raw.get(bucket, agent.temperature_raw[QTYPES["choice"]])
    applied_temperature = agent.temperature_by_options.get(bucket, agent.temperature[QTYPES["choice"]])
    return {
        "laya_version": laya.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": device,
        "max_len": int(agent.cfg.get("max_len", 512)),
        "head_max_len": int(agent.cfg.get("head_max_len", 192)),
        "choice_temperature_bucket": bucket,
        "choice_temperature_raw": float(raw_temperature),
        "choice_temperature_applied": float(applied_temperature),
        "choice_temperature_clamped": float(raw_temperature) != float(applied_temperature),
    }
