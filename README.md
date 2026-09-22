# Jev Papers

Classifies 1,000 recent arXiv AI papers into 24 topics with one Jev decision per paper, then checks those labels against an LLM judge on the same papers.

Three scripts and one static page. `fetch.py` pulls the papers from the public arXiv API, `classify.py` asks Jev one 24-way choice question per paper, `eval.py` asks a strong chat model the same question on a seeded sample of 100 and reports where the two disagree. `site/index.html` is a topic map of the result: tile area is the paper count, click a topic to list its papers, search everything, every paper shows Jev's confidence.

![preview](docs/preview.png)

## Measured

All numbers below are from the run committed in `data/`. Papers: 1,000 submissions to cs.AI, cs.LG, cs.CL and cs.CV from 2026-09-16 to 2026-09-18.

| | Jev (`typesafe/jev-1.13-20260917`) | LLM judge (`anthropic/claude-opus-5`) |
|---|---|---|
| Papers | 1,000 | 100 (seeded sample) |
| Total cost | $0.0585 | $0.8940 |
| Cost per paper | $0.000058 | $0.008940 |
| Latency per paper | 56.9 ms median, 90.5 ms p95 | 1,923 ms median, 2,810 ms p95 |

Jev's latency per paper is the request time divided by the four papers in the request. Per request it is 227.8 ms median and 362.1 ms p95, over 250 requests, 1,392,141 input tokens. The judge gets one paper per request and answers with a bare label, extended reasoning off.

Agreement on the 100 sampled papers:

- Jev and the judge pick the same topic on 85.
- The judge's pick is Jev's first or second choice on 95.
- When Jev's confidence is 0.90 or higher (62 of the 100), they agree on 98%. Below 0.90 they agree on 63%.
- 14 of the 15 disagreements are papers where Jev itself was under 0.90. The confidence number is a usable signal for which labels to double check.
- No disagreement pair repeats. The 15 are 15 different pairs, mostly papers that sit between two topics: a diffusion sampling paper that Jev files under image generation and the judge under theory, an RL paper for telescope optics that Jev files under reinforcement learning and the judge under AI for science. The judge leans toward `theory` (4 of 15) and `agents` (3 of 15).

The judge is another model, not ground truth. 85% is agreement, not accuracy.

## Run it

You need Python 3.10 or newer and an OpenRouter API key with access to `~typesafe/jev-latest`. No packages to install.

```
export OPENROUTER_API_KEY=...       # or put it in .env, or pass --key-file
python3 fetch.py                    # under a minute, arXiv asks for 3 seconds between requests
python3 classify.py                 # about 6 cents
python3 eval.py                     # about 90 cents with the default judge
```

Then open `site/index.html` in a browser. It reads `site/data.js`, which `classify.py` and `eval.py` rewrite, so it works from disk with no server.

`fetch.py` keeps its result in `data/papers.json` and does nothing on a second run unless you pass `--refresh`. `classify.py` and `eval.py` are resumable: a failed request is reported and the next run picks up only what is missing. `eval.py --judge <model id>` takes any chat model on OpenRouter, `--seed` and `--sample` change the sample.

## How it works

The 24 topics live in `topics.json`, one line of criterion each. Edit them and run `classify.py --fresh` to get your own map. `classify.py` packs four papers into one request, one `choice` question per paper, six requests in flight:

```json
{
  "model": "~typesafe/jev-latest",
  "state": {
    "papers": {
      "p0": { "title": "GALA: Geometry-Aware Latent Action Modeling for ...", "abstract": "..." },
      "p1": { "title": "...", "abstract": "..." }
    }
  },
  "questions": {
    "p0_topic": {
      "type": "choice",
      "instructions": "Paper p0. Which single research topic fits this AI paper best? Judge by the main contribution described in the title and abstract, not by a method it only uses along the way.",
      "criteria": {
        "llm_reasoning": "Reasoning in language models: chain of thought, math and logic problem solving, ...",
        "agents": "LLM agents that act over many steps: tool calling, web or computer use, ...",
        "...": "22 more"
      }
    },
    "p1_topic": { "...": "same question for p1" }
  }
}
```

It goes to `POST https://openrouter.ai/api/alpha/decisions`. Jev answers with a choice and a probability per label. No text comes back, which is why it is fast and cheap:

```json
{
  "model": "typesafe/jev-1.13-20260917",
  "answers": {
    "p1_topic": {
      "type": "choice",
      "choice": "medical",
      "confidence": 0.94,
      "probabilities": { "medical": 0.94, "time_series": 0.03, "llm_reasoning": 0.01, "graphs_tabular": 0.01, "...": 0 }
    }
  },
  "usage": { "input_tokens": 5726, "output_tokens": 906, "cost": 0.000240492 }
}
```

Cost and latency are recorded per request from that response and from the clock around the HTTP call, on a warm connection per worker. Nothing is estimated. In the last `test/jev_check.py` run (`data/jev_check.json`) a warm request with four papers took 189 ms and a request with one paper took 231 ms. A batch of four is no slower than a single paper, so batching cuts the time per paper by about four. The criteria are repeated for every question, so it barely moves the cost: $0.000060 per paper batched, $0.000070 alone.

`eval.py` draws 100 paper ids with `random.Random(7)`, sends the judge the same instructions, the same 24 criteria and the same title and abstract through `POST https://openrouter.ai/api/v1/chat/completions`, and asks for the topic id only. It writes every judged paper with both labels to `data/eval.json`. On the site those papers carry a badge: "judge agrees" or the judge's topic.

## Verify it yourself

```
python3 test/self_check.py          # offline: feed parsing, batching, request shape, answer parsing, topics.json sanity
python3 test/jev_check.py           # real Jev call on 8 fixture papers, prints labels, confidence and cost
python3 test/render_check.py        # needs playwright: clicks through the page, writes docs/preview.png and docs/preview-mobile.png
```

`test/fixtures/papers.json` holds eight real papers from the run, one per topic across eight topics. The last `jev_check.py` run here cost $0.000480 for the eight papers, and the single-paper request returned the same label as the batched one.

Every number in this README can be read back from `data/classified.json` and `data/eval.json` (key `summary` in both, the per-paper rows next to it) and `data/jev_check.json`.

## Laya comparison

The same frozen corpus and topic rubric were also run through the local English Laya 0.3.5 model. On the seed-7 sample, Laya agrees with a low-effort `gpt-5.6-sol` judge on 30% of exact labels and 40% of top-two labels; the committed Jev labels reach 88% and 98% against that same judge. See the [full Laya benchmark](docs/laya-benchmark.md) for the method, raw artifacts, reproducibility instructions and limitations of applying Laya's default configuration to a 24-choice task.

## Why

[@nutlope classified 1,018 AI research papers with Jev](https://x.com/nutlope/status/2100426999546184123) and put the result up at 1kpapers.com: $0.08 total, 256 ms median per paper, 24 topics. The pipeline code was not published, and the comparison against an LLM judge was mentioned as still running. This is an open rebuild of both: the pipeline end to end, and the eval with its raw rows. The papers, the topics and the numbers here are our own run, not his.

## License

MIT
