# Local Laya benchmark

This is a local [Laya](https://github.com/NandhaKishorM/laya) rerun of the Jev Papers benchmark. It classifies the same frozen corpus of 1,000 recent English arXiv AI papers into the same 24 topics, then asks an independent GPT judge to label the same seeded 100-paper sample.

The Laya inference path is fully local. It uses the English [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) model downloaded to the Hugging Face cache; it does not call Jev, OpenRouter, or a hosted Laya endpoint. The GPT judge runs once through Codex CLI with `gpt-5.6-sol` at low reasoning effort.

## Result

The recorded run used Laya 0.3.5 at model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, PyTorch 2.7.1+cu118, Transformers 5.17.0, and a CUDA device.

| Measurement | Result |
|---|---:|
| Papers classified | 1,000 |
| Local forward passes | 1,000, batch size 1 |
| API spend for Laya | $0 |
| Tokens processed by Laya | 475,565 |
| Median inference time per paper | 1,384.5 ms |
| p95 inference time per paper | 1,400.9 ms |
| Model load / warmup | 32.6 s / 2.2 s, excluded from inference timing |
| GPT judge | `gpt-5.6-sol`, low effort, one Codex CLI call |
| Judge usage | 43,715 input, 2,182 output tokens |
| Judge wall time | 90.3 s |

Agreement on the seed-7 sample of 100 papers:

| Classifier compared with the same GPT judge | Exact topic | Judge topic in classifier top two |
|---|---:|---:|
| Laya | **30%** | **40%** |
| Previously committed Jev labels | 88% | 98% |

This is agreement with another model, not measured accuracy. Three checks make the large gap less likely to be an artifact of this particular judge:

- The new GPT judge and the previously committed Claude Opus 5 judge agree on 88 of the same 100 papers.
- Laya agrees with that previous Claude judge on 29%, with Claude's label in Laya's top two on 41%. Those figures closely match the new 30% and 40% results.
- Across all 1,000 papers, Laya and the committed Jev result choose the same topic on 29.6%; Jev's choice appears in Laya's top two on 42.0%.

Laya's output is concentrated: `llm_reasoning`, `theory`, `code`, and `nlp` account for 593 of 1,000 predictions, and `society` is never selected. In the judged sample Laya predicts `code` 17 times while the judge predicts it once; none of those 17 Laya `code` labels agree with the judge. The raw rows are in `data/laya_eval.json` so these errors can be inspected rather than inferred from aggregate scores.

## Limits of this result

The benchmark is a 24-way choice question. Laya's own benchmark notes recommend keeping choice questions under roughly 20 options at default settings because all option text shares a fixed head budget. This run deliberately preserves the original 24-topic task, so it tests Laya just beyond that documented range. The English model uses a 512-token context with `head_max_len = 192`; 386 paper sequences reached the 512-token ceiling and the mean sequence length was 475.6 tokens.

Confidence is not calibrated for this task. The downloaded model stores a temperature of `0.1005828` for the `choice:11+` bucket. Laya 0.3.5 warns that this would distort confidence and clamps it to `0.5`. The run records both values and the clamp flag. Only 49 of all 1,000 predictions, and four of the judged 100, have reported confidence at or above 0.90, so no strong conclusion should be drawn from that subset.

The corpus is English. [Discussion #6](https://huggingface.co/convaiinnovations/laya/discussions/6) reports that routed Laya reproduced the root checkpoint's English results exactly while improving multilingual inputs, so this run uses the root English model directly and does not load the multilingual router.

## Run it

The corpus and original Jev result are already committed. Create an environment, install a PyTorch build appropriate for the machine, then install Laya:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python -r requirements-laya.txt
```

Run the local classifier. `--fresh` discards an existing Laya result; without it, the script resumes and checks that the model revision, batch size, and rubric fingerprint still match.

```bash
USE_TF=0 .venv/bin/python classify_laya.py --fresh --device cuda --batch 1
```

Run the judge with an authenticated Codex CLI installation:

```bash
python3 eval_codex.py --sample 100 --seed 7 --judge gpt-5.6-sol --effort low --fresh
```

The evaluator makes one structured-output call for all 100 papers. This avoids paying Codex CLI's fixed context overhead once per paper. It records token usage and wall time; no dollar cost is asserted because Codex CLI does not return a metered charge for this subscription-backed run.

## Method

`classify_laya.py` gives each paper its own state containing the title and abstract, followed by one choice question containing the unchanged 24 topic criteria from `topics.json`. The exact rubric fingerprint for this run is `7c59bb1f789e11b55507bda6b8442b6bba91721ab0ef7121250e717688f84bfb`.

The adapter batches independent encoded sequences through Laya's native model dimension. This matters because Laya's public multi-question call applies all questions to one shared state; packing several papers into that state would consume the 512-token context before later papers were reached. Batch size 1 was fastest on the measured device. A one-paper adapter result was also checked against `Agent.predict` and produced the same label, confidence, runner-up, and token count.

Timing includes input construction, tokenization, and the synchronized forward pass. It excludes snapshot download, model load, and one warmup pass. The model snapshot is resolved to an immutable Hugging Face revision and recorded in the result.

`eval_codex.py` draws exactly the same `random.Random(7)` sample used by the original evaluation. It sends only the rubric, paper IDs, titles, and abstracts to the judge; it does not reveal either classifier's predictions. The JSON schema restricts answers to the known paper and topic IDs, and the script rejects missing, duplicate, or invalid labels. It then scores both Laya and the committed Jev rows against those same judgments.

## Artifacts and checks

- `data/laya_classified.json`: every local prediction, runner-up, confidence, timing, token count, runtime version, and model revision.
- `data/laya_eval.json`: the complete seeded sample, GPT judgments, Laya and Jev labels, aggregate comparisons, judge timing, and token usage.
- `data/classified.json` and `data/eval.json`: the untouched Jev and Claude baseline artifacts.

Run the offline integrity checks with:

```bash
python3 test/self_check.py
python3 -m py_compile laya_bench.py classify_laya.py eval_codex.py
```

The Laya source checkout used for this run was commit `573e5b62696ba441230cd6be71d593331b5d23af`. The frozen benchmark corpus came from Jev Papers commit `9889d544e648e77d028a5227e5667cb54da93a85`.

## License

MIT
