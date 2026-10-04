# Independent candidate rerun

The user will operate this rerun. Only Home Base full recall is required. Competitor ingestion models, provider accounts and competitor runs are unnecessary.

## Fresh checkout and frozen settings

```sh
git clone https://github.com/austindixson/homebase-memory-eval.git
cd homebase-memory-eval
git checkout v0.1.1-candidate-only
git rev-parse HEAD
```

Record that commit, your name/role, UTC start time and host separately with the run. Follow the [README setup commands](README.md#reproduce-from-a-fresh-checkout): isolated Python environment, fixed upstream repositories, fixed tokenizer, and the hash-checked MLX Serve binary. The memory modules remain byte-identical to the original evaluated candidate, as verified by `runtime-export.json`.

If you need a fresh model snapshot, the installed Hugging Face client can download the exact revision:

```sh
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    'mlx-community/Qwen3-14B-8bit',
    revision='da33cf28f06636847fd9e93e0a03d819b84cb55e',
    cache_dir='.runs/models'))
PY
```

Use the printed snapshot path with `scripts/serve_qwen.sh`. On a shared host, reserve its GPU through the existing scheduler before starting the server or inference. Flash Next has been restored after cancellation of our comparator jobs; it must not be displaced by an uncoordinated rerun. A different binary, reader precision, prompt or model revision is a different configuration and must be reported.

## Candidate only

Use a fresh output directory. Do not copy any published answer, request or judge cache into it. Model weights and source repositories may be reused.

```sh
.venv/bin/python -m homebase.bench.aml.competitive prepare \
  --runtime-manifest runtime-export.json \
  --refined-root .runs/upstream/refined \
  --aml-root .runs/upstream/aml \
  --tokenizer .runs/tokenizer.json \
  --out .runs/independent-reproduction

.venv/bin/python -m homebase.bench.aml.competitive run \
  --refined-root .runs/upstream/refined \
  --out .runs/independent-reproduction \
  --model-url http://127.0.0.1:11240/v1
```

Preparation writes the three registered evidence inputs without inference. The default `run` command evaluates **full recall only**. Do not add `--controls` or competitor method arguments. If interrupted, resume the same command and output directory so successful responses are retained. Keep all failures and retries; do not create repeats to select a better score.

## Check and return the result

```sh
.venv/bin/python scripts/verify_rerun.py \
  --out .runs/independent-reproduction \
  > .runs/independent-rerun-check.json
```

This performs no model calls and does not judge answers. It checks the saved official outcomes: all 1,382 unique IDs, matching predictions, successful binary Refined judgments, frozen questions, input checksums and completion markers. Exit 0 means the numerical gate passes; exit 1 means a complete numerical failure; exit 2 means incomplete or invalid artifacts. Operator independence and runtime provenance must also be checked from the run record.

The original result is **938/1,382**. Passing the registered gate requires **925–951 correct, inclusive**: above the published reference and within one percentage point of the original. A score outside that range is retained and investigated rather than rerun until passing. This range is the existing preregistered tolerance expressed as integer counts, not a new target.

Return the complete `.runs/independent-reproduction` directory, `independent-rerun-check.json`, your operator record, server command/binary hash and actual model snapshot revision. The directory includes source evidence, exact prompts, raw requests/responses, official scorer log, predictions, judgments and summaries. Keep the full receipts even if the check fails. No credentials or unrelated host files are needed.

The rerun is a public benchmark reproduction. It does not submit a leaderboard entry or establish AML rank #1.
