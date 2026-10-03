# Home Base memory: LoCoMo-Refined reproduction

Complete public-benchmark evaluation of a model-free source-memory adapter. **This is not an official leaderboard entry or a claim of global #1.** Matched competitor evaluation and independent operator reproduction remain in progress.

| Method | Correct / 1,382 | Official Refined score |
| --- | ---: | ---: |
| Home Base full recall | 938 | **67.87%** |
| Plain chronological conversation history | 879 | 63.60% |
| Home Base BM25 top-20 source-turn retrieval | 773 | 55.93% |

All methods use the same AML published answer prompt and Qwen3-14B 8-bit reader/judge, with thinking disabled and temperature zero. The judge prompt, runtime and parser are the unchanged official LoCoMo-Refined release. Answers are generated once; successful answers and judgments are retained on restart rather than replaced by new votes. Data overlap with earlier development is disclosed; this is a public benchmark reproduction, not a pristine held-out test.

Full recall returns source conversations chronologically when they fit 100,000 estimated tokens. Larger histories preserve blocks containing retrieval hits first and fill by BM25. This dataset fit entirely: no evidence was truncated. Source-only deterministic date notes are added during ingestion; Search never constructs answers. This evaluated profile uses no extraction LLM or embeddings.

## Code and evidence

`runtime-export.json` maps 25 exported brain modules to their exact original evaluated bytes. Those hashes match the original execution manifest. Unused private chat/routing code and private Git history are excluded. The public harness adds support for verifying this export without needing the private commit; it does not change memory code or the official scorer.

- `results/manifest.json`: original frozen model, data, prompt, tokenizer and runtime hashes.
- `results/*/predictions.jsonl`, `scored.jsonl`, summaries and gates: every question and official outcome.
- `results/completed_results_20261003.json`: paired comparisons and conversation-cluster intervals.
- `LOCOMO_RCA_20261003.md`: descriptive failure audit, separated from any future method development.
- [Release v0.1.0-locomo-frozen](https://github.com/austindixson/homebase-memory-eval/releases/tag/v0.1.0-locomo-frozen): full supplied contexts, prompts, raw HTTP requests/responses, raw judge outputs, provenance and preparation timings. Ten ordered archive parts total 667,525,468 bytes. Each public asset digest and size was verified against the local archive; concatenation SHA-256 is `b341113a43b57e055dfbb906b898b1196e82e18c130fdef52784092eda8a5a52`. Download and verify with `scripts/download_receipts.py` or use `results/artifact-sha256.json`.

The source data/scorer is [LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined), pinned at `887091190789e8d6760e70b9edd696539923dc4f`. The answer template comes from [AML](https://github.com/AML-memory/agent-memory-leaderboard), pinned at `1b8142bfe0f20f1c5218d6b554aa0012de34e504`. Reader/judge weights are [mlx-community/Qwen3-14B-8bit](https://huggingface.co/mlx-community/Qwen3-14B-8bit), revision `da33cf28f06636847fd9e93e0a03d819b84cb55e`.

## Reproduce from a fresh checkout

Use Python 3.11 or 3.12. Install in an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r deploy/aml/competitive-requirements.txt
```

Download the two upstream repositories without altering their source:

```sh
mkdir -p .runs/upstream
git clone https://github.com/mem-eval-suite/LoCoMo_refined.git .runs/upstream/refined
git -C .runs/upstream/refined checkout 887091190789e8d6760e70b9edd696539923dc4f
git clone https://github.com/AML-memory/agent-memory-leaderboard.git .runs/upstream/aml
git -C .runs/upstream/aml checkout 1b8142bfe0f20f1c5218d6b554aa0012de34e504
```

Download the model tokenizer from the fixed weights revision:

```sh
curl -fL https://huggingface.co/mlx-community/Qwen3-14B-8bit/resolve/da33cf28f06636847fd9e93e0a03d819b84cb55e/tokenizer.json -o .runs/tokenizer.json
```

Its SHA-256 must be `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.

Start the pinned local server using `scripts/serve_qwen.sh /path/to/mlx-serve /path/to/model-snapshot`. Obtain MLX Serve from the [upstream v26.9.5 release](https://github.com/ddalcu/mlx-serve/releases/tag/v26.9.5). The server script checks the actual binary hash; another server/precision is a different reproduction and must be disclosed. Apple Silicon with sufficient unified memory is required; the original host has 128 GB. No model download, launch-agent change, or shared-GPU interruption is automated by this repository.

Prepare a **fresh** output directory. Preparation ingests source-only conversations and constructs evidence; it makes no answer/judge calls:

```sh
.venv/bin/python -m homebase.bench.aml.competitive prepare \
  --runtime-manifest runtime-export.json \
  --refined-root .runs/upstream/refined \
  --aml-root .runs/upstream/aml \
  --tokenizer .runs/tokenizer.json \
  --out .runs/independent-reproduction
```

Then run the candidate with the untouched official scorer:

```sh
.venv/bin/python -m homebase.bench.aml.competitive run \
  --refined-root .runs/upstream/refined \
  --out .runs/independent-reproduction \
  --model-url http://127.0.0.1:11240/v1
```

`--controls` also evaluates the retrieval and plain-context controls. Never copy our response cache into the new run. A completed marker is valid only with all 1,382 unique answers and binary official scores. No unresolved errors may be omitted from the denominator.

Independent reproduction passes the registered gate only if the new candidate score and the original score both exceed the published reference threshold (at least 880/1,382) and differ by at most one percentage point. Publish both outcomes, including failures. A second run by the same agent/operator is an internal reproduction, not an independent verification.

## Competitors

Current public source versions were pinned before the original candidate result. The MemPalace comparison executes its unmodified hybrid session retrieval with a common top-20 limit and MiniLM embeddings, then routes retrieved source evidence through the same frozen reader/scorer. It does not equate native retrieval recall with answer accuracy or claim historical top-10 settings. Other comparator configurations and any unavailable methods must be disclosed. The comparison criteria are in `LOCOMO_REFINED_ACCEPTANCE.md` and `LOCOMO_MATCHED_COMPARATORS_20261003.md`.

## Licenses

Adapter/harness code: Apache-2.0, see `LICENSE`. Benchmark source text, QA and distributed benchmark artifacts: upstream CC BY-NC 4.0, see `results/licenses/LoCoMo-Refined-LICENSE.txt`. MemPalace's code remains in its upstream MIT repository; this project invokes it rather than vendoring or relabeling its implementation. Model and serving software retain their upstream licenses.
