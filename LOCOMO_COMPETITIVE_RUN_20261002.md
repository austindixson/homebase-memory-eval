# Full LoCoMo-Refined run — execution registration

The user authorized the real competitive evaluation on 2026-10-02, after reviewing [the acceptance criteria](LOCOMO_REFINED_ACCEPTANCE.md). This authorizes the benchmark run; the $0 additional Railway constraint remains. The candidate stage uses the existing local Qwen server, with no hosted inference charges or Railway deployment.

## Frozen candidate and evaluation

- Memory runtime: `745609d5de83648d2183e732f34567bd878e2ea3`, model-free Add/Search, full recall enabled, 100,000 estimated-token memory budget, 6,000-character blocks, Search top_k 20.
- Dataset/scorer: `mem-eval-suite/LoCoMo_refined` at `887091190789e8d6760e70b9edd696539923dc4f`, all 1,382 questions / 10 conversations, unmodified official scorer entry point and judge/parser.
- Reader prompt: AML LoCoMo template at `1b8142bfe0f20f1c5218d6b554aa0012de34e504`, no Codex system instructions or suffix. The execution manifest records its checksum.
- Reader and judge weights: `mlx-community/Qwen3-14B-8bit` revision `da33cf28f06636847fd9e93e0a03d819b84cb55e`. This is an explicitly disclosed 8-bit reproduction, not a claim to know the benchmark table's undisclosed serving precision.
- Serving: MLX Serve 26.9.5; binary SHA-256 `f6b32efcbbaa3d2d82baa0948071a5037eef7ded868902ab296ba182749e902e`; launch-script SHA-256 `768f6bc0ade959ac6003af98b084eed231464e1582ba309f7b949f78a9ec6477`. Served context 131,072; YaRN factor 4, original context 32,768; top_p 0.95, top_k 20, KV/decode-attention quantization off; lossless PLD, no drafter. No server settings are changed for this run.
- Requests: temperature 0, thinking explicitly disabled, reader maximum output 2,048. Judge request comes directly from the official runtime; its omitted output limit uses the already configured server default 1,024.
- Every method has the same 100,000 actual-token evidence cap, measured with the checkpoint tokenizer. An oversized result uses a deterministic character prefix in returned rank order. Full prompts must fit served context with output/template reserve. Both speakers' names stay in the evidence text, placed in speaker_1_memories; speaker_2_memories is empty consistently across methods.
- Transcript representation includes the published image URLs, captions and image queries. No image encoder is used. Gold answers and QA evidence annotations stay out of ingestion, retrieval and reader inputs.
- First stage: full-recall candidate. At least 880 correct / 1,382 passes the published-reference gate. Only after that passes, run the full retrieval and raw-full-context controls. Preserve all outcomes; no tuning on this run.
- Public data exposure is disclosed: original LoCoMo has already been used in development. This is a public benchmark reproduction, not a pristine held-out evaluation or an AML official submission.

## Matched competitor stage

Before any candidate quality result, these current public source versions were pinned for the reproduction audit:

| Method | Repository | Commit |
| --- | --- | --- |
| MemOS | https://github.com/MemTensor/MemOS | `a7367d07e55db61099f7b4e2c1108bc5831a24f3` |
| EverMemOS / EverOS | https://github.com/EverMind-AI/EverMemOS | `d2aa9494da062246e665a21e3f045583d81df90f` |
| Mem0 | https://github.com/mem0ai/mem0 | `abb81c88e1f738a8117d8293530fbc31a5ef8fd9` |
| MemPalace | https://github.com/MemPalace/mempalace | `a33fdcc98637f0b2c4194b978fec6439fa2a8b93` |

These are not asserted to be the versions behind the historical table. Their own ingestion/retrieval settings and supported model access must be frozen before matched inference. Do not substitute our BM25 for a competitor, use hosted product endpoints as if they were the open-source implementation, or count an unreproducible baseline as defeated. A candidate pass opens this stage; it does not establish global #1.

## Checkpoint and compute handling

Preparation produces frozen inputs, prompts, source/runtime checksums, tokenizer and manifest. A raw request/response recorder serializes inference. It rewrites only the public model alias to the server's checkpoint ID; official prompts and parsing remain unchanged. Successful identical requests are cached across resumption. Invalid judge JSON is retained as an attempt and left to the official retry behavior. Valid judgments are never replaced by fresh votes.

The official scorer may accept exact answer matches without a judge call and evaluates alternative gold answers using its published logic. Both behaviors are preserved. Scorer restarts consume saved successful HTTP responses; they do not regenerate successful model outputs. Incomplete answers/judgments have no reported overall score.

Shared compute must be scheduled in Ghost128's existing `gpu` lane after prior jobs. Append only this job under the health lock; never change another job, adapter, service configuration, credentials or completion marker. No broad guard/pkill script is used. Source snapshots and output directories belong only to this run.

Local preparation command:

```sh
.venv/bin/python -m homebase.bench.aml.competitive prepare \
  --refined-root /tmp/clm-locomo-refined-verification-20261002 \
  --aml-root /tmp/moon/aml \
  --tokenizer /tmp/clm-qwen3-14b-tokenizer.json \
  --out .runs/locomo-refined-20261002
```

The scheduled runner uses `competitive run --controls`, an immutable source snapshot, its own locked output directory and the unchanged public scorer checkout. Full requests and contexts remain outside Git because they are bulky; the compact execution manifest and status receipts belong in the PR.

Validation before scheduling: 195 AML tests pass, including seven new integrity tests and an integration test using the actual official scorer with synthetic HTTP fixture responses. Queue registration is tested to preserve every other job and service. No synthetic response is used as a benchmark judgment.

## Registered run receipt

The immutable runner snapshot is commit `807424ab66ecd88f45e989fe6d2e47048c81e694` in [PR #10](https://github.com/austindixson/CLM-private/pull/10). Payload SHA-256: `1d5e1db0019e855efbe9331f787d9415dc367fff04321638987ccab9433a35dc`. On Ghost128 the payload, all memory runtime files, all three input files and the public benchmark checkout passed checksum/revision verification. Its isolated Python 3.11 environment uses the pinned requirements; `pip check` and module startup passed.

The job `full-recall-refined-1382-807424a` was appended to the existing `gpu` lane under `health.lock`, with no other queue/service edits. Outputs: `/Users/ghost128/.homebase/aml/runs/full-recall-refined-20261002-807424a`. Source and isolated environment: `/Users/ghost128/.homebase/aml/competitive_807424a`. The normal scheduler starts it after preceding jobs finish; successful HTTP responses and prediction checkpoints survive restarts. The output marker denotes candidate/control evaluation completion, not a claim that all competitor or public-release gates passed.

At registration, two prior GPU jobs remained: `leader-calendar-answers` and `leader-calendar-confirmation`. There were zero candidate answers and no score. Compact registration/status receipts are in `homebase/bench/aml/results/locomo_competitive_20261002/`; the status snapshot is dated, not a live result.

Read current complete-only status without model calls:

```sh
ssh ghost128 /Users/ghost128/.homebase/aml/competitive_807424a/.venv/bin/python \
  /Users/ghost128/.homebase/aml/competitive_807424a/status_remote.py
```

Matched competitor inference is not yet registered. The public source audit found that the current MemOS and Mem0 evaluation clients can use hosted product APIs, which must not be presented as their open-source implementations; EverOS's native pipeline requires separate backbone/decider/provider setup; MemPalace's native public benchmark reports retrieval recall rather than the required judged-answer accuracy. Each requires an audited Add/Search-to-common-reader integration before its matched score is valid. The prepared candidate run proceeds independently of this remaining work.
