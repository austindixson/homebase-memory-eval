# LoCoMo-Refined acceptance criteria — 2026-10-02

Status: decision rules fixed before new quality runs; execution manifest still needs model revisions, baseline commits and the answer-prompt checksum. No run, deployment, spending or public release is authorized by this document.

## What the existing numbers establish

The prior 73.8% result is 177/240 sampled questions from original LoCoMo, using a GLM reader and GLM judge with AML's published prompts. Its source file has 1,540 non-adversarial questions. It is evidence of improvement over our own retrieval baseline in that experiment, not a LoCoMo-Refined score. The new model-free candidate has no measured answer-quality score. Do not transfer the Jev-backed result to it.

The [LoCoMo-Refined published table](https://github.com/mem-eval-suite/LoCoMo_refined#-evaluation-results) lists MemOS at 63.60%, the highest open-source system identified among its five listed systems. The table re-scores existing predictions; it does not establish a common answer-model generation protocol. It is not a census of every open-source LoCoMo paper. Original LoCoMo and LoCoMo-Refined scores must stay separate.

## Gate 1 — protocol and integrity

Pass only when all conditions hold:

- Evaluate all 1,382 official question IDs from all 10 conversations. Category counts are 213 / 299 / 68 / 802 for categories 1 / 2 / 3 / 4. Ingest the published conversation representation including its image-related text; do not silently drop the 521 questions marked multimodal.
- Pin the benchmark at `887091190789e8d6760e70b9edd696539923dc4f`. Use its unmodified refined judge prompt, runtime and parser through `scripts/run_eval.sh --metrics llm f1 bleu --llm-judge refined`. Judge: `Qwen/Qwen3-14B`, temperature 0, thinking disabled when supported. Record exact checkpoint revision, precision/quantization, serving software and whether thinking was actually disabled; an alias alone is insufficient provenance.
- Preserve gold-answer lists as alternative complete answers using the official scorer. Do not stringify lists or concatenate alternatives into a required set.
- Primary metric: official overall LLM-judged accuracy, expressed as correct / 1,382. BLEU and F1 are secondary and cannot substitute for this score.
- Freeze model-free candidate code at `745609d5de83648d2183e732f34567bd878e2ea3`, with `AML_MEMORY_PROFILE=model-free`, `AML_FULL_RECALL=1`, recall budget 100,000 estimated tokens and block size 6,000 characters. Corrections after evaluation require a new dated protocol and a disclosed new run.
- Generate answers with `Qwen/Qwen3-14B` and AML's published LoCoMo answer prompt pinned at AML commit `1b8142bfe0f20f1c5218d6b554aa0012de34e504`. Freeze the extracted prompt checksum and generation parameters before running; no extra Codex instructions. Fix temperature 0, thinking disabled, max output 2,048 tokens, Search `top_k=20`, and an actual-token evidence cap of 100,000 for every method. Truncate only by a disclosed deterministic policy fixed before generation. If implementation cannot meet these settings, revise the manifest before any benchmark quality result is observed.
- Same question text, reader checkpoint, prompt, output limit, evidence cap and serving settings for every comparison. Each conversation has an isolated store. Add sees conversation text only; Search sees the question only. Neither sees QA gold answers or previous judgments. Retrieved material must be source evidence, never a computed answer.
- Exactly one recorded answer per question per method. Resume transport failures without replacing successful answers; log every retry. Missing answers count as wrong, never disappear from the denominator. Unresolved judge errors make the run incomplete. Never choose the best of multiple generations or judge votes.
- Before the run, archive code/config/checksums and data-exposure history. Original LoCoMo has already been used for development, so the overlapping Refined set is a public benchmark evaluation, not a pristine held-out test. No tuning on its results.

Frozen source SHA-256:

| File | SHA-256 |
| --- | --- |
| `data/raw/locomo_refined.json` | `1aef6da702087d72515d1b9224f0956a2fbab415c11936253bf7d967d3cf8c17` |
| `data/public/questions.jsonl` | `4dd84cf65a28ece0ed6b2a1b7b2700ea639f4e7b2c5e1afcc511e47e81732790` |
| `data/public/conversations.jsonl` | `abbe220013815b1761c07d9ab7d6147ee08fb089f1a8a083fd31188cdcd91444` |
| `src/llm_judge.py` | `a24d3480e003d985175b1889d6de208585fa694cdb1f02a3739f3cda30a9bb2a` |
| `src/llm_judge_runtime.py` | `6fa81ca9b0347293f81666b794788b18e70018dadb5793895a8f0e53437d5790` |

## Gate 2 — ranking and method comparison

Two different conclusions have separate conditions:

1. **Exceeds the published MemOS reference:** Gate 1 passes and the official score exceeds the rounding range of the published 63.60% (at least 880 correct out of 1,382, or 63.68%). A result of 879/1,382 rounds to 63.60% and is not a clear win. Describe this as exceeding a published reference under our disclosed generation conditions. Do not infer that the memory method beats MemOS under matched conditions.
2. **Robust win over the compared open-source methods:** rerun MemOS, EverMemOS, MemPalace and Mem0 under Gate 1's common generation protocol, with their public versions/configs fixed before results. The candidate must have higher overall accuracy than each. For each paired difference, the lower endpoint of a two-sided 98.75% conversation-cluster bootstrap interval must exceed zero (Bonferroni adjustment for four comparisons). Resample 10 whole conversations with replacement 100,000 times, seed 20261002; compute the question-weighted candidate-minus-baseline accuracy in each resample and use percentiles 0.625 and 99.375. Publish all intervals; only 10 conversation clusters means uncertainty can remain large. An unavailable/unreproducible comparator narrows the claim to the methods actually compared; it is not scored as a loss for that comparator.

Also run two attribution controls: the same candidate with only `AML_FULL_RECALL=0`, and raw chronological full-context evidence without memory selection or date-note enrichment. Preserve the earlier material-improvement rule for the retrieval comparison: at least +5 percentage points overall and no category with at least 15 questions worse by more than 5 points. Report conversation-cluster uncertainty for that difference. If raw full context ties or wins, report that result plainly; it limits any claim that the adapter adds value beyond giving the reader the conversation.

Report all per-conversation and per-category results, exact correct counts, token usage, ingestion/search/generation latency, failures and cost. The superiority rule above is our preregistered standard, not an official benchmark requirement. A point-score leaderboard win and statistically supported superiority are distinct outcomes.

## Gate 3 — public reproducibility

Pass only after:

- Release the actual evaluated memory code and evaluation harness in a public repository, at fixed commits, with applicable open-source licenses and upstream attribution. Record any mapping from the private frozen commit to the public export and verify identical evaluated runtime files.
- Publish the execution manifest, commands, dependency lock, checkpoint revisions, exact prompts, retrieved contexts, answer JSONL, raw judge outputs, parsed judgments, official summaries and checksum manifest. Use public benchmark IDs and respect its license; exclude credentials and unrelated private data.
- Have an independent operator rerun the frozen candidate end to end, with the same data and settings. Both candidate runs must exceed the historical reference, and their accuracy must differ by at most 1 percentage point. Publish both, including failures; never select the larger score. A larger difference leaves reproducibility unresolved and triggers investigation, not repeated runs until one passes.
- Submit the reproducible result to the benchmark maintainers when the user authorizes communication. Until accepted, say it is our reproducible evaluation, not an official leaderboard entry.

Permitted final claim after all applicable conditions pass: **“Highest score among [explicit compared open-source methods] on LoCoMo-Refined, under [disclosed reader and protocol].”** A global “best open-source LoCoMo result” requires a broader up-to-date comparison audit and still must name the benchmark version. None of these gates establishes AML overall rank #1; that needs AML's own evaluation.

## Execution constraints

Prepare and verify the runner without model calls first. Actual paid experiments require the user's approval under the handoff. Use RunInfra on this Mac for paid model experiments, at most four concurrent, and never read its key files. Shared ghost128 hardware must be idle and coordinated before any load; do not start its broad process-killing guard casually. Keep additional Railway charges at $0. No deployment is needed for this local benchmark protocol.
