# Matched comparator execution registration — 2026-10-03

This continues the authorized competitive evaluation, without changing the frozen candidate or its completed answers/judgments. The original acceptance rules in `LOCOMO_REFINED_ACCEPTANCE.md` remain authoritative. All comparator results use the same 1,382 questions, answer prompt, Qwen3-14B 8-bit checkpoint, temperature, thinking setting, output limit, evidence cap, and unmodified official scorer/parser.

## MemPalace configuration, frozen before answer/judge inference

- Public repository `MemPalace/mempalace`, commit `a33fdcc98637f0b2c4194b978fec6439fa2a8b93`.
- Execute its unmodified `benchmarks/locomo_bench.py` with `--mode hybrid --granularity session --top-k 20 --embed-model default`; LLM reranking disabled. This is its public hybrid session method with the common 20-result limit, not a claim to reproduce its historical top-10 table configuration.
- ChromaDB 1.5.4; native default all-MiniLM-L6-v2 ONNX embedding archive SHA-256 `913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3`; ONNX Runtime 1.30.0, NumPy 2.5.3. Full local dependency lock in `deploy/aml/mempalace-requirements.txt`.
- Supply only source conversations, with the same image URLs, captions and query text available to the candidate. No source summaries or derived date notes. Native indexing uses its own speaker-said corpus formatting. The benchmark-native QA input contains question/category only: gold answers and QA evidence annotations are omitted. Its retrieval-recall printout is consequently meaningless and is not used or reported as a score.
- Native selected session IDs map to their exact native corpus text and source timestamps. Evidence is rendered with the same dated-bullet wrapper and passed to the common reader; selection and ordering remain native. Prepared artifacts include every question exactly once; no evidence is truncated, maximum 33,809 actual evidence tokens.
- Native source hash, preprocessing policy and input hash are frozen in `competitor-registration.json`. Embedding provenance is separately frozen before any answer/judge inference. Every final reader input/prompt is hashed in `manifest.json`.
- Comparison rule: full recall must have higher primary accuracy and a positive lower endpoint of the previously frozen paired 98.75% conversation-cluster interval (100,000 whole-conversation resamples, seed 20261002, question-weighted differences, percentiles 0.625/99.375). Publish the outcome even if MemPalace wins. No tuning or selection among baseline configurations after scoring.

## Remaining comparators and reproduction

MemOS, Mem0 and EverOS require extraction/provider/service setup beyond this no-LLM retrieval comparator. Their source commits remain the ones fixed before candidate quality was observed. Hosted product APIs do not substitute for their public code. A supported local model configuration must be disclosed as a configuration change from published provider defaults, never as a historical-score reproduction. No unavailable method is assigned a losing score.

The public export consists of the transitive import closure of the original evaluated model-free Add/Search, with 25 brain module hashes identical to the original candidate and original execution manifest. Public preparation verifies that mapping instead of requiring inaccessible private Git history. A rerun by this agent can check the clean export internally; it does not satisfy the independent-operator gate. Release requests, contexts, predictions, parsed scores and provenance before requesting an independent operator. Maintainer communication remains a separate action requiring explicit instruction.

Additional Railway and hosted-model charges remain zero. Local GPU jobs are serialized through the existing Ghost128 queue, use their own immutable snapshots and output directories, and resume saved successful responses without new judge votes. Flash Next may be paused only while unused and restored by its existing queue watcher afterward.
