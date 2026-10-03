# LoCoMo-Refined descriptive root-cause audit, 2026-10-03

Full recall scored 938/1,382 (67.87%); retrieval scored 773/1,382 (55.93%). These are complete primary official scorer outputs from frozen runner `807424ab66ecd88f45e989fe6d2e47048c81e694`, not AML platform scores. This audit reads saved evidence and outcomes only. No new answers, judge votes, code changes to the evaluated method, or held-out tuning were performed. The raw chronological control is still pending and its results are not used here.

The paired outcomes are more informative than the net 165-answer gain: both methods succeed on 629 questions; full recall alone succeeds on 309; retrieval alone succeeds on 144; both fail on 300. Long context therefore has a substantial tradeoff, rather than strictly improving every answer.

## Search: lexical ranking and disconnected turns

The evaluated open-source-compatible profile is **model-free BM25**, not the usual Jev/hybrid search. `homebase/brain/aml_model_free.py::retrieve_sources` ranks source turns independently by `_relevance_order` in `aml_adapter.py`, takes the first 20, and uses the default `window=0`. It has no embedding similarity, entity-aware filtering, question decomposition, or neighborhood expansion. A reply may answer a question without repeating its vocabulary; the preceding turn supplies the missing subject.

Annotation coverage was measured by checking whether each annotated evidence message's complete text occurs in the supplied context after casefolding and whitespace normalization. This is a coverage proxy: annotations can be incomplete, repeated text can match another turn, and having every annotated text is not proof that all contextual or image information needed to answer is present. Six questions have no usable annotated text; they are reported as unknown, not successes or failures of retrieval.

| Retrieval evidence coverage | Questions | Correct | Wrong |
| --- | ---: | ---: | ---: |
| All annotated text spans | 839 | 644 | 195 |
| Some annotated text spans | 159 | 50 | 109 |
| No annotated text spans | 378 | 75 | 303 |
| Unknown | 6 | 4 | 2 |

At least one annotated span is omitted on 537/1,376 assessable questions (39.03%). Among the 609 retrieval failures, 412 omit some or all annotated spans, 195 include all spans, and two have unknown coverage. The omission group is not an exact causal attribution: 125 questions with incomplete annotation coverage are still answered correctly.

Concrete rank traces against the frozen source store:

- `conv-26#q0069`, Caroline's summer plans: the answer turn, researching adoption agencies, ranks **342**. The preceding turn explicitly asks about summer plans. Search supplies the question-like turn but omits the reply; the reader answers with Melanie's camping plans instead. This demonstrates both turn fragmentation and speaker confusion.
- `conv-26#q0038`, Melanie's pet names: the Bailey/Oliver turn ranks **18**, while the Luna/Oliver turn ranks **81**. Retrieval answers Oliver and Bailey and misses Luna. Full recall answers Luna and Oliver and misses Bailey despite seeing both turns.
- `conv-26#q0079`, family camping activities: the supporting turn ranks **134**. It says explored nature, roasted marshmallows, and went on a hike. Retrieval and full recall both answer with marshmallows, stories and company; the latter has the complete supporting text but still substitutes a plausible generic camping answer.

Only 36/212 assessable multi-hop questions have all annotated spans in retrieval. The method does not explicitly collect all pieces needed for a multi-fact answer.

## Full recall: evidence delivery is not reliable extraction

Every assessable full-recall question includes all annotated text spans: **1,376/1,376**. Neither method truncated any evidence. Full recall supplies 15,380–41,069 actual evidence tokens, mean 30,001; retrieval supplies 991–2,712, mean 1,506. These observations rule out the configured evidence cap as the cause of these failures. They do not prove visual sufficiency: this run supplies image URLs, captions and query text to a text-only reader, not image pixels.

Full recall still fails **444** questions: 443 with all annotated text present and one with unknown annotation coverage. Confirmed examples show:

- **Incomplete aggregation:** `conv-26#q0045` includes both violin and clarinet in both contexts, but both answers mention only clarinet. `conv-26#q0038` drops the later-added pet Bailey in full recall.
- **Wrong event/date anchor:** `conv-26#q0017` includes the explicit note `Yesterday = 5 July 2023`. Retrieval answers 5 July; full recall answers 2 July.
- **Time representation mismatch:** `conv-26#q0001` returns “Last year” for a gold of “2022,” despite the year being present in the time note. The official refined prompt does not allow relative/absolute equivalence. Other examples give a session date instead of the week in which the event happened.
- **Selecting a competing event or generic completion:** `conv-26#q0069` full recall selects an August art show instead of researching adoption agencies. `conv-26#q0079` misses hiking/nature and supplies other camping activities.
- **Inference failure:** `conv-42#q0010` asks Nate's nickname for Joanna; the source says “Hey Jo,” but full recall answers that no nickname is provided.

These examples establish failures after evidence delivery, but do not isolate how much is caused by model capacity, long-context attention, evidence formatting, or generation settings. The 144 retrieval-only successes are consistent with focused context helping; a different model/precision/thinking configuration has not been tested under this frozen protocol. Quantization is not an established cause.

| Question type | Retrieval correct | Full recall correct |
| --- | ---: | ---: |
| Multi-hop | 66/213 (30.99%) | 107/213 (50.23%) |
| Temporal | 140/299 (46.82%) | 145/299 (48.49%) |
| Open-domain | 31/68 (45.59%) | 37/68 (54.41%) |
| Single-hop | 536/802 (66.83%) | 649/802 (80.92%) |

## Scoring limits and next diagnostic boundary

The unchanged official judge also enforces complete fact sets and strict time granularity. One apparent grading anomaly, `conv-26#q0000`, marks retrieval's “Yesterday, 7 May 2023” wrong against “7 May 2023,” while full recall's exact answer passes. Other inspected questions contain source/subject ambiguity (for example `conv-26#q0078` asks about Melanie's bowl but the cited speaker is Caroline). These remain official failures; this audit does not override or replace judgments. No percentage of the remaining errors is attributed to judge/dataset noise without a systematic independent audit.

The evidence supports prioritizing source-preserving contextual retrieval: broader candidate recall, neighboring turns, speaker/event identity, collection of multiple supporting spans, and compact focused evidence. That is a future hypothesis, not a validated improvement. Finish the frozen raw-context control to assess formatting/date-note attribution. Any method change needs a new development set and a newly frozen comparison; these inspected benchmark items must not become a tuning set.

Machine-readable counts, exact score-file hashes, selected evidence/rank traces and category results are in `homebase/bench/aml/results/locomo_competitive_20261002/rca_20261003.json`. Counts were checked against all 1,382 unique paired IDs and the saved official binary `llm_score` fields. The remote rank audit opened the source SQLite stores in read-only mode; no GPU calls were made.

## Completion update

All three methods have now completed answers and official scoring for every question. The plain chronological full-context control scored **879/1,382 (63.60%)**. Full recall improves on it by **59 answers / 4.27 percentage points**: 133 questions favor full recall, 74 favor plain context, 805 are correct under both and 370 are wrong under both. This supports a benefit from the full-recall representation package, which combines formatting, timestamps and date notes; it does not isolate which component caused the gain.

Descriptive conversation-cluster intervals reuse the frozen competitor bootstrap parameters (100,000 resamples of ten whole conversations with replacement, question-weighted score differences, seed 20261002, percentiles 0.625 and 99.375). Python's stdlib random generator and linear percentile interpolation produce these 98.75% intervals: full recall minus retrieval **[5.92, 16.57] points**; full recall minus plain context **[0.64, 7.05] points**. These control comparisons do not satisfy the separate matched-competitor gate. All category differences against retrieval are positive, so the earlier +5-point overall/no >5-point category regression rule passes for this run.

The evaluator is stopped and its completion marker exists. The existing watcher restored Flash Next at **2026-10-03 10:56:35 PDT**; its listener on port 11234 was verified. Final counts, per-category/per-conversation scores, paired transitions, intervals and exact scored-file hashes are saved in `homebase/bench/aml/results/locomo_competitive_20261002/completed_results_20261003.json`; `latest-status.json` is now the completed snapshot. Matched competitors, independent reproduction, public release and any official leaderboard submission remain outstanding.
