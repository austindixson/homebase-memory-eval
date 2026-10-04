# Evaluation scope amendment — 2026-10-03

The user instructed us to compare the completed candidate with published competitor results, without rerunning their systems. This supersedes the task scope for matched comparator execution; it does not retroactively change any frozen score or classify an unrun comparator as defeated.

- Original candidate: 938/1,382 (67.8726483%). Published MemOS reference: 63.60%. The registered historical-reference threshold of at least 880 correct is passed.
- Attribution controls remain published: retrieval 773/1,382, plain full context 879/1,382. All candidate, control and judge outcomes are retained.
- MemPalace's queued run was stopped after 39 answers; it is incomplete and has no final comparison score. Mem0 was disabled before any benchmark extraction/answer/judge calls. MemOS and EverOS preparation was deferred before benchmark inference. Cancellation receipts and partial evidence are retained locally.
- The user will perform the independent rerun. The original tolerance remains unchanged: both runs exceed the historical reference and differ by at most one percentage point. No rerun has completed yet.
- Public memory code, manifests and raw receipts have been released. Internal clean-export preparation matched all 1,382 original prompts for each of the three methods; this verifies export integrity, not independent reproduction.

`LOCOMO_REFINED_ACCEPTANCE.md` and `LOCOMO_MATCHED_COMPARATORS_20261003.md` remain historical registrations. The four matched-comparison confidence-interval condition is not passed or weakened; the associated matched-superiority claim is not being pursued. This amendment supports the narrower statement: **the completed disclosed evaluation exceeds the highest open-source reference in the pinned LoCoMo-Refined published table**. Official leaderboard acceptance and AML evaluation remain separate requirements.
