# RAG System — Test Findings Report

**System under test:** NimbusCart customer-support RAG assistant
**Test method:** DeepEval RAG triad (Contextual Relevancy, Faithfulness,
Answer Relevancy) over a golden dataset of 5 questions.
**Judge model:** local llama3.1 (Ollama). Thresholds: 0.4 (lenient — local
judge + keyword retrieval are not production-grade; goal is to surface issues).
**Result:** 2 passed, 3 failed.

---

## Summary

| # | Question | Result | Failing metric(s) |
|---|---|---|---|
| golden0 | What payment methods do you accept? | PASS | — |
| golden1 | How do I reset my password? | PASS | — |
| golden2 | How long do I have to return electronics? | FAIL | Contextual Relevancy, Faithfulness |
| golden3 | Is standard shipping free? | FAIL | Contextual Relevancy, Faithfulness, Answer Relevancy |
| golden4 | Do you offer same-day grocery delivery? | FAIL | Contextual Relevancy, Answer Relevancy |

Overall pass rate: 40% (2/5). Three distinct, actionable issues found.

---

## Finding 1 — Generation contradicts the source (Faithfulness)

**Where:** golden3 ("Is standard shipping free?")
**Severity:** High (gives customers wrong information)
**Metric:** Faithfulness scored 0.00.

**Judge reason:** The answer stated standard shipping is free on orders over
$50, but the context also states it costs $4.99 otherwise — the answer
presented a partial/condition as if absolute, contradicting the full policy.

**Why it matters:** An unfaithful answer misinforms the user. Faithfulness is
the metric that catches "the answer isn't fully supported by the source."

**Root cause:** The generator oversimplified the retrieved context (the mock-
style summarization did not preserve the full conditional). Answer Relevancy
also failed because the answer gave a condition instead of directly answering
"is it free?".

**Recommended fix:** Strengthen the generation prompt to require complete,
direct answers ("state all conditions; answer the question directly"). Re-run
and confirm Faithfulness and Answer Relevancy recover.

---

## Finding 2 — Specific facts buried inside general chunks (Chunking / Contextual Relevancy)

**Where:** golden2 ("How long do I have to return electronics?")
**Severity:** Medium (relevant info retrieved, but diluted)
**Metric:** Contextual Relevancy scored 0.33; Faithfulness 0.00.

**Judge reason:** The retrieved context was mostly about the GENERAL return
policy and packaging, with the specific electronics 15-day window mentioned only
in passing — so most of the retrieved context was not relevant to the specific
question.

**Why it matters:** When a specific fact is buried inside a chunk about a more
general topic, retrieval returns a lot of irrelevant text alongside it, lowering
contextual relevancy and confusing generation.

**Root cause:** Chunking strategy. The electronics 15-day rule lives inside the
general "Return window" chunk rather than as its own focused chunk.

**Recommended fix:** Improve chunking so distinct facts (e.g. electronics window)
are separated, or add finer-grained chunks. Alternatively, retrieve more
targeted spans. Re-run and confirm Contextual Relevancy improves.

---

## Finding 3 — Weak out-of-scope handling (Hallucination risk)

**Where:** golden4 ("Do you offer same-day grocery delivery?") — not covered by
the knowledge base.
**Severity:** High (system should refuse, not improvise)
**Metric:** Contextual Relevancy 0.00; Answer Relevancy 0.00.

**Judge reason:** The retrieved context (exchanges, refunds, subscriptions) was
entirely unrelated to grocery delivery, and the answer did not address the
question.

**Why it matters:** For questions outside its knowledge, a RAG system should
clearly say "I don't have that information" rather than retrieve loosely-related
chunks and attempt an answer (a hallucination risk).

**Root cause:** (a) Keyword retrieval still returns low-relevance chunks because
some words coincidentally overlap; (b) no "no good match → say I don't know"
guard before generation.

**Recommended fix:** Add a retrieval-confidence threshold — if no chunk scores
above a minimum, short-circuit to "I'm sorry, I don't have information on that."
Re-run golden4 and confirm the system refuses gracefully.

---

## What PASSED (and why that's meaningful)

- golden0 (payment methods) and golden1 (password reset) passed all three triad
  metrics. These are "clean" cases: the fact lives in a single focused chunk,
  keyword retrieval matches well, and the answer is direct. This confirms the
  pipeline works correctly end-to-end when conditions are favourable.

---

## QA Conclusions

1. The RAG triad successfully surfaced THREE distinct failure modes —
   generation accuracy, chunking strategy, and out-of-scope handling — each with
   a clear, metric-backed explanation. A suite that only passed would have
   proven nothing; these failures are the value.
2. No single metric tells the whole story: Faithfulness caught the contradiction,
   Contextual Relevancy caught the retrieval/chunking issues, Answer Relevancy
   caught the off-target answers. This is exactly why the triad is used together.
3. Next step is the improvement loop: fix one issue, re-run the SAME golden
   dataset (regression testing), and confirm scores improve without breaking the
   passing cases.

## Prioritised fix backlog

1. [High] Add out-of-scope guard (Finding 3) — prevents misinformation.
2. [High] Prompt tuning for complete/direct answers (Finding 1).
3. [Medium] Re-chunk to separate specific facts (Finding 2).
