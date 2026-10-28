# Benchmark datasheet

`benchmark/questions.jsonl`: 289 questions over the NTSB aviation corpus for events from
2018-01-01 on (the retrieval corpus; see `docs/results.md`). One JSON object per line:

| field | meaning |
|---|---|
| `id` | `L###` lookup, `S###` structured, `A###` analytical |
| `type` | `lookup`, `structured`, `analytical` |
| `question` | the natural-language question |
| `gold_events` | NTSB `ev_id`s that answer it (empty for unanswerable items) |
| `gold_answer` | reference answer used by the judge |
| `answerable` | `false` for the 5 questions the corpus cannot answer |
| `meta` | provenance: NTSB number and year (lookup), template and parameters (structured), the gold SQL predicate (analytical) |

## Composition

| type | n | gold events per item | how it was made |
|---|---:|---|---|
| lookup | 153 | exactly 1 | LLM-written from one event's narrative; gold answer = that event's probable cause |
| structured | 106 | median 2, max 9 | 5 SQL templates (58 count, 48 list questions); gold = the SQL result |
| analytical | 30 | median 41, max 168 (25 items); none (5 unanswerable) | hand-written; gold = an explicit SQL predicate over probable-cause text and coded fields |

## How it was built (reproducible)

```bash
uv run pc bench lookup        # 20 events/year 2018-2025, 20 per LLM call, leak-filtered
uv run pc bench structured    # seeded random template parameters, keep 1-10 gold events
uv run pc bench assemble      # + benchmark/analytical.yaml -> questions.jsonl
uv run pc bench rewrite       # query plans for the rewrite experiments -> plans.jsonl
```

All LLM calls (claude-sonnet-5-5 via `claude -p`) are cached in `benchmark/llm-cache/`, so
re-running the commands reproduces the files byte-for-byte without calling a model.

**Lookup.** Candidate events: single-aircraft US events, NTSB-led (not `WA` foreign
assists), with a probable cause of at least 40 characters and an analysis or factual
narrative; 20 per year in `md5(ev_id)` order. The model saw the event's metadata, the first
220 words of the analysis, and the probable cause marked "do not reveal". Of 160 generated
questions, 7 were rejected by the leak filter: none leaked an ID, registration or exact
date, but all 7 repeated half or more of the probable-cause content words.

**Structured.** Templates: fatal accidents by FAR part and year; fatal accidents by aircraft
category, state and year; night-IMC airplane accidents by phase and year; accidents by
defining event, state and year; serious/fatal accidents by FAR part, phase and year. A
parameter draw is kept only if the result has 1-10 events and every one has narrative text,
so a perfect retriever can reach all of them.

**Analytical.** Written by hand after querying the database; each gold set is the result of
the `gold_where` predicate stored with the question, and the gold answer summarizes what
those probable-cause statements say.

## Known biases and limits

- **Lookup questions are easier than real user questions.** The generator was told to use
  distinguishing circumstances, so most questions name the aircraft make/model, the US
  state and the year (median 28 words). That favours lexical retrieval and filter
  extraction; real users often know less.
- **Lookup gold is single-event.** A question may plausibly describe a near-identical
  accident; such near-misses count as failures.
- **Gold answers for lookup are NTSB probable-cause statements**, which are terse and
  formulaic; an answer that paraphrases the analysis instead may be judged differently.
- **Structured ground truth uses coded fields** (FAR part, phase of the defining event,
  light and weather condition). Narratives occasionally disagree with the coding, and the
  phase filter uses the *defining* occurrence only.
- **Analytical gold sets are predicate-defined**, so they are both incomplete (events that
  describe carburetor ice without the phrase "carburetor ic" are missed) and coarse
  (any mention counts). Retrieval scores for them are capped recall: share of the first
  page that is in the gold set.
- **Year coverage**: 2018-2025 for lookup, 2018-2024 for structured. 2025-2026 events are
  under-represented because many investigations are still open.
- **Unanswerable items are few (5)**, so the refusal rate on them is a smoke test, not a
  measurement.
