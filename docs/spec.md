# Probable Cause: Spec (v1)

> Ask questions about 20 years of aviation accident investigations. Every answer cites its sources, and the retrieval pipeline is chosen by benchmark, not vibes.

## Problem
Most RAG demos run on clean text and are never measured. This one runs on a genuinely messy public corpus, NTSB aviation accident reports (scanned PDFs, tables, structured CAROL data plus narrative), and publishes a benchmark showing which retrieval choices actually help.

## Product
- Chat UI: ask questions like "What are the most common causes of fatal icing accidents in Part 91 operations since 2010?"
- Answers cite specific reports (docket links) with highlighted passages; the system refuses when evidence is insufficient
- Hybrid queries: structured filters (date, aircraft type, phase of flight, fatalities) combined with semantic search over narratives
- An "Explain this answer" view: retrieved chunks, scores, reranker decisions

## Pipeline
1. **Ingest:** NTSB CAROL bulk data + report PDFs → text extraction (PDF parser, OCR fallback) → normalization
2. **Chunking strategies (compared):** fixed-size, section-aware (Probable Cause / Findings / Narrative), and parent-child
3. **Retrieval (compared):** BM25, dense embeddings (pgvector), hybrid (RRF), with and without a reranker, and with and without LLM query rewriting into structured filters
4. **Generation:** Claude with mandatory citations; a citation verifier checks that each claim is supported by a cited chunk

## Benchmark (the headline)
- ~300 question/answer pairs: drawn from report "probable cause" fields (ground truth exists), plus hand-written analytical questions
- Metrics: retrieval recall@k and MRR, answer correctness (LLM judge calibrated against a human-labeled subset), citation precision, latency, cost per query
- A results table and charts in the README: "Hybrid + section-aware chunking + rerank: recall@10 X% → Y%"
- Runs in CI on a small subset; full runs are reproducible via one command
- Traced to **Tracewright**

## Stack
Python (ingest + eval + FastAPI API), Postgres + pgvector (Neon), Next.js chat UI (Vercel), Anthropic API with OpenRouter as an option. The hosted demo has a daily query cap.

## Milestones
1. Data ingest + structured DB + text extraction
2. Baseline RAG + citation UI
3. Benchmark dataset + eval harness
4. Strategy comparison experiments + results
5. Deploy, README with charts, write-up ("What actually improved retrieval on messy documents")
