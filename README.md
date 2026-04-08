# Legal Contract Intelligence System

AI-powered contract analysis using hybrid RAG retrieval, cross-encoder reranking, and LLM answer generation — achieving **75% citation rate** and **sub-18s response latency** on a CPU-only stack.

---

## What it does

Upload any legal contract (NDA, lease, employment agreement) and ask plain-English questions. The system retrieves the most relevant clauses using a multi-stage retrieval pipeline and returns grounded answers with direct clause citations — no hallucination, no guesswork.

```
"How can this agreement be terminated?"
"What obligations survive termination?"
"What are the remedies for breach?"
"What is considered confidential information?"
```

---

## Evaluation results

Benchmarked across 8 diverse legal query types on a real NDA contract:

| Metric | Score |
|---|---|
| Citation rate | 75% |
| Substantial answer rate | 100% |
| Avg keyword coverage | 57.3% |
| Avg response time | 17.4s |
| Avg sources per answer | 2.4 clauses |
| Avg answer length | 106 words |

Full results in `evaluation_results_20260124_154547.json`.

---

## Pipeline architecture

The system is a four-stage pipeline — each stage is independently testable and swappable.

```
┌─────────────────────────────────────────────────────────┐
│  1. Document ingestion          build_vector_store.py   │
│     PDF / DOCX / TXT → section-aware chunking           │
│     → e5-base-v2 embeddings → FAISS + BM25 index        │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  2. Hybrid retrieval            search_clauses.py        │
│     top-30 FAISS (semantic) + top-30 BM25 (keyword)      │
│     scores normalised → fused 0.6/0.4 → top-15          │
│     reranked with ms-marco-MiniLM-L-6-v2 cross-encoder  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  3. Smart clause selection   smart_clause_selector.py   │
│     12 legal query types detected via regex             │
│     clause count set by complexity (2–5)                │
│     near-duplicate filtering (char similarity > 0.7)    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  4. Answer generation          answer_generator.py      │
│     FLAN-T5-Large · CPU-only · no GPU required          │
│     query-type-aware formatting · citations injected    │
└────────────────────────┬────────────────────────────────┘
                         │
          Grounded answer with clause citations
          avg 106 words · 75% citation rate
```

---

## Design decisions

**Why hybrid retrieval?**
Dense vectors catch semantic meaning ("what survives termination?" matches "obligations post-expiry") but miss exact legal terms. BM25 catches verbatim phrases ("injunctive relief", "indemnification") that embeddings dilute. The 60/40 split was chosen empirically — legal language sits at the intersection of both failure modes.

**Why a cross-encoder reranker?**
Bi-encoders (like e5-base-v2) compress query and document independently, which loses interaction signal. The cross-encoder sees both together, catching cases where a clause is semantically adjacent but not quite right. It runs only on 15 candidates, not the full corpus, keeping latency acceptable.

**Why variable clause counts?**
A "who are the parties?" question needs exactly one well-placed clause. A liability question might require four to cover damages, remedies, limitations, and jurisdiction. Fixed top-k retrieval produces bloated answers for simple queries and incomplete ones for complex queries.

**Why FLAN-T5-Large over Mistral?**
For a CPU-only deployment targeting sub-20s latency, T5's encoder-decoder architecture is significantly faster on CPU than autoregressive decoder-only models. The answer quality gap is compensated by structured post-processing in `_format_natural_answer()`.

---

## Project structure

```
├── build_vector_store.py     # Ingestion: chunking, embedding, FAISS + BM25 index
├── chunk_contract.py         # Section-aware chunking logic
├── load_contract.py          # PDF / DOCX / TXT loader
├── search_clauses.py         # Hybrid retrieval + reranking pipeline
├── smart_clause_selector.py  # Query intent detection + clause selection
├── answer_generator.py       # FLAN-T5-Large answer generation
├── frontend_app.py           # Gradio UI
├── evaluation.py             # Evaluation harness
└── evaluation_results_*.json
```

---

## Quickstart

```bash
pip install faiss-cpu sentence-transformers rank-bm25 transformers gradio pdfplumber python-docx

# Index a contract
mkdir uploads && cp your_contract.pdf uploads/
python build_vector_store.py

# Launch the UI
python frontend_app.py
# → http://127.0.0.1:7860
```

First query takes ~30s (model load). Subsequent queries: 5–18s on CPU.

---

## Known limitations

- **Type detection accuracy is 50%.** The regex classifier misroutes multi-signal queries — e.g. "Can either party terminate without cause?" fires on `party` before `termination`. Replacing regex with a fine-tuned classifier or few-shot prompt is the next improvement.
- **Answer quality is bounded by FLAN-T5.** The model's seq2seq architecture produces terse outputs; the `_format_natural_answer()` wrapper compensates but can misfire on novel question structures.
- **Single-document only.** The current pipeline indexes the most recently uploaded file and wipes previous indexes. Multi-document comparison is not yet supported.
- **Response time is CPU-bound at ~17s.** Moving to a smaller generative model or GPU inference would bring this under 5s.
