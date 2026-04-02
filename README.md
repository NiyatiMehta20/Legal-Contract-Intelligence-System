# Legal-Contract-Intelligence-System
AI-powered contract analysis using hybrid RAG retrieval, cross-encoder reranking, and LLM answer generation, achieving 75% citation rate and sub-18s response latency


What It Does

Upload any legal contract (NDA, lease agreement, employment contract) and ask natural language questions about it. The system retrieves the most relevant clauses using a multi-stage AI pipeline and generates grounded, explainable answers with direct citations.

Ask questions like:

"How can this agreement be terminated?"

"What obligations survive termination?"

"What are the remedies for breach?"

"What is considered confidential information?"

Key Features

Hybrid Retrieval — Combines FAISS dense vector search (60%) with BM25 keyword search (40%) for high recall across both semantic and exact-match queries

Cross-Encoder Reranking — Uses ms-marco-MiniLM-L-6-v2 to rerank top-15 candidates for precision

Smart Clause Selection — Deduplication and query-type-aware selection returns 2–5 clauses based on question complexity

Query Intent Detection — Classifies 12 legal query types (termination, liability, confidentiality, etc.) to tune retrieval strategy

Explainable Outputs — Every answer includes clause-level citations so you can verify against the source

Gradio UI — Clean web interface with upload, Q&A, and source viewer panels

CPU-Optimised — Runs fully on CPU with no GPU required

Evaluation Results

Evaluated across 8 diverse legal query types on a real NDA contract:

MetricScoreCitation Rate75%

Substantial Answer Rate100%

Avg Keyword Coverage 57.3%

Avg Response Time 17.4s

Avg Sources Per Answer 2.4 clauses

Avg Answer Length 106 words


Full results in evaluation_results_20260124_154547.json

How the Pipeline Works

1. Document Ingestion (build_vector_store.py)

Loads PDF, DOCX, or TXT contracts via load_contract.py

Chunks using chunk_contract_advanced() — respects section boundaries, preserves clause structure, overlaps by 100 characters

Embeds chunks using intfloat/e5-base-v2 (E5 model with passage: prefix)

Stores in FAISS IndexFlatIP (inner product / cosine similarity)

Builds a parallel BM25 index using rank-bm25

2. Retrieval (search_clauses.py)

Embeds the query with query: prefix (E5 asymmetric encoding)

Retrieves top-30 from FAISS and top-30 from BM25 independently

Normalises both score distributions to [0, 1]

Fuses: hybrid = 0.6 × dense + 0.4 × BM25

Reranks top-15 candidates using a cross-encoder (ms-marco-MiniLM-L-6-v2)

3. Clause Selection (smart_clause_selector.py)

Detects query type from 12 intent classes using regex patterns

Sets suggested_clause_count (2–5) based on query complexity

Filters near-duplicate chunks using character-level similarity (threshold: 0.7)

Falls back to top-2 clauses if insufficient results pass threshold

4. Answer Generation (answer_generator.py)

Formats selected clauses into a structured prompt for FLAN-T5-Large

Applies question-type-aware response formatting (termination / party / definition etc.)

Injects clause citations into the final answer

Includes a low-relevance fallback with a user-facing confidence note

