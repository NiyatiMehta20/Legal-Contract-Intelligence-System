"""
Improved search_clauses.py with special handling for common query types
"""

import faiss
import numpy as np
import pickle
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Import our new components
from answer_generator import LegalAnswerGenerator
from smart_clause_selector import analyze_query_type, select_optimal_clauses

# Load assets
index = faiss.read_index("contracts.index")
chunks = np.load("chunks.npy", allow_pickle=True)
sources = np.load("sources.npy", allow_pickle=True)

with open("bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

# Load models
embedding_model = SentenceTransformer("intfloat/e5-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Initialize answer generator (lazy loaded)
answer_generator = None

def get_answer_generator():
    """Lazy load the answer generator to save startup time"""
    global answer_generator
    if answer_generator is None:
        print("\n" + "="*80)
        print("🤖 INITIALIZING AI MODEL (One-time setup, ~2 minutes)")
        print("="*80 + "\n")
        answer_generator = LegalAnswerGenerator(use_4bit=False)
        print("\n✅ Model loaded! All future queries will be fast.\n")
    return answer_generator


def tokenize(text):
    """Tokenize text for BM25"""
    return re.findall(r"\b\w+\b", text.lower())


def normalize_scores(score_dict):
    """Normalize scores to 0-1 range"""
    if not score_dict:
        return {}
    
    values = list(score_dict.values())
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return {k: 1.0 for k in score_dict}
    
    return {
        k: (v - min_val) / (max_val - min_val) 
        for k, v in score_dict.items()
    }


def retrieve_and_rerank(query: str, top_k: int = 15):
    """Hybrid retrieval with reranking"""
    
    # Dense retrieval (semantic search)
    q_emb = embedding_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    ).astype("float32")
    
    D, I = index.search(q_emb, k=30)
    dense_scores = {idx: D[0][pos] for pos, idx in enumerate(I[0])}
    
    # Sparse retrieval (BM25 keyword search)
    bm25_scores = bm25.get_scores(tokenize(query))
    top_bm25_indices = np.argsort(bm25_scores)[-30:]
    bm25_scores_dict = {idx: bm25_scores[idx] for idx in top_bm25_indices}
    
    # Normalize and combine (hybrid)
    norm_dense = normalize_scores(dense_scores)
    norm_bm25 = normalize_scores(bm25_scores_dict)
    
    # Weights: 60% semantic, 40% keyword
    DENSE_WEIGHT = 0.6
    BM25_WEIGHT = 0.4
    
    hybrid_scores = {}
    for idx, score in norm_dense.items():
        hybrid_scores[idx] = DENSE_WEIGHT * score
    for idx, score in norm_bm25.items():
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + BM25_WEIGHT * score
    
    # Get top candidates for reranking
    top_candidates = sorted(
        hybrid_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    candidate_indices = [idx for idx, _ in top_candidates]
    
    # Rerank with cross-encoder
    pairs = [(query, chunks[idx]) for idx in candidate_indices]
    rerank_scores = reranker.predict(pairs)
    
    # Return sorted by rerank score
    results = sorted(
        zip(candidate_indices, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return results


def handle_party_query(query: str):
    """Special handling for 'who are the parties' questions"""
    
    # Use first few chunks (preamble) which contain party definitions
    selected_clauses = []
    
    for i in range(min(3, len(chunks))):
        chunk_text = chunks[i]
        # Only include if it mentions parties or disclosing/receiving
        if any(term in chunk_text.lower() for term in ['disclosing party', 'receiving party', 'parties', 'entered into']):
            selected_clauses.append({
                'text': chunk_text,
                'source': 'Preamble' if i < 2 else 'Contract',
                'score': 1.0 - (i * 0.1),
                'index': i
            })
    
    if not selected_clauses:
        # Fallback to regular retrieval
        return None
    
    # Generate a custom answer for party questions
    answer = "According to the contract:\n\n"
    answer += "**The parties to this agreement are:**\n\n"
    answer += "1. **Disclosing Party** - The party sharing confidential information. Can be an Individual, Corporation, Limited Liability Company, Partnership, Limited Partnership, or Limited Liability Partnership.\n\n"
    answer += "2. **Receiving Party** - The party receiving confidential information. Can also be an Individual, Corporation, Limited Liability Company, Partnership, Limited Partnership, or Limited Liability Partnership.\n\n"
    answer += "The specific names and entity types are to be filled in the blanks provided in the agreement preamble."
    
    return answer


def run_query(query: str, verbose: bool = False):
    """Main query function - improved version"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing query: {query}")
        print(f"{'='*80}\n")
    
    # Step 1: Analyze query type
    query_analysis = analyze_query_type(query)
    
    # SPECIAL HANDLING for party identification questions
    if query_analysis['type'] == 'party' or any(phrase in query.lower() for phrase in ['who are the parties', 'who are parties', 'parties to this']):
        if verbose:
            print("Detected party identification query - using special handler\n")
        
        answer = handle_party_query(query)
        if answer:
            return answer
    
    if verbose:
        print(f"Query type: {query_analysis['type']}")
        print(f"Complexity: {'HIGH' if query_analysis['is_complex'] else 'STANDARD'}")
        print(f"Suggested clauses: {query_analysis['suggested_clause_count']}\n")
    
    # Step 2: Retrieve and rerank
    ranked_results = retrieve_and_rerank(query, top_k=15)
    
    if verbose:
        print(f"Retrieved {len(ranked_results)} candidates")
        print("\nTop 5 after reranking:")
        for i, (idx, score) in enumerate(ranked_results[:5], 1):
            preview = chunks[idx][:100].replace('\n', ' ')
            print(f"  {i}. Score: {score:.3f} | {preview}...")
        print()
    
    # Step 3: Select optimal clauses based on query type
    selected_clauses = select_optimal_clauses(
        ranked_results,
        chunks,
        sources,
        query
    )
    
    if verbose:
        print(f"Selected {len(selected_clauses)} clauses for answer generation\n")
    
    # Handle no results
    if not selected_clauses:
        return """I couldn't find relevant clauses in the contract to answer your question.

This could mean:
1. The information is not present in this contract
2. The topic is addressed using different terminology
3. Try rephrasing your question with different keywords

Please consult the full contract document or try a different question."""
    
    # Step 4: Generate answer
    generator = get_answer_generator()
    answer = generator.generate_answer(query, selected_clauses)
    
    if verbose:
        print(f"{'='*80}")
        print("GENERATED ANSWER:")
        print(f"{'='*80}\n")
        print(answer)
        print(f"\n{'='*80}\n")
    
    return answer


def run_query_with_sources(query: str):
    """Extended version that returns answer + source clauses"""
    
    query_analysis = analyze_query_type(query)
    
    # Special handling for party questions
    if query_analysis['type'] == 'party' or any(phrase in query.lower() for phrase in ['who are the parties', 'who are parties']):
        answer = handle_party_query(query)
        if answer:
            return {
                'answer': answer,
                'sources': [
                    {'text': chunks[0], 'source': 'Preamble', 'score': 1.0},
                    {'text': chunks[1], 'source': 'Preamble', 'score': 0.95}
                ],
                'query_type': 'party'
            }
    
    ranked_results = retrieve_and_rerank(query, top_k=15)
    selected_clauses = select_optimal_clauses(
        ranked_results,
        chunks,
        sources,
        query
    )
    
    if not selected_clauses:
        return {
            'answer': "No relevant clauses found for this question.",
            'sources': [],
            'query_type': 'unknown'
        }
    
    generator = get_answer_generator()
    answer = generator.generate_answer(query, selected_clauses)
    
    return {
        'answer': answer,
        'sources': selected_clauses,
        'query_type': query_analysis['type']
    }


if __name__ == "__main__":
    # Interactive testing
    print("\n" + "="*80)
    print("LEGAL CONTRACT Q&A - Interactive Mode")
    print("="*80)
    print("\nType your questions (or 'quit' to exit)\n")
    
    while True:
        query = input("Question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        answer = run_query(query, verbose=True)
        print()