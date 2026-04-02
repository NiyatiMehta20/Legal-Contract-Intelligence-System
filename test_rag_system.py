"""
Test script to evaluate RAG system performance across different question types
"""

import sys
from search_clauses import (
    embedding_model, reranker, index, chunks, sources, bm25,
    tokenize, normalize_scores, analyze_query_type, select_optimal_clauses
)
from answer_generator import generate_answer
import numpy as np

# Test queries covering different question types
TEST_QUERIES = [
    # Termination questions
    "When can this NDA be terminated?",
    "How do I terminate the agreement?",
    "What is the notice period for termination?",
    
    # Duration/Timeline questions
    "How long does this agreement last?",
    "What is the duration of the NDA?",
    
    # Obligations questions
    "What happens to confidential information after termination?",
    "When must confidential information be returned?",
    "Do obligations survive after termination?",
    
    # Rights questions
    "What intellectual property rights are granted?",
    "Who owns the confidential information?",
    
    # Liability questions
    "What are the remedies for breach?",
    "Is there any limitation of liability?",
    
    # General questions
    "What is considered confidential information?",
    "Who are the parties to this agreement?",
    "Can this agreement be assigned to another party?",
    
    # Yes/No questions
    "Can either party terminate without cause?",
    "Is mutual consent required for termination?",
]

def test_retrieval_quality(query, top_k=5):
    """
    Test retrieval quality for a single query
    """
    
    # Dense retrieval
    q_emb = embedding_model.encode(
        ["query: " + query], 
        normalize_embeddings=True
    ).astype("float32")
    
    D, I = index.search(q_emb, k=30)
    dense_scores = {idx: D[0][pos] for pos, idx in enumerate(I[0])}
    
    # Sparse retrieval
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-30:]
    bm25_scores_dict = {idx: bm25_scores[idx] for idx in top_bm25_indices}
    
    # Hybrid
    norm_dense = normalize_scores(dense_scores)
    norm_bm25 = normalize_scores(bm25_scores_dict)
    
    hybrid_candidates = {}
    for idx, score in norm_dense.items():
        hybrid_candidates[idx] = 0.6 * score
    for idx, score in norm_bm25.items():
        hybrid_candidates[idx] = hybrid_candidates.get(idx, 0) + 0.4 * score
    
    top_candidates = sorted(hybrid_candidates.items(), key=lambda x: x[1], reverse=True)[:15]
    candidate_indices = [idx for idx, _ in top_candidates]
    
    # Rerank
    pairs = [(query, chunks[idx]) for idx in candidate_indices]
    rerank_scores = reranker.predict(pairs)
    
    results = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
    
    return results[:top_k]


def evaluate_single_query(query, show_details=True):
    """
    Evaluate system performance on a single query
    """
    
    if show_details:
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
    
    # Analyze query
    query_analysis = analyze_query_type(query)
    
    # Retrieve
    results = test_retrieval_quality(query, top_k=10)
    
    if show_details:
        print(f"\nQuery Type: {query_analysis['type'].upper()}")
        print(f"Complexity: {'HIGH' if query_analysis['is_complex'] else 'STANDARD'}")
        print(f"\nTop 5 Retrieved Clauses:")
        
        for rank, (idx, score) in enumerate(results[:5], 1):
            print(f"\n{rank}. Score: {score:.3f} | {sources[idx]}")
            print(f"   {chunks[idx][:150].replace(chr(10), ' ')}...")
    
    # Select clauses
    selected = select_optimal_clauses(results, query_analysis)
    top_clauses = [chunks[idx] for idx, _ in selected]
    
    # Generate answer
    answer = generate_answer(query, top_clauses)
    
    if show_details:
        print(f"\n{'='*80}")
        print("GENERATED ANSWER:")
        print(f"{'='*80}\n")
        print(answer)
        print("\n" + "-"*80)
    
    return {
        'query': query,
        'query_type': query_analysis['type'],
        'num_results': len(results),
        'top_score': results[0][1] if results else 0,
        'num_selected': len(selected),
        'answer_length': len(answer),
        'has_citation': 'clause' in answer.lower() or '[clause' in answer.lower()
    }


def run_full_evaluation():
    """
    Run evaluation on all test queries
    """
    
    print("\n" + "="*80)
    print("RUNNING FULL RAG SYSTEM EVALUATION")
    print("="*80)
    print(f"\nTesting {len(TEST_QUERIES)} queries...\n")
    
    results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Testing: {query[:60]}...")
        result = evaluate_single_query(query, show_details=False)
        results.append(result)
        
        # Show quick summary
        status = "✓" if result['has_citation'] else "⚠"
        print(f"     {status} Score: {result['top_score']:.2f} | Selected: {result['num_selected']} clauses | Answer: {result['answer_length']} chars")
    
    # Summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    total = len(results)
    with_citations = sum(1 for r in results if r['has_citation'])
    avg_score = np.mean([r['top_score'] for r in results])
    avg_answer_length = np.mean([r['answer_length'] for r in results])
    
    print(f"\nTotal Queries: {total}")
    print(f"Answers with Citations: {with_citations}/{total} ({with_citations/total*100:.1f}%)")
    print(f"Average Top Score: {avg_score:.3f}")
    print(f"Average Answer Length: {avg_answer_length:.0f} characters")
    
    # By query type
    print("\nPerformance by Query Type:")
    query_types = set(r['query_type'] for r in results)
    for qtype in query_types:
        type_results = [r for r in results if r['query_type'] == qtype]
        type_citations = sum(1 for r in type_results if r['has_citation'])
        type_avg_score = np.mean([r['top_score'] for r in type_results])
        print(f"  {qtype.upper():10} - {len(type_results)} queries | Citations: {type_citations}/{len(type_results)} | Avg Score: {type_avg_score:.3f}")
    
    return results


def test_specific_query():
    """
    Interactive mode to test specific queries
    """
    
    print("\n" + "="*80)
    print("INTERACTIVE QUERY TESTING")
    print("="*80)
    print("\nEnter a query to test (or 'back' to return):\n")
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ['back', 'exit', 'quit']:
            break
        
        if not query:
            continue
        
        evaluate_single_query(query, show_details=True)
        print("\n")


def main():
    """
    Main testing interface
    """
    
    print("\n" + "="*80)
    print("CONTRACT ANALYSIS RAG SYSTEM - TESTING INTERFACE")
    print("="*80)
    print("\nOptions:")
    print("  1. Run full evaluation on test queries")
    print("  2. Test a specific query")
    print("  3. Show test query list")
    print("  4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            run_full_evaluation()
        elif choice == '2':
            test_specific_query()
        elif choice == '3':
            print("\nTest Queries:")
            for i, query in enumerate(TEST_QUERIES, 1):
                print(f"  {i:2}. {query}")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please select 1-4.")


if __name__ == "__main__":
    main()