"""
Comprehensive evaluation script for Legal RAG system
Run this to get metrics for your CV!
"""

import json
import time
from datetime import datetime
from typing import List, Dict
import numpy as np
import re
from search_clauses import run_query_with_sources, analyze_query_type

# Test queries with expected characteristics
TEST_QUERIES = [
    {
        'question': 'How can this agreement be terminated?',
        'type': 'termination',
        'expected_keywords': ['terminate', 'notice', 'days', 'written']
    },
    {
        'question': 'What is the duration of this NDA?',
        'type': 'duration',
        'expected_keywords': ['year', 'term', 'period', 'effective']
    },
    {
        'question': 'What obligations survive termination?',
        'type': 'obligations',
        'expected_keywords': ['survive', 'obligation', 'termination', 'confidential']
    },
    {
        'question': 'What happens to confidential information after termination?',
        'type': 'confidentiality',
        'expected_keywords': ['return', 'destroy', 'confidential', 'information']
    },
    {
        'question': 'What are the remedies for breach of this agreement?',
        'type': 'liability',
        'expected_keywords': ['remedy', 'breach', 'damages', 'injunction']
    },
    {
        'question': 'Who are the parties to this agreement?',
        'type': 'party',
        'expected_keywords': ['party', 'parties', 'between']
    },
    {
        'question': 'What is considered confidential information?',
        'type': 'definition',
        'expected_keywords': ['confidential', 'information', 'include', 'mean']
    },
    {
        'question': 'Can either party terminate without cause?',
        'type': 'termination',
        'expected_keywords': ['terminate', 'cause', 'without']
    },
]


class RAGEvaluator:
    
    def __init__(self):
        self.results = []
        
    def evaluate_answer_quality(self, answer: str, expected_keywords: List[str]) -> Dict:
        """Simple keyword-based quality check"""
        
        answer_lower = answer.lower()
        
        # Check for keywords
        keywords_found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        keyword_coverage = keywords_found / len(expected_keywords) if expected_keywords else 0
        
        # Check for citations
        has_clause_citation = bool(re.search(r'\bclause\s+\d+\b', answer_lower))
        
        # Check answer length
        word_count = len(answer.split())
        is_substantial = word_count >= 30
        
        # Check for "not found" indicators
        not_found_indicators = [
            "couldn't find", "not present", "no information",
            "unable to locate", "doesn't contain"
        ]
        is_not_found = any(ind in answer_lower for ind in not_found_indicators)
        
        return {
            'keyword_coverage': keyword_coverage,
            'keywords_found': keywords_found,
            'total_keywords': len(expected_keywords),
            'has_citation': has_clause_citation,
            'word_count': word_count,
            'is_substantial': is_substantial,
            'is_not_found': is_not_found
        }
    
    def evaluate_single_query(self, test_case: Dict, verbose: bool = False) -> Dict:
        """Evaluate a single query"""
        
        question = test_case['question']
        expected_type = test_case['type']
        expected_keywords = test_case.get('expected_keywords', [])
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {question}")
            print(f"{'='*80}")
        
        # Measure response time
        start_time = time.time()
        
        try:
            result = run_query_with_sources(question)
            answer = result['answer']
            sources = result['sources']
            detected_type = result['query_type']
            
            elapsed = time.time() - start_time
            
            # Evaluate quality
            quality = self.evaluate_answer_quality(answer, expected_keywords)
            
            # Type detection accuracy
            type_match = detected_type == expected_type
            
            eval_result = {
                'question': question,
                'expected_type': expected_type,
                'detected_type': detected_type,
                'type_match': type_match,
                'answer': answer,
                'answer_length': len(answer),
                'num_sources': len(sources),
                'response_time': round(elapsed, 2),
                'quality_metrics': quality,
                'timestamp': datetime.now().isoformat()
            }
            
            if verbose:
                print(f"\nDetected Type: {detected_type} (Expected: {expected_type}) {'✓' if type_match else '✗'}")
                print(f"Response Time: {elapsed:.2f}s")
                print(f"Sources Used: {len(sources)}")
                print(f"Keyword Coverage: {quality['keyword_coverage']:.1%}")
                print(f"Has Citations: {'Yes' if quality['has_citation'] else 'No'}")
                print(f"\nAnswer ({quality['word_count']} words):")
                print("-" * 80)
                print(answer)
                print("-" * 80)
            
            return eval_result
            
        except Exception as e:
            print(f"ERROR: {e}")
            return {
                'question': question,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_full_evaluation(self, test_cases: List[Dict] = None, verbose: bool = True) -> Dict:
        """Run evaluation on all test cases"""
        
        if test_cases is None:
            test_cases = TEST_QUERIES
        
        print("\n" + "="*80)
        print("RUNNING FULL RAG SYSTEM EVALUATION")
        print("="*80)
        print(f"\nEvaluating {len(test_cases)} test queries...")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Testing: {test_case['question'][:60]}...")
            
            result = self.evaluate_single_query(test_case, verbose=False)
            results.append(result)
            
            # Quick status
            if 'error' not in result:
                qm = result['quality_metrics']
                status = '✓' if qm['is_substantial'] and qm['has_citation'] else '⚠'
                print(f"     {status} {result['response_time']}s | {qm['word_count']} words | "
                      f"{qm['keywords_found']}/{qm['total_keywords']} keywords | "
                      f"{'Cited' if qm['has_citation'] else 'No citation'}")
        
        # Compute aggregate metrics
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("\n❌ No valid results to analyze!")
            return {'results': results, 'summary': {}}
        
        summary = self._compute_summary_metrics(valid_results)
        
        # Print summary
        self._print_summary(summary, len(test_cases))
        
        # Save results
        output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary': summary,
                'results': results,
                'test_cases': test_cases
            }, f, indent=2)
        
        print(f"\n📊 Full results saved to: {output_file}")
        
        return {
            'summary': summary,
            'results': results
        }
    
    def _compute_summary_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics"""
        
        total = len(results)
        
        # Type detection accuracy
        type_matches = sum(1 for r in results if r.get('type_match', False))
        type_accuracy = type_matches / total
        
        # Quality metrics
        keyword_coverages = [r['quality_metrics']['keyword_coverage'] for r in results]
        avg_keyword_coverage = np.mean(keyword_coverages)
        
        with_citations = sum(1 for r in results if r['quality_metrics']['has_citation'])
        citation_rate = with_citations / total
        
        substantial_answers = sum(1 for r in results if r['quality_metrics']['is_substantial'])
        substantial_rate = substantial_answers / total
        
        not_found_answers = sum(1 for r in results if r['quality_metrics']['is_not_found'])
        not_found_rate = not_found_answers / total
        
        # Response times
        response_times = [r['response_time'] for r in results]
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        
        # Source usage
        source_counts = [r['num_sources'] for r in results]
        avg_sources = np.mean(source_counts)
        
        # Answer lengths
        answer_lengths = [r['quality_metrics']['word_count'] for r in results]
        avg_answer_length = np.mean(answer_lengths)
        
        return {
            'total_queries': total,
            'type_detection_accuracy': round(type_accuracy, 3),
            'avg_keyword_coverage': round(avg_keyword_coverage, 3),
            'citation_rate': round(citation_rate, 3),
            'substantial_answer_rate': round(substantial_rate, 3),
            'not_found_rate': round(not_found_rate, 3),
            'avg_response_time_sec': round(avg_response_time, 2),
            'max_response_time_sec': round(max_response_time, 2),
            'avg_sources_used': round(avg_sources, 1),
            'avg_answer_length_words': round(avg_answer_length, 1)
        }
    
    def _print_summary(self, summary: Dict, total_tests: int):
        """Print formatted summary"""
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY - USE THESE NUMBERS FOR YOUR CV!")
        print("="*80)
        
        print(f"\n📈 Overall Performance:")
        print(f"  Total Queries Tested: {summary['total_queries']}/{total_tests}")
        print(f"  Type Detection Accuracy: {summary['type_detection_accuracy']:.1%}")
        print(f"  Avg Keyword Coverage: {summary['avg_keyword_coverage']:.1%}")
        
        print(f"\n✅ Answer Quality:")
        print(f"  Citation Rate: {summary['citation_rate']:.1%}")
        print(f"  Substantial Answers: {summary['substantial_answer_rate']:.1%}")
        print(f"  Avg Answer Length: {summary['avg_answer_length_words']:.0f} words")
        
        print(f"\n⚡ Performance:")
        print(f"  Avg Response Time: {summary['avg_response_time_sec']:.2f}s")
        print(f"  Max Response Time: {summary['max_response_time_sec']:.2f}s")
        print(f"  Avg Sources Used: {summary['avg_sources_used']:.1f} clauses")
        
        print("\n" + "="*80)
        print("💡 FOR YOUR CV, YOU CAN NOW CLAIM:")
        print("="*80)
        print(f"  • Achieved {summary['citation_rate']:.0%} citation rate in contract analysis")
        print(f"  • {summary['avg_keyword_coverage']:.0%} keyword coverage across legal queries")
        print(f"  • Sub-{int(summary['avg_response_time_sec'])+1}-second query response time")
        print(f"  • {summary['avg_sources_used']:.1f} average clauses per analysis")
        print("="*80 + "\n")


def main():
    """Main evaluation interface"""
    
    evaluator = RAGEvaluator()
    
    print("\n" + "="*80)
    print("LEGAL RAG SYSTEM - EVALUATION TOOL")
    print("="*80)
    
    print("\nThis will run 8 test queries and give you metrics for your CV.")
    input("\nPress Enter to start evaluation...")
    
    evaluator.run_full_evaluation(verbose=True)


if __name__ == "__main__":
    main()