"""
Intelligent clause selection based on query type and context
Replaces the "top 1 clause only" approach with smarter selection
"""

import re
from typing import List, Dict, Tuple

class SmartClauseSelector:
    
    # Question type patterns
    QUESTION_PATTERNS = {
        'termination': r'\b(terminat|end|cancel|dissolve|exit)\b',
        'duration': r'\b(how long|duration|term|period|expire)\b',
        'obligations': r'\b(obligation|duty|must|shall|require|responsible)\b',
        'rights': r'\b(right|entitle|permit|allow|authorize)\b',
        'liability': r'\b(liab|remedy|damage|breach|penalt|indemnif)\b',
        'confidentiality': r'\b(confidential|secret|proprietary|disclos)\b',
        'payment': r'\b(pay|fee|cost|price|compensat|royalt)\b',
        'intellectual_property': r'\b(ip|intellectual property|patent|copyright|trademark)\b',
        'assignment': r'\b(assign|transfer|sublicense)\b',
        'governing_law': r'\b(govern|jurisdiction|dispute|arbitrat|law)\b',
        'definition': r'\b(what is|define|definition|mean|refer)\b',
        'party': r'\b(who|party|parties|between)\b',
    }
    
    # Complexity indicators
    COMPLEX_INDICATORS = [
        r'\band\b.*\b(also|additionally|furthermore)',
        r'\bor\b.*\bor\b',
        r'\bhow.*\band.*\bwhen\b',
        r'\bwhat.*\bif\b',
        r'\bcompare\b',
        r'\bdifference\b',
    ]
    
    @staticmethod
    def analyze_query_type(query: str) -> Dict:
        """Analyze question to determine type and complexity"""
        query_lower = query.lower()
        
        # Detect question type
        detected_types = []
        for q_type, pattern in SmartClauseSelector.QUESTION_PATTERNS.items():
            if re.search(pattern, query_lower):
                detected_types.append(q_type)
        
        primary_type = detected_types[0] if detected_types else 'general'
        
        # Detect complexity
        is_complex = any(
            re.search(pattern, query_lower) 
            for pattern in SmartClauseSelector.COMPLEX_INDICATORS
        )
        
        # Determine how many clauses needed
        if is_complex or len(detected_types) > 2:
            suggested_count = 5
        elif primary_type in ['definition', 'party']:
            suggested_count = 2
        elif primary_type in ['termination', 'obligations', 'liability']:
            suggested_count = 4
        else:
            suggested_count = 3
        
        needs_context = primary_type in ['obligations', 'liability', 'governing_law']
        
        return {
            'type': primary_type,
            'detected_types': detected_types,
            'is_complex': is_complex,
            'needs_context': needs_context,
            'suggested_clause_count': suggested_count
        }
    
    @staticmethod
    def select_optimal_clauses(
        ranked_results: List[Tuple[int, float]], 
        chunks: List[str],
        sources: List[str],
        query_analysis: Dict,
        min_score_threshold: float = 0.3,
        diversity_window: int = 100
    ) -> List[Dict]:
        """Select optimal set of clauses based on query type and relevance"""
        
        suggested_count = query_analysis['suggested_clause_count']
        
        selected = []
        seen_content = []
        
        for idx, score in ranked_results:
            if len(selected) >= suggested_count:
                break
            
            if score < min_score_threshold:
                continue
            
            chunk_text = chunks[idx]
            
            # Check for near-duplicate content
            is_duplicate = False
            for seen_text in seen_content:
                if SmartClauseSelector._is_similar_content(
                    chunk_text, seen_text, window=diversity_window
                ):
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            selected.append({
                'text': chunk_text,
                'source': sources[idx] if idx < len(sources) else 'Unknown',
                'score': float(score),
                'index': int(idx)
            })
            
            seen_content.append(chunk_text)
        
        # If we didn't get enough results, take top 2 regardless
        if len(selected) < 2 and len(ranked_results) > 0:
            for idx, score in ranked_results[:2]:
                if not any(s['index'] == idx for s in selected):
                    selected.append({
                        'text': chunks[idx],
                        'source': sources[idx] if idx < len(sources) else 'Unknown',
                        'score': float(score),
                        'index': int(idx)
                    })
        
        return selected
    
    @staticmethod
    def _is_similar_content(text1: str, text2: str, window: int = 100) -> bool:
        """Check if two chunks have similar content"""
        sample1 = text1[:window].lower().strip()
        sample2 = text2[:window].lower().strip()
        
        if len(sample1) < 20 or len(sample2) < 20:
            return False
        
        common = sum(1 for a, b in zip(sample1, sample2) if a == b)
        similarity = common / max(len(sample1), len(sample2))
        
        return similarity > 0.7


def analyze_query_type(query: str) -> Dict:
    """Convenience function"""
    return SmartClauseSelector.analyze_query_type(query)


def select_optimal_clauses(
    ranked_results: List[Tuple[int, float]],
    chunks: List[str],
    sources: List[str],
    query: str,
    **kwargs
) -> List[Dict]:
    """Convenience function that auto-analyzes query"""
    query_analysis = analyze_query_type(query)
    
    return SmartClauseSelector.select_optimal_clauses(
        ranked_results, 
        chunks, 
        sources, 
        query_analysis,
        **kwargs
    )