"""
CPU-optimized answer generator using FLAN-T5-Large
Much faster on CPU than Mistral - perfect for development and testing
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

class LegalAnswerGenerator:
    def __init__(self, model_name="google/flan-t5-large", use_4bit=False):
        """
        Initialize the answer generator with FLAN-T5 model
        
        Args:
            model_name: HuggingFace model identifier
            use_4bit: Not used for T5, kept for compatibility
        """
        print(f"Loading {model_name}...")
        print("Loading lightweight model (optimized for CPU)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def _build_legal_prompt(self, question, clauses_with_metadata):
        """Build a structured prompt for legal question answering"""
        
        # Format clauses - keep concise for T5
        context_parts = []
        for i, clause_info in enumerate(clauses_with_metadata[:3], 1):
            clause_text = clause_info['text'].strip()
            if len(clause_text) > 250:
                clause_text = clause_text[:250] + "..."
            context_parts.append(f"[{i}] {clause_text}")
        
        context = " ".join(context_parts)
        
        # Ultra-simple prompt for T5
        prompt = f"""Question: {question}
Context: {context}
Answer:"""
        
        return prompt
    
    def _format_natural_answer(self, question, raw_answer, clauses_with_metadata):
        """Convert T5's output into a natural, citation-rich answer"""
        
        # If the model generated a good answer already, use it
        if len(raw_answer) > 50 and "clause" in raw_answer.lower():
            return raw_answer
        
        # Otherwise, build a structured answer from clauses
        question_lower = question.lower()
        
        # Detect question type
        if any(word in question_lower for word in ["terminate", "termination", "end"]):
            answer = "Based on the contract:\n\n"
            for i, clause in enumerate(clauses_with_metadata, 1):
                answer += f"• According to Clause {i}, {clause['text'][:200]}"
                if i < len(clauses_with_metadata):
                    answer += "\n\n"
            return answer
        
        elif any(word in question_lower for word in ["who", "parties", "party"]):
            answer = f"According to the contract, {clauses_with_metadata[0]['text'][:300]}"
            return answer
        
        elif any(word in question_lower for word in ["what is", "define", "definition"]):
            answer = f"According to Clause 1, {clauses_with_metadata[0]['text'][:400]}"
            if len(clauses_with_metadata) > 1:
                answer += f"\n\nClause 2 further clarifies: {clauses_with_metadata[1]['text'][:200]}"
            return answer
        
        elif any(word in question_lower for word in ["how", "when", "where"]):
            answer = "Based on the relevant contract clauses:\n\n"
            for i, clause in enumerate(clauses_with_metadata[:2], 1):
                answer += f"Clause {i} states: {clause['text'][:250]}\n\n"
            return answer
        
        else:
            # Generic format
            answer = f"According to the contract:\n\n"
            for i, clause in enumerate(clauses_with_metadata, 1):
                answer += f"[Clause {i}] {clause['text'][:200]}"
                if i < len(clauses_with_metadata):
                    answer += "\n\n"
            return answer
    
    def generate_answer(self, question, clauses_with_metadata, max_new_tokens=200, temperature=0.7):
        """
        Generate answer to legal question
        
        Args:
            question: User's question
            clauses_with_metadata: List of dicts with clause info
            max_new_tokens: Maximum length of generated answer
            temperature: Sampling temperature (lower = more focused)
        
        Returns:
            Generated answer text
        """
        
        prompt = self._build_legal_prompt(question, clauses_with_metadata)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=400
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        print("Generating answer...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                early_stopping=True
            )
        
        # Decode
        raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format into natural answer with citations
        formatted_answer = self._format_natural_answer(
            question, 
            raw_answer.strip(), 
            clauses_with_metadata
        )
        
        return formatted_answer
    
    def generate_with_fallback(self, question, clauses_with_metadata):
        """Generate answer with automatic fallback for "not found" scenarios"""
        
        if not clauses_with_metadata or len(clauses_with_metadata) == 0:
            return self._generate_not_found_response(question)
        
        # Check if top clause has very low relevance score
        if clauses_with_metadata[0].get('score', 1.0) < 0.3:
            answer = self.generate_answer(question, clauses_with_metadata)
            answer += "\n\n**Note:** The retrieved clauses may have limited relevance to your question. Please verify the answer against the full contract."
            return answer
        
        return self.generate_answer(question, clauses_with_metadata)
    
    def _generate_not_found_response(self, question):
        """Generate response when no relevant clauses found"""
        return f"""I couldn't find specific clauses in the contract that directly answer: "{question}"

This could mean:
1. The information is not present in this contract
2. The topic is addressed using different terminology
3. The contract may need to be reviewed manually for this specific point

Please try rephrasing your question or consult the full contract document."""


# Global generator instance
_global_generator = None

def generate_answer(question, clauses, generator=None):
    """
    Backward compatible interface
    
    Args:
        question: User question
        clauses: List of clause texts (strings) or list of dicts
        generator: Optional pre-initialized generator instance
    """
    
    global _global_generator
    
    if generator is None:
        # Use global generator (lazy initialization)
        if _global_generator is None:
            print("Initializing answer generator (this takes ~10 seconds first time)...")
            _global_generator = LegalAnswerGenerator(use_4bit=False)
        generator = _global_generator
    
    # Convert simple list of strings to metadata format
    if clauses and isinstance(clauses[0], str):
        clauses_with_metadata = [
            {'text': clause, 'source': 'Contract', 'score': 1.0}
            for clause in clauses
        ]
    else:
        clauses_with_metadata = clauses
    
    return generator.generate_with_fallback(question, clauses_with_metadata)


if __name__ == "__main__":
    # Test the generator
    print("Testing Legal Answer Generator...")
    
    test_clauses = [
        {
            'text': 'Either party may terminate this Agreement upon thirty (30) days written notice to the other party.',
            'source': 'Section 8.1',
            'score': 0.95
        },
        {
            'text': 'Upon termination, all Confidential Information must be returned or destroyed within fifteen (15) days.',
            'source': 'Section 8.2',
            'score': 0.87
        }
    ]
    
    test_question = "How can I terminate this agreement?"
    
    generator = LegalAnswerGenerator(use_4bit=False)
    answer = generator.generate_answer(test_question, test_clauses)
    
    print("\n" + "="*80)
    print("TEST RESULT:")
    print("="*80)
    print(f"Question: {test_question}\n")
    print(f"Answer:\n{answer}")