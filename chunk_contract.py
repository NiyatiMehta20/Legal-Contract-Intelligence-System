"""
Improved contract chunking that respects semantic boundaries
Preserves section headers and complete clauses
"""

import re

def chunk_contract(text, chunk_size=600, overlap=100):
    """
    Chunk contract text while respecting natural boundaries
    
    Args:
        text: Full contract text
        chunk_size: Target size for each chunk (characters)
        overlap: Overlap between chunks (characters)
    
    Returns:
        List of chunk strings
    """
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Split by numbered sections (1., 2., etc.) or major headers
    section_pattern = r'(?=\n\d+\.\s+[A-Z])'
    sections = re.split(section_pattern, text)
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        # If this section is small, add to current chunk
        if len(current_chunk) + len(section) < chunk_size:
            current_chunk += "\n\n" + section if current_chunk else section
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If section itself is too large, split by sentences
            if len(section) > chunk_size:
                sub_chunks = split_by_sentences(section, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = section
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def split_by_sentences(text, max_size=600, overlap=100):
    """
    Split long text by sentences while respecting max_size
    """
    
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds max_size
        if len(current) + len(sentence) > max_size and current:
            chunks.append(current.strip())
            # Start new chunk with overlap (last few sentences)
            words = current.split()
            overlap_text = ' '.join(words[-20:]) if len(words) > 20 else current
            current = overlap_text + " " + sentence
        else:
            current += " " + sentence if current else sentence
    
    if current:
        chunks.append(current.strip())
    
    return chunks


def chunk_contract_advanced(text, chunk_size=600, overlap=100):
    """
    Advanced chunking that preserves contract structure
    Handles: preamble, numbered clauses, definitions, etc.
    """
    
    # Clean text
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    chunks = []
    
    # Extract and preserve the preamble (everything before "1.")
    preamble_match = re.search(r'^(.*?)(?=\n\s*1\.)', text, re.DOTALL)
    if preamble_match:
        preamble = preamble_match.group(1).strip()
        # Ensure preamble is chunked properly if too long
        if len(preamble) > chunk_size:
            preamble_chunks = split_by_sentences(preamble, chunk_size, overlap)
            chunks.extend(preamble_chunks)
        else:
            chunks.append(preamble)
        
        # Remove preamble from text
        text = text[len(preamble):]
    
    # Now chunk the rest (numbered sections)
    remaining_chunks = chunk_contract(text, chunk_size, overlap)
    chunks.extend(remaining_chunks)
    
    return chunks


if __name__ == "__main__":
    # Test with sample text
    sample = """State of __________
LANDLORD-TENANT NON-DISCLOSURE AND CONFIDENTIALITY AGREEMENT
This Agreement is entered into by and between:
Disclosing Party: ____________________________, as a(n) Individual Corporation
Receiving Party: ____________________________, as a(n) Individual Corporation

1. Confidential Information. This is the first clause with lots of details about what constitutes confidential information including financial data, trade secrets, and proprietary information.

2. Exclusions from Confidential Information. The obligation of confidentiality will not apply to information that is publicly known or independently developed.
"""
    
    chunks = chunk_contract_advanced(sample)
    
    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"CHUNK {i} ({len(chunk)} chars):")
        print(chunk[:200])
        print("\n" + "="*80 + "\n")