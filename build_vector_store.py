import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from load_contract import load_contract_text
from chunk_contract import chunk_contract_advanced

UPLOAD_DIR = "uploads"

# 🔥 CRITICAL: wipe old indexes
for f in ["contracts.index", "chunks.npy", "sources.npy", "bm25.pkl"]:
    if os.path.exists(f):
        os.remove(f)

# Find uploaded files (TXT, PDF, or DOCX)
all_files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
contract_files = [f for f in all_files if f.endswith(('.txt', '.pdf', '.docx'))]

if not contract_files:
    raise RuntimeError(f"No contract files found in {UPLOAD_DIR}")

print(f"Found {len(contract_files)} contract file(s)")

# Process the most recent file
contract_files.sort(key=lambda x: os.path.getmtime(os.path.join(UPLOAD_DIR, x)), reverse=True)
contract_path = os.path.join(UPLOAD_DIR, contract_files[0])

print(f"Processing: {contract_files[0]}")

# Load text
text = load_contract_text(contract_path)

if not text or len(text.strip()) < 100:
    raise RuntimeError(f"Contract appears empty or too short: {len(text)} characters")

# Chunk the contract
chunks = chunk_contract_advanced(text)
sources = [os.path.basename(contract_path)] * len(chunks)

print(f"Chunked document into {len(chunks)} chunks")

if len(chunks) == 0:
    raise RuntimeError("No chunks generated from contract")

# Embeddings
model = SentenceTransformer("intfloat/e5-base-v2")

print("Generating embeddings...")
embeddings = model.encode(
    ["passage: " + c for c in chunks],
    normalize_embeddings=True,
    show_progress_bar=True
)

# Convert to float32 and ensure 2D array
embeddings = np.array(embeddings).astype("float32")

if embeddings.ndim == 1:
    embeddings = embeddings.reshape(1, -1)

print(f"Embeddings shape: {embeddings.shape}")

# FAISS
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "contracts.index")

# BM25
tokenized_chunks = [c.lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Save
np.save("chunks.npy", np.array(chunks, dtype=object))
np.save("sources.npy", np.array(sources, dtype=object))
with open("bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("Vector store built successfully.")