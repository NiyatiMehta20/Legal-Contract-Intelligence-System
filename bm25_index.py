import numpy as np
import pickle
import re
from rank_bm25 import BM25Okapi

chunks = np.load("chunks.npy", allow_pickle=True)

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized = [tokenize(c["text"]) for c in chunks]

bm25 = BM25Okapi(tokenized)

with open("bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("BM25 index built.")
