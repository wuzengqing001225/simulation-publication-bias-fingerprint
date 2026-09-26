"""Sentence-embedding vectors per construct for the item-similarity check.

Usage (repository root):  python code/embed_constructs.py
Needs sentence-transformers. Each construct vector is the normalized mean of its
item embeddings (protocols/construct_item_texts.json). Output:
analysis_tables/construct_embeddings.json
"""
import json, numpy as np
from sentence_transformers import SentenceTransformer
texts = json.load(open("protocols/construct_item_texts.json"))
out = {}
for name in ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]:
    m = SentenceTransformer(name); embs = {}
    for k in sorted(texts):
        v = m.encode(texts[k], normalize_embeddings=True).mean(0)
        embs[k] = (v / np.linalg.norm(v)).tolist()
    out[name] = embs
json.dump(out, open("analysis_tables/construct_embeddings.json", "w"))
print("saved analysis_tables/construct_embeddings.json")
