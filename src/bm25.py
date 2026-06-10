"""Offline pre-computation: BM25 scores of every candidate against a
JD-derived query. Demoted from recall gate to fusion signal (the
plain-language Tier 5 trap means no candidate may be dropped for lacking
JD vocabulary — BM25 only adds evidence, never filters).

    python src/bm25.py --candidates <candidates.jsonl>

Output: artifacts/bm25_scores.npy aligned to artifacts/candidate_ids.json
order (both iterate candidates.jsonl in file order).
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

import numpy as np

from common import ARTIFACTS, candidate_text, iter_candidates

K1, B = 1.5, 0.75
TOKEN = re.compile(r"[a-z0-9+#@.]+")

# Query terms distilled from the JD's must-haves and ideal-candidate
# sketch. Multi-word concepts are collapsed to their distinctive tokens.
QUERY = (
    "embedding embeddings retrieval ranking ranker search semantic vector "
    "recommendation recommender personalization relevance bm25 faiss "
    "elasticsearch opensearch pinecone weaviate qdrant milvus hybrid "
    "ndcg mrr evaluation a/b ab-test offline llm fine-tuning lora rag "
    "re-ranking transformer python production deployed shipped scale "
    "latency index drift nlp"
).split()


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()

    docs = [Counter(tokenize(candidate_text(c)))
            for c in iter_candidates(args.candidates)]
    n = len(docs)
    avgdl = sum(sum(d.values()) for d in docs) / n

    df = Counter()
    for d in docs:
        for t in set(QUERY) & set(d):
            df[t] += 1
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(QUERY)}

    scores = np.zeros(n, dtype=np.float32)
    for i, d in enumerate(docs):
        dl = sum(d.values())
        s = 0.0
        for t in QUERY:
            tf = d.get(t, 0)
            if tf:
                s += idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
        scores[i] = s

    ARTIFACTS.mkdir(exist_ok=True)
    np.save(ARTIFACTS / "bm25_scores.npy", scores)
    print(f"BM25 scored {n} candidates; "
          f"mean={scores.mean():.2f} max={scores.max():.2f}")


if __name__ == "__main__":
    main()
