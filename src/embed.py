"""Offline pre-computation: embed all 100K candidate narratives and the
JD facet texts with a small CPU bi-encoder.

This step is allowed to exceed the 5-minute window (submission spec
section 10.3); rank.py only loads the resulting artifacts. Run:

    python src/embed.py --candidates <candidates.jsonl>

Outputs (artifacts/):
    embeddings.npy      float32 [N, 384], L2-normalized
    candidate_ids.json  row order
    facets.npy          float32 [F, 384], L2-normalized
    facets.json         facet names
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import ARTIFACTS, candidate_text, iter_candidates

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ideal-candidate facets authored from the JD's "how to read between the
# lines" section. Similarity is taken per-facet and fused in rank.py.
FACETS = {
    "retrieval_ranking": (
        "Engineer who designed and shipped embedding-based retrieval and "
        "ranking systems to production: vector search, hybrid retrieval "
        "combining BM25 and dense embeddings, semantic search, candidate "
        "or content matching, re-ranking."),
    "recsys_product": (
        "Built and launched a recommendation or personalization system at "
        "a product company serving real users at meaningful scale, owning "
        "it end to end from data pipeline to serving."),
    "ranking_eval": (
        "Designed evaluation frameworks for search and ranking quality: "
        "NDCG, MRR, offline benchmarks, A/B testing, relevance judgments, "
        "engagement metrics, offline-to-online correlation."),
    "llm_pragmatic": (
        "Applied LLMs pragmatically in production: fine-tuning with LoRA, "
        "retrieval-augmented generation, LLM-based re-ranking, balancing "
        "latency and quality."),
    "ml_infra": (
        "Hands-on machine learning engineer with strong Python who "
        "deployed models to production and handled embedding drift, index "
        "refresh, monitoring, and retrieval-quality regressions."),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")

    ids, texts = [], []
    for c in iter_candidates(args.candidates):
        ids.append(c["candidate_id"])
        # MiniLM truncates at 256 tokens; keep the most recent material,
        # which candidate_text() puts first.
        texts.append(candidate_text(c)[:1600])

    t0 = time.time()
    emb = model.encode(texts, batch_size=args.batch_size,
                       normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)
    print(f"encoded {len(ids)} candidates in {time.time() - t0:.0f}s")

    facet_names = list(FACETS)
    fac = model.encode([FACETS[k] for k in facet_names],
                       normalize_embeddings=True).astype(np.float32)

    ARTIFACTS.mkdir(exist_ok=True)
    np.save(ARTIFACTS / "embeddings.npy", emb)
    np.save(ARTIFACTS / "facets.npy", fac)
    (ARTIFACTS / "candidate_ids.json").write_text(json.dumps(ids))
    (ARTIFACTS / "facets.json").write_text(json.dumps(facet_names))
    print(f"artifacts written to {ARTIFACTS}")


if __name__ == "__main__":
    main()
