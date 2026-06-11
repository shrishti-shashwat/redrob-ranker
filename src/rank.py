"""The constrained ranking step: produces the top-100 submission CSV.

    python src/rank.py --candidates <candidates.jsonl> --out submission.csv

Budget: <= 5 min wall-clock, CPU-only, no network, 16 GB RAM. The heavy
lifting (embeddings, BM25) is precomputed by embed.py / bm25.py; this
step streams the candidate file once for structured features, loads the
artifacts, fuses, applies the availability multiplier, and writes the CSV
with per-candidate reasoning.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time

import numpy as np

from common import ARTIFACTS, iter_candidates
from features import availability, fit_features, honeypot_reasons
from reasoning import build_reasoning


def load_candidate_rows(path: str, ids_order: list[str]):
    """Yield (candidate_id, features, avail, av_facts, is_honeypot).

    Prefers the features.pkl artifact (precompute_features.py) — parsing
    the ~487 MB JSONL in here measured 127-311s of the 5-min budget.
    Falls back to live streaming so rank.py stays runnable standalone.
    """
    cache = ARTIFACTS / "features.pkl"
    if cache.exists():
        rows = pickle.loads(cache.read_bytes())
        if [r[0] for r in rows] == ids_order:
            return rows
        print("features.pkl stale (id mismatch); re-extracting live")
    return [(c["candidate_id"], fit_features(c), *availability(c),
             bool(honeypot_reasons(c)))
            for c in iter_candidates(path)]

# Fusion weights (tuned via audit.py inspection, no labels exist).
W_STRUCT, W_SEMANTIC, W_BM25 = 0.45, 0.40, 0.15

# Graded JD disqualifiers: "we will not move forward" -> multiplicative
# kill; "probably not" -> heavy penalty inside the structured score.
KILL_HONEYPOT = 0.02
KILL_CONSULTING_ONLY = 0.05   # explicit JD kill, carve-out already handled
KILL_RESEARCH_ONLY = 0.05     # explicit JD kill
DAMP_NON_TECH = 0.10          # "non-engineering titles regardless of skills"


def structured_score(f: dict) -> float:
    """Combine fit_features() components into [0, ~1]. Weights mirror the
    JD's own emphasis: retrieval/ranking evidence and shipped-to-
    production dominate; nice-to-haves add small bonuses."""
    s = 0.0
    s += 0.16 * min(f["ev_retrieval"], 5) / 5
    s += 0.14 * (1.0 if (f["ml_title"] or f["search_title"]) else 0.0)
    s += 0.08 * min(f["hist_ml"] + f["hist_search"], 3) / 3
    s += 0.12 * f["ev_production"]
    s += 0.10 * f["ev_eval"]
    s += 0.05 * f["ev_llm"]
    s += 0.08 * min(f["sk_retrieval"], 6) / 6      # corroboration-gated
    s += 0.04 * min(f["sk_ml"], 5) / 5
    s += 0.02 * min(f["sk_llm"], 4) / 4
    s += 0.03 * f["has_python"]
    s += 0.08 * f["product_ratio"]
    s += 0.04 * f["ai_native"]
    s += 0.02 * f["faang"]
    s += 0.04 * f["yoe_score"]
    # graded "probably not" penalties
    if f["job_hopper"]:
        s -= 0.08
    if f["cv_only"]:
        s -= 0.10
    if f["langchain_only"]:
        s -= 0.08
    return max(s, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()

    emb = np.load(ARTIFACTS / "embeddings.npy")
    facets = np.load(ARTIFACTS / "facets.npy")
    bm25 = np.load(ARTIFACTS / "bm25_scores.npy")
    ids_order = json.loads((ARTIFACTS / "candidate_ids.json").read_text())
    assert len(ids_order) == emb.shape[0] == bm25.shape[0]

    # Semantic fit: mean of top-2 facet similarities — rewards depth on
    # the JD's core facets without requiring all five.
    sims = emb @ facets.T                      # [N, F]
    sims.sort(axis=1)
    semantic = sims[:, -2:].mean(axis=1)

    def znorm(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-9)

    sem_z, bm25_z = znorm(semantic), znorm(bm25)

    sem01 = 1 / (1 + np.exp(-sem_z))
    bm01 = 1 / (1 + np.exp(-bm25_z))

    rows = []
    for i, (cid, f, avail, av_facts, hp) in enumerate(
            load_candidate_rows(args.candidates, ids_order)):
        assert cid == ids_order[i], "artifact/data order mismatch"
        struct = structured_score(f)
        fit = W_STRUCT * struct + W_SEMANTIC * sem01[i] + W_BM25 * bm01[i]

        if hp:
            fit *= KILL_HONEYPOT
        if f["consulting_only"]:
            fit *= KILL_CONSULTING_ONLY
        if f["research_only"]:
            fit *= KILL_RESEARCH_ONLY
        if f["non_tech"]:
            fit *= DAMP_NON_TECH

        score = fit * avail
        rows.append((score, cid, f, av_facts, hp))

    # deterministic tie-break: score desc, then candidate_id asc
    rows.sort(key=lambda r: (-r[0], r[1]))
    top = rows[: args.top]

    with open(args.out, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, f, av_facts, hp) in enumerate(top, 1):
            w.writerow([cid, rank, f"{score:.6f}",
                        build_reasoning(rank, f, av_facts)])

    print(f"wrote top {len(top)} to {args.out} in {time.time() - t0:.0f}s")
    print(f"score range: {top[0][0]:.4f} .. {top[-1][0]:.4f}")


if __name__ == "__main__":
    main()
