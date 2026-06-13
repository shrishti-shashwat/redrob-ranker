"""Sandbox ranking pipeline, independent of the Gradio UI.

Runs the SAME scoring/fusion as src/rank.py but computes embeddings and
BM25 live (a <=100-candidate sample embeds in seconds, so no precomputed
artifacts are needed). Imported by app.py for the hosted demo and by the
local test harness.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bm25 import B, K1, QUERY, tokenize          # noqa: E402
from common import candidate_text                # noqa: E402
from embed import FACETS, MODEL                  # noqa: E402
from features import availability, fit_features, honeypot_reasons  # noqa: E402
from rank import (DAMP_NON_TECH, KILL_CONSULTING_ONLY,             # noqa: E402
                  KILL_HONEYPOT, KILL_RESEARCH_ONLY, W_BM25,
                  W_SEMANTIC, W_STRUCT, structured_score)
from reasoning import build_reasoning            # noqa: E402

SAMPLE = ROOT / "sample_candidates_100.jsonl"
_model = None


def get_model():
    """Load MiniLM once, lazily (keeps Space cold-start cheap)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL, device="cpu")
    return _model


def bm25_live(cands: list[dict]) -> np.ndarray:
    """BM25 vs the JD query, computed over just this sample."""
    docs = [Counter(tokenize(candidate_text(c))) for c in cands]
    n = len(docs)
    avgdl = sum(sum(d.values()) for d in docs) / n if n else 1.0
    df = Counter()
    for d in docs:
        for t in set(QUERY) & set(d):
            df[t] += 1
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(QUERY)}
    scores = np.zeros(n, dtype=np.float32)
    for i, d in enumerate(docs):
        dl = sum(d.values()) or 1
        s = 0.0
        for t in QUERY:
            tf = d.get(t, 0)
            if tf:
                s += idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
        scores[i] = s
    return scores


def rank_candidates(cands: list[dict], top_k: int = 100) -> pd.DataFrame:
    """Full pipeline on a small sample, mirroring rank.py's fusion."""
    model = get_model()
    texts = [candidate_text(c)[:1600] for c in cands]
    emb = model.encode(texts, normalize_embeddings=True,
                       batch_size=64).astype(np.float32)
    fac = model.encode([FACETS[k] for k in FACETS],
                       normalize_embeddings=True).astype(np.float32)

    sims = emb @ fac.T
    sims.sort(axis=1)
    semantic = sims[:, -2:].mean(axis=1)
    bm25 = bm25_live(cands)

    def znorm(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    sem01 = 1 / (1 + np.exp(-znorm(semantic)))
    bm01 = 1 / (1 + np.exp(-znorm(bm25)))

    rows = []
    for i, c in enumerate(cands):
        f = fit_features(c)
        fit = (W_STRUCT * structured_score(f)
               + W_SEMANTIC * sem01[i] + W_BM25 * bm01[i])
        if honeypot_reasons(c):
            fit *= KILL_HONEYPOT
        if f["consulting_only"]:
            fit *= KILL_CONSULTING_ONLY
        if f["research_only"]:
            fit *= KILL_RESEARCH_ONLY
        if f["non_tech"]:
            fit *= DAMP_NON_TECH
        avail, av_facts = availability(c)
        rows.append((fit * avail, c["candidate_id"], f, av_facts))

    rows.sort(key=lambda r: (-r[0], r[1]))
    out = []
    for rank, (score, cid, f, av) in enumerate(rows[:top_k], 1):
        out.append({"rank": rank, "candidate_id": cid,
                    "score": round(float(score), 6),
                    "reasoning": build_reasoning(rank, f, av)})
    return pd.DataFrame(out)


def load_jsonl(src) -> list[dict]:
    if hasattr(src, "read"):
        lines = src.read().decode("utf8").splitlines()
    elif Path(str(src)).exists():
        lines = Path(src).read_text(encoding="utf8").splitlines()
    else:
        lines = str(src).splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()][:100]


def rank_from_source(src=None, top_k: int = 100):
    """Returns (status_markdown, dataframe, csv_path)."""
    src = src if src is not None else str(SAMPLE)
    cands = load_jsonl(src)
    flagged = {c["candidate_id"] for c in cands if honeypot_reasons(c)}
    t0 = time.time()
    df = rank_candidates(cands, top_k)
    elapsed = time.time() - t0
    csv_path = ROOT / "ranked_output.csv"
    df.to_csv(csv_path, index=False)
    shortlist = max(1, len(df) // 2)          # top half = the "shortlist" zone
    hp_top = sum(cid in flagged
                 for cid in df["candidate_id"].head(shortlist))
    status = (
        f"**Ranked {len(cands)} candidates in {elapsed:.1f}s** (CPU, "
        f"budget 300s).  \n"
        f"Honeypots planted in this sample: **{len(flagged)}** — reaching "
        f"the top-{shortlist} shortlist: **{hp_top}** "
        f"(a sound ranker pushes impossible profiles to the bottom).")
    return status, df, str(csv_path)
