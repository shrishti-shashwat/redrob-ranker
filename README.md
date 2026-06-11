# Redrob Candidate Ranker

Ranks the 100K-candidate pool against the Senior AI Engineer JD for the
Intelligent Candidate Discovery & Ranking Challenge. Hybrid pipeline:
deterministic honeypot/consistency checks → structured JD-fit features →
precomputed bi-encoder semantic similarity + BM25 fusion → multiplicative
behavioral-availability adjustment → programmatic per-candidate reasoning.

## Reproduce the submission

```bash
pip install -r requirements.txt

# 1. Offline pre-computation (may exceed 5 min; spec §10.3 allows this)
python src/embed.py --candidates ./candidates.jsonl                # ~80 min CPU
python src/bm25.py  --candidates ./candidates.jsonl                # ~3 min
python src/precompute_features.py --candidates ./candidates.jsonl  # ~5 min

# 2. Constrained ranking step (~2 s with artifacts, CPU-only, no network)
python src/rank.py --candidates ./candidates.jsonl --out submission.csv

# 3. Self-audit before submitting
python src/audit.py --candidates ./candidates.jsonl --submission submission.csv
```

Artifacts land in `artifacts/` (embeddings ~150 MB, features cache ~25 MB —
regenerate with the step-1 scripts if not shipped). `rank.py` falls back to
live feature extraction (~2–5 min) when `features.pkl` is absent; with all
artifacts present it completes in seconds, so the 5-minute window is never
at the mercy of disk cache or CPU contention.

## Design in one paragraph

Skills lists have uniform marginals across the pool (every top skill ≈12.1K
holders) but real conditional signal (P(PyTorch | ML-titled) = 21.6% vs
1.2%) — so skills only score when corroborated by title or career-history
text, which kills keyword stuffers without discarding evidence. The
narrative text (summary + career descriptions) carries the true signal and
feeds both the bi-encoder facets and BM25. Honeypots (~80 planted, 68
detectable record-internally) are caught by three checks: expert-with-0-
months, current-job implied-now ≠ the dataset's frozen clock of **2026-06**,
and career-span vs stated-experience drift >3 yrs. All recency features
anchor to `REFERENCE_DATE = 2026-06-01` — never the wall clock — so Stage 3
reproduction is bit-identical. JD disqualifiers are graded: "will not move
forward" (consulting-only career, research-only career) are multiplicative
kills; "probably not" (job-hopping, CV/speech-only, LangChain-only) are
penalties. Behavioral signals form a multiplicative availability factor
(floor 0.25) per the signals doc, which is what separates behavioral twins.

## Layout

| Path | Role |
|---|---|
| `src/common.py` | Frozen reference date, company taxonomy, shared parsing |
| `src/features.py` | Honeypot checks, structured fit features, availability |
| `src/embed.py` | Offline: MiniLM embeddings for all 100K + JD facets |
| `src/bm25.py` | Offline: BM25 scores vs JD-derived query |
| `src/precompute_features.py` | Offline: cache structured features/availability/honeypots |
| `src/rank.py` | The ≤5-min step: fuse, multiply, top-100 CSV |
| `src/reasoning.py` | Programmatic reasoning (no LLM, no hallucination) |
| `src/audit.py` | Honeypot zero-check, twin check, top-50 manual dump |
