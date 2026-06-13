---
title: Starva Redrob Candidate Ranker
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Starva — Redrob Candidate Ranker (Sandbox)

Hosted demo for the India.Runs Intelligent Candidate Discovery & Ranking
Challenge. Ranks candidates for the Senior AI Engineer JD using the same
pipeline as the full submission: deterministic honeypot checks →
corroboration-gated structured features → MiniLM semantic + BM25 fusion →
behavioral availability multiplier → fact-grounded reasoning (no LLM at
runtime).

Click **Rank candidates** to run on the bundled 100-candidate sample
(55 ML fits, 12 honeypots, 33 generic), or upload your own `.jsonl`
(≤100 rows). Output is a ranked CSV, produced well within the CPU budget.

> This is the small-sample sandbox required by submission spec §10.5.
> The full 100K ranking lives in the GitHub repo and runs in ~2 s from
> precomputed artifacts.
