"""Pre-submission self-evaluation. There is no leaderboard and only 3
submissions, so this is the only feedback loop we get:

1. Honeypot check — zero tolerance: no flagged ID may appear in the
   top 100 (disqualification threshold is >10, our bar is 0).
2. Behavioral-twin check — among profile-similar pairs in/near the top
   100 (embedding cosine > 0.93), verify the better-engaged candidate
   ranks higher. Catches an availability multiplier that's too weak.
3. Top-K dump for manual review of fit and reasoning quality.

    python src/audit.py --candidates <candidates.jsonl> --submission submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from common import ARTIFACTS, iter_candidates
from features import availability, honeypot_reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--dump-top", type=int, default=50)
    args = ap.parse_args()

    with open(args.submission, encoding="utf8") as f:
        sub = list(csv.DictReader(f))
    sub_ids = [r["candidate_id"] for r in sub]
    rank_of = {r["candidate_id"]: int(r["rank"]) for r in sub}

    by_id = {}
    honeypots = set()
    for c in iter_candidates(args.candidates):
        cid = c["candidate_id"]
        if honeypot_reasons(c):
            honeypots.add(cid)
        if cid in rank_of:
            by_id[cid] = c

    # 1 — honeypots
    hp_in_top = [cid for cid in sub_ids if cid in honeypots]
    print(f"[honeypots] flagged in pool: {len(honeypots)}; "
          f"in submission: {len(hp_in_top)} {'FAIL ' + str(hp_in_top) if hp_in_top else 'OK'}")

    # 2 — behavioral twins
    ids_order = json.loads((ARTIFACTS / "candidate_ids.json").read_text())
    row = {cid: i for i, cid in enumerate(ids_order)}
    emb = np.load(ARTIFACTS / "embeddings.npy")
    sub_emb = emb[[row[cid] for cid in sub_ids]]
    sims = sub_emb @ sub_emb.T
    np.fill_diagonal(sims, 0)
    bad = checked = 0
    for i in range(len(sub_ids)):
        for j in range(i + 1, len(sub_ids)):
            if sims[i, j] > 0.93:
                a, b = sub_ids[i], sub_ids[j]
                av_a, _ = availability(by_id[a])
                av_b, _ = availability(by_id[b])
                if abs(av_a - av_b) > 0.12:
                    checked += 1
                    hi, lo = (a, b) if av_a > av_b else (b, a)
                    if rank_of[hi] > rank_of[lo]:
                        bad += 1
                        print(f"  twin violation: {lo} (rank {rank_of[lo]}) "
                              f"outranks better-engaged {hi} (rank {rank_of[hi]})")
    print(f"[twins] similar pairs with availability gap: {checked}; "
          f"violations: {bad} {'FAIL' if bad else 'OK'}")

    # 3 — manual dump
    print(f"\n[top {args.dump_top}] title | yoe | company | reasoning")
    for r in sub[: args.dump_top]:
        c = by_id[r["candidate_id"]]
        p = c["profile"]
        print(f"{int(r['rank']):3d}. {p['current_title']:38.38s} "
              f"{p['years_of_experience']:4.1f}y  {p['current_company']:14.14s} "
              f"| {r['reasoning'][:90]}")


if __name__ == "__main__":
    main()
