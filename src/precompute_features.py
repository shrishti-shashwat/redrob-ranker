"""Offline pass: extract structured features, availability, and honeypot
flags for all candidates and cache them as an artifact.

    python src/precompute_features.py --candidates <candidates.jsonl>

Rationale: parsing the ~487 MB JSONL inside the timed ranking step costs
2-5 minutes depending on disk cache and CPU contention — measured runs
ranged 127s to 311s, which gambles the 5-minute Stage 3 budget on
machine weather. Spec section 10.3 allows precomputed artifacts as long
as the ranking step that produces the CSV fits the window, so the
per-candidate extraction moves here and rank.py loads the cache
(falling back to live streaming if the artifact is absent).
"""

from __future__ import annotations

import argparse
import pickle
import time

from common import ARTIFACTS, iter_candidates
from features import availability, fit_features, honeypot_reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()
    t0 = time.time()

    rows = []
    for c in iter_candidates(args.candidates):
        f = fit_features(c)
        avail, av_facts = availability(c)
        rows.append((c["candidate_id"], f, avail, av_facts,
                     bool(honeypot_reasons(c))))

    out = ARTIFACTS / "features.pkl"
    with open(out, "wb") as fh:
        pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cached features for {len(rows)} candidates to {out} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
