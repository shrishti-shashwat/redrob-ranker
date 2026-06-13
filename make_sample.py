"""Build a 100-candidate demo sample for the sandbox.

Curated (not random) so the hosted demo visibly shows the ranker working:
strong ML/search fits that should rise, planted honeypots that must stay
out of the top, and generic non-tech profiles that should sink. Output is
a plain JSONL the sandbox loads by default.

    python make_sample.py --candidates <full candidates.jsonl>
"""

import argparse
import json
import sys

sys.path.insert(0, "src")
from features import honeypot_reasons  # noqa: E402

ML_HINT = ("ml engineer", "machine learning", "ai engineer", "ai research",
           "search engineer", "recommendation", "nlp engineer", "data scientist",
           "applied")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="sample_candidates_100.jsonl")
    args = ap.parse_args()

    strong, honeypots, generic = [], [], []
    for line in open(args.candidates, encoding="utf8"):
        c = json.loads(line)
        title = c["profile"]["current_title"].lower()
        if honeypot_reasons(c):
            if len(honeypots) < 12:
                honeypots.append(line)
        elif any(h in title for h in ML_HINT):
            if len(strong) < 55:
                strong.append(line)
        elif len(generic) < 33:
            generic.append(line)
        if len(strong) >= 55 and len(honeypots) >= 12 and len(generic) >= 33:
            break

    rows = strong + honeypots + generic
    with open(args.out, "w", encoding="utf8") as f:
        f.writelines(rows)
    print(f"wrote {len(rows)} candidates to {args.out} "
          f"({len(strong)} ML fits, {len(honeypots)} honeypots, "
          f"{len(generic)} generic)")


if __name__ == "__main__":
    main()
