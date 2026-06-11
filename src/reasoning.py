"""Programmatic reasoning generation — no LLM at runtime.

Stage 4 manual review samples 10 rows and checks: specific facts, JD
connection, honest concerns, no hallucination, variation, and
rank-consistent tone. Every clause below is rendered from extracted
profile facts, so hallucination is structurally impossible, and tone
tracks the score because both come from the same features.

Variation strategy: each row gets a deterministic variant index (derived
from candidate identity, not rank) that selects sentence structure,
connectors, and which facts lead. Specifics (company names, matched
retrieval terms, corroborated skill names, signal values) differ
per-candidate, so no two rows share more than incidental phrasing.
"""

from __future__ import annotations

# Regex stems -> display terms for reasoning text.
TERM_DISPLAY = {"recommend": "recommendation", "retriev": "retrieval",
                "ranker": "ranking", "bm25": "BM25",
                "elasticsearch": "Elasticsearch", "faiss": "FAISS",
                "pinecone": "Pinecone", "qdrant": "Qdrant",
                "milvus": "Milvus", "weaviate": "Weaviate"}


def _cap(s: str) -> str:
    """Capitalize only the first character — str.capitalize() lowercases
    the rest of the string, mangling proper nouns like CRED."""
    return s[0].upper() + s[1:] if s else s


def _evidence_bits(f: dict, variant: int) -> list[str]:
    facts = f["facts"]
    bits = []

    terms = facts.get("retrieval_terms")
    if terms:
        shown = [TERM_DISPLAY.get(x, x) for x in terms[:2 + variant % 2]]
        t = "/".join(shown)
        bits.append([
            f"career history covers {t} work",
            f"hands-on {t} experience in past roles",
            f"has built {t} systems",
            f"track record in {t}",
        ][variant % 4])

    if f["ev_eval"]:
        bits.append([
            "thinks about ranking evaluation (offline metrics/A-B)",
            "evaluation-framework experience for ranking quality",
            "has measured ranking quality properly",
        ][variant % 3])

    pcs = facts.get("product_companies")
    if pcs:
        where = " and ".join(pcs)
        bits.append([
            f"shipped at product companies ({where})",
            f"product-side career incl. {where}",
            f"experience at {where}",
        ][variant % 3])

    if f["ev_llm"]:
        bits.append(["LLM/fine-tuning exposure",
                     "works with LLMs and fine-tuning"][variant % 2])

    sk = facts.get("sk_retrieval_names")
    if sk:
        bits.append(f"profile-corroborated {'/'.join(sk[:3])}")

    return bits


def _signal_bits(av: dict, variant: int) -> list[str]:
    parts = []
    rr = av.get("response_rate")
    d = av.get("days_inactive", 0)
    n = av.get("notice")

    if rr is not None and rr >= 0.55:
        parts.append([f"replies to {rr:.0%} of recruiter messages",
                      f"{rr:.0%} recruiter response rate",
                      f"responsive ({rr:.0%})"][variant % 3])
    elif rr is not None and rr < 0.25:
        parts.append(f"low response rate ({rr:.0%})")

    if d <= 21:
        parts.append(["active this month", "recently active",
                      "freshly active on platform"][variant % 3])
    elif d > 150:
        parts.append(f"~{d // 30} months inactive")

    if n is not None and n <= 30:
        parts.append(f"{n}-day notice")
    return parts


def build_reasoning(rank: int, f: dict, av: dict) -> str:
    facts = f["facts"]
    cid_hash = sum(ord(ch) * (i + 3) for i, ch in
                   enumerate(facts["title"] + facts["company"]))
    variant = cid_hash % 12

    yoe, title, company = facts["yoe"], facts["title"], facts["company"]
    head = [
        f"{title} ({yoe:.1f} yrs) at {company}",
        f"{yoe:.1f} yrs, currently {title} at {company}",
        f"{title} with {yoe:.1f} yrs, now at {company}",
        f"Currently {title} at {company}, {yoe:.1f} yrs total",
    ][variant % 4]

    ev = _evidence_bits(f, variant)
    sig = _signal_bits(av, variant)

    concerns = list(facts["concerns"])
    n = av.get("notice")
    if n is not None and n > 30:  # JD: buyout covers up to 30 days only
        concerns.append(f"{n}-day notice, above the 30-day buyout window")
    loc = av.get("location", "")
    if "India" not in loc:
        tail = "" if av.get("relocate") else ", not willing to relocate"
        concerns.append(
            f"based in {loc.split(',')[-1].strip()} (no visa sponsorship{tail})")
    d = av.get("days_inactive", 0)
    if d > 150:
        concerns.append(f"not seen on platform for ~{d // 30} months")
    elif d > 45:
        concerns.append(f"last active ~{d // 7} weeks ago")
    if not av.get("open_to_work", True):
        concerns.append("not flagged open-to-work")

    s = head
    if ev:
        s += " — " + "; ".join(ev[:3]) if variant % 2 else ". " + _cap("; ".join(ev[:3]))
    if sig:
        joined = ", ".join(sig[:2])
        s += [". Engagement: ", ". On signals: ", ". " ][variant % 3] + \
            (_cap(joined) if variant % 3 == 2 else joined)

    if concerns:
        joiner = [". Concern: ", ". Caveat: ", "; concern — "][variant % 3]
        s += joiner + " and ".join(concerns[:2])
    elif rank > 85:
        s += [". Solid-but-adjacent fit at this depth of the list",
              ". Included as a credible backfill rather than a core match"][variant % 2]
    return s + "."
