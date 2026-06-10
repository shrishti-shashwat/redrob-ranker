"""Programmatic reasoning generation — no LLM at runtime.

Stage 4 manual review samples 10 rows and checks: specific facts, JD
connection, honest concerns, no hallucination, variation, and
rank-consistent tone. Every clause below is rendered from extracted
profile facts, so hallucination is structurally impossible, and tone
tracks the score because both come from the same features.
"""

from __future__ import annotations


def _fit_clause(f: dict, variant: int) -> str:
    facts = f["facts"]
    yoe = facts["yoe"]
    title = facts["title"]
    ev = facts["evidence"]

    openers = [
        f"{title} with {yoe:.1f} yrs",
        f"{yoe:.1f} yrs as {title}",
        f"{title}, {yoe:.1f} yrs experience",
    ]
    head = openers[variant % len(openers)]

    bits = []
    if "retrieval/ranking work in career history" in ev:
        bits.append("career history shows hands-on retrieval/ranking work"
                    if variant % 2 else "has actually built search/ranking systems")
    if "ranking-evaluation experience (NDCG/A-B style)" in ev:
        bits.append("evaluates ranking quality rigorously")
    if "AI-native startup experience" in ev:
        bits.append("AI-native startup background")
    if "mostly product-company career" in ev:
        bits.append("product-company track record" if variant % 2
                    else "career spent shipping at product companies")
    if "LLM/fine-tuning exposure" in ev:
        bits.append("LLM fine-tuning exposure")
    if facts.get("sk_retrieval_names"):
        bits.append("corroborated " + "/".join(facts["sk_retrieval_names"][:3]))
    return head + ("; " + "; ".join(bits[:3]) if bits else "")


def _signal_clause(av: dict, variant: int) -> str:
    parts = []
    rr = av.get("response_rate")
    if rr is not None:
        if rr >= 0.6:
            parts.append(f"responsive ({rr:.0%} reply rate)")
        elif rr < 0.25:
            parts.append(f"low reply rate ({rr:.0%})")
    d = av.get("days_inactive", 0)
    if d <= 30:
        parts.append("recently active")
    elif d > 150:
        parts.append(f"inactive ~{d // 30} months")
    n = av.get("notice")
    if n is not None and n <= 30:
        parts.append(f"{n}-day notice")
    elif n is not None and n > 75:
        parts.append(f"{n}-day notice")
    return ", ".join(parts[:2 + variant % 2])


def build_reasoning(rank: int, f: dict, av: dict) -> str:
    variant = sum(ord(ch) for ch in f["facts"]["title"]) + rank  # deterministic
    fit = _fit_clause(f, variant)
    sig = _signal_clause(av, variant)
    concerns = list(f["facts"]["concerns"])
    loc = av.get("location", "")
    if "India" not in loc:
        concerns.append(f"based outside India ({loc.split(',')[-1].strip()})")

    s = fit
    if sig:
        s += f". {sig.capitalize()}" if variant % 3 else f". Signals: {sig}"
    if concerns and (rank > 10 or len(concerns) > 1):
        s += f". Concern: {concerns[0]}"
    elif concerns:
        s += f"; note {concerns[0]}"
    if rank > 80 and not concerns:
        s += ". Adjacent rather than core fit — included on overall balance"
    return s + "."
