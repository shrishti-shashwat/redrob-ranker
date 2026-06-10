"""Shared constants and helpers for the Redrob candidate ranker.

The dataset's clock is frozen: 99,965 of 100,000 current jobs imply
"now" = 2026-06 from start_date + duration_months. All recency math must
anchor to REFERENCE_DATE, never the wall clock, or the ranking drifts
day by day and Stage 3 reproduction diverges from the submitted CSV.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REFERENCE_DATE = date(2026, 6, 1)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

# Company taxonomy. The dataset contains exactly 63 distinct companies; the
# generator assigns them to size bands (large ~23.5K stints, mid ~2.9K,
# small ~340, AI-native ~60-80, FAANG ~10). Real-world founding years are
# NOT respected by the generator (verified: zero stints predate each
# company's in-dataset earliest start), so no founding-year checks here.
CONSULTING = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Tech Mahindra", "Mphasis", "Mindtree", "Genpact AI",
}
AI_NATIVE = {
    "Sarvam AI", "Krutrim", "Observe.AI", "Yellow.ai", "Haptik", "Wysa",
    "Verloop.io", "Saarthi.ai", "Rephrase.ai", "Mad Street Den", "Niramai",
    "Aganitha", "Locobuzz", "Glance",
}
FAANG = {
    "Google", "Amazon", "Meta", "Microsoft", "Apple", "Netflix",
    "Salesforce", "Adobe", "Uber", "LinkedIn",
}
INDIA_PRODUCT = {
    "Swiggy", "Razorpay", "CRED", "Zomato", "Flipkart", "Meesho", "Nykaa",
    "InMobi", "BYJU'S", "PolicyBazaar", "Ola", "Zoho", "Vedantu", "Paytm",
    "Unacademy", "PharmEasy", "upGrad", "Freshworks", "PhonePe", "Dream11",
}
# Fictional corps (Pied Piper, Hooli, ...) are deliberately ambiguous —
# treated as neutral "generic large company".
PRODUCT = AI_NATIVE | FAANG | INDIA_PRODUCT

# The 12 generic non-tech titles covering ~64% of the pool. JD-sanctioned
# deprioritization: "non-engineering titles regardless of skill keywords".
NON_TECH_TITLES = {
    "business analyst", "hr manager", "mechanical engineer", "accountant",
    "project manager", "customer support", "operations manager",
    "content writer", "sales executive", "civil engineer",
    "graphic designer", "marketing manager",
}

# JD location preferences: Pune/Noida preferred; Hyderabad, Pune, Mumbai,
# Delhi NCR welcome; no visa sponsorship outside India.
PREFERRED_CITIES = ("pune", "noida")
WELCOME_CITIES = ("hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
                  "ghaziabad", "faridabad", "bangalore", "bengaluru")


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return None


def months_between(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def add_months(d: date, m: int) -> date:
    y = d.year + (d.month - 1 + m) // 12
    mo = (d.month - 1 + m) % 12 + 1
    return date(y, mo, 1)


def iter_candidates(path: str | Path):
    """Stream candidates from a .jsonl file without loading all 465 MB."""
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def candidate_text(c: dict) -> str:
    """The narrative text where the real signal lives: headline, summary,
    and career-history descriptions. Deliberately excludes the skills list
    (uncorroborated skills are the keyword-stuffer trap surface)."""
    p = c["profile"]
    parts = [p.get("headline", ""), p.get("summary", "")]
    for j in c["career_history"]:
        parts.append(f"{j['title']} at {j['company']}: {j.get('description', '')}")
    return "\n".join(parts)
