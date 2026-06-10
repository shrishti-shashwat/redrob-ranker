"""Structured feature extraction: honeypot flags, JD-fit features,
availability multiplier, and the evidence facts that feed reasoning text.

Design notes (validated empirically on the full 100K pool):
- Skills lists have uniform marginals but real conditional signal
  (P(PyTorch | ML-titled) = 21.6% vs 1.2%). A skill therefore only scores
  when corroborated by title or career-history text; uncorroborated
  skills get zero credit. This kills keyword stuffers without discarding
  a genuinely discriminative feature.
- Honeypots (~80 in pool, 68 detectable): three record-internal checks.
  Closed stints are clean (tolerance +/-1 month is REQUIRED: 64,649 of
  200,171 closed stints are off by exactly one month of calendar
  rounding; beyond one month: zero).
- No company founding-year checks: the generator does not respect
  real-world founding dates (verified: zero stints predate each
  company's in-dataset earliest start).
"""

from __future__ import annotations

import re

from common import (
    AI_NATIVE, CONSULTING, FAANG, NON_TECH_TITLES, PREFERRED_CITIES,
    PRODUCT, REFERENCE_DATE, WELCOME_CITIES, add_months, candidate_text,
    months_between, parse_date,
)

# ---------------------------------------------------------------- skills

RETRIEVAL_SKILLS = {
    "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus", "Elasticsearch",
    "OpenSearch", "pgvector", "Vector Search", "Semantic Search",
    "Embeddings", "Sentence Transformers", "BM25", "Information Retrieval",
    "Information Retrieval Systems", "Indexing Algorithms",
    "Search & Discovery", "Search Backend", "Search Infrastructure",
    "Ranking Systems", "Learning to Rank", "Recommendation Systems",
    "Text Encoders", "Vector Representations", "Haystack", "Content Matching",
}
ML_SKILLS = {
    "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning", "NLP",
    "Natural Language Processing", "Hugging Face Transformers",
    "scikit-learn", "MLOps", "MLflow", "Kubeflow", "Feature Engineering",
}
LLM_SKILLS = {
    "LLMs", "Fine-tuning LLMs", "LoRA", "QLoRA", "PEFT", "RAG",
    "Model Adaptation", "LlamaIndex", "Prompt Engineering",
}
CV_SPEECH_SKILLS = {
    "Computer Vision", "OpenCV", "Object Detection", "Image Classification",
    "YOLO", "CNN", "GANs", "Diffusion Models", "Speech Recognition",
    "ASR", "TTS",
}

# ------------------------------------------------------- text evidence

RX_RETRIEVAL = re.compile(
    r"retriev|ranking|ranker|recommend|relevance|semantic search|"
    r"vector (search|database|index)|embedding|faiss|elasticsearch|"
    r"opensearch|pinecone|weaviate|qdrant|milvus|bm25|hybrid search|"
    r"search (system|engine|infra|quality|backend|platform)|learning.to.rank",
    re.I)
RX_PRODUCTION = re.compile(
    r"production|deployed|shipped|serving|inference|launch|real users|"
    r"at scale|latency|on.call|rolled out", re.I)
RX_EVAL = re.compile(
    r"ndcg|\bmrr\b|recall@|precision@|map@|a/b test|ab.test|offline eval|"
    r"evaluation (framework|harness|metric|pipeline)|relevance judg|"
    r"engagement metric", re.I)
RX_LLM = re.compile(
    r"\bllms?\b|fine.?tun|lora|qlora|\brag\b|retrieval.augmented|"
    r"prompt|transformer|\bbert\b|\bgpt\b", re.I)
RX_NLP = re.compile(r"\bnlp\b|natural language|text classification|"
                    r"named entity|\bner\b|language model", re.I)
RX_CV_SPEECH = re.compile(r"computer vision|object detection|image |speech|"
                          r"\basr\b|\btts\b|robotic", re.I)
RX_RESEARCH = re.compile(r"research|paper|publication|phd|academic|lab\b", re.I)
RX_LANGCHAIN_DEMO = re.compile(r"langchain|gpt wrapper|prompt engineering", re.I)

RX_ML_TITLE = re.compile(
    r"\bml\b|machine learning|ai engineer|ai research|ai specialist|"
    r"data scien|applied scien|nlp|deep learning", re.I)
RX_SEARCH_TITLE = re.compile(r"search|recommend|relevance|ranking", re.I)
RX_TECH_TITLE = re.compile(
    r"engineer|developer|architect|scientist|sre|devops|programmer|"
    r"analyst|technolog", re.I)
RX_RESEARCH_TITLE = re.compile(r"research (scientist|fellow|assistant)|"
                               r"postdoc|professor|phd", re.I)


# ------------------------------------------------------------ honeypots

def honeypot_reasons(c: dict) -> list[str]:
    """Record-internal impossibility checks. Returns human-readable
    reasons (empty list = clean). Finds 68 of the ~80 planted honeypots."""
    reasons = []
    p = c["profile"]

    # 1. "Expert" proficiency with zero months of use.
    for s in c["skills"]:
        if s["proficiency"] == "expert" and s.get("duration_months", 0) == 0:
            reasons.append(f"claims expert {s['name']} with 0 months used")
            break

    # 2. Current job's implied "now" (start + duration) far from the
    #    dataset's frozen clock of 2026-06. Catches future-dated profiles.
    for j in c["career_history"]:
        sd = parse_date(j["start_date"])
        if sd is None:
            continue
        if j["end_date"] is None:
            implied_now = add_months(sd, j["duration_months"])
            drift = abs(months_between(REFERENCE_DATE, implied_now))
            if drift > 2:
                reasons.append(
                    f"current role at {j['company']} implies today is "
                    f"{implied_now:%Y-%m}, not {REFERENCE_DATE:%Y-%m}")
                break

    # 3. Career span contradicting stated years_of_experience by > 3 years.
    starts = [parse_date(j["start_date"]) for j in c["career_history"]]
    starts = [s for s in starts if s]
    if starts:
        span_years = months_between(min(starts), REFERENCE_DATE) / 12
        if abs(span_years - p["years_of_experience"]) > 3:
            reasons.append(
                f"career span {span_years:.0f}y contradicts stated "
                f"{p['years_of_experience']:.0f}y experience")

    return reasons


# ------------------------------------------------------------- fit score

def fit_features(c: dict) -> dict:
    """Structured JD-fit features plus the evidence facts used for
    reasoning generation. Returns raw components; combination happens
    in rank.py so fusion weights live in one place."""
    p = c["profile"]
    title = p["current_title"].lower()
    text = candidate_text(c)
    yoe = p["years_of_experience"]

    facts: dict = {"yoe": yoe, "title": p["current_title"],
                   "company": p["current_company"], "evidence": [],
                   "concerns": []}

    # --- title archetype across history, weighted toward recent roles
    ml_title = bool(RX_ML_TITLE.search(title))
    search_title = bool(RX_SEARCH_TITLE.search(title))
    tech_title = bool(RX_TECH_TITLE.search(title)) and title not in NON_TECH_TITLES
    non_tech = title in NON_TECH_TITLES
    hist_ml = sum(1 for j in c["career_history"] if RX_ML_TITLE.search(j["title"]))
    hist_search = sum(1 for j in c["career_history"] if RX_SEARCH_TITLE.search(j["title"]))

    # --- text evidence (career descriptions + summary)
    ev_retrieval = len(set(RX_RETRIEVAL.findall(text)))
    ev_production = bool(RX_PRODUCTION.search(text))
    ev_eval = bool(RX_EVAL.search(text))
    ev_llm = bool(RX_LLM.search(text))
    ev_nlp = bool(RX_NLP.search(text))
    corroborated = ml_title or search_title or hist_ml or hist_search or ev_retrieval >= 2

    # --- corroboration-gated skills (zero credit if uncorroborated)
    names = {s["name"] for s in c["skills"]}
    sk_retrieval = len(names & RETRIEVAL_SKILLS) if corroborated else 0
    sk_ml = len(names & ML_SKILLS) if corroborated else 0
    sk_llm = len(names & LLM_SKILLS) if corroborated else 0
    has_python = "Python" in names

    # --- career composition
    months_product = months_consulting = months_total = 0
    for j in c["career_history"]:
        m = j["duration_months"]
        months_total += m
        if j["company"] in PRODUCT:
            months_product += m
        if j["company"] in CONSULTING:
            months_consulting += m
    consulting_only = months_total > 0 and months_consulting == months_total
    product_ratio = months_product / months_total if months_total else 0.0
    ai_native_exp = any(j["company"] in AI_NATIVE for j in c["career_history"])
    faang_exp = any(j["company"] in FAANG for j in c["career_history"])

    # --- JD anti-patterns (graded: JD's "will not" = kill in rank.py,
    #     "probably not" = penalty here)
    n_jobs = len(c["career_history"])
    avg_stint = months_total / n_jobs if n_jobs else 0
    job_hopper = n_jobs >= 3 and avg_stint < 20
    research_only = (
        all(RX_RESEARCH_TITLE.search(j["title"]) for j in c["career_history"])
        and not ev_production)
    cv_only = (bool(RX_CV_SPEECH.search(text)) and not ev_nlp
               and not ev_retrieval and not ml_title)
    # "AI experience" that is only recent LangChain/prompt work
    langchain_only = (RX_LANGCHAIN_DEMO.search(text) is not None
                      and not ev_retrieval and yoe < 2)

    # --- experience band: 5-9 preferred, soft outside (JD: "a range,
    #     not a requirement")
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif yoe < 5:
        yoe_score = max(0.0, 1.0 - (5 - yoe) * 0.25)
    else:
        yoe_score = max(0.3, 1.0 - (yoe - 9) * 0.12)

    # --- evidence facts for reasoning
    if ev_retrieval:
        m = RX_RETRIEVAL.search(text)
        facts["evidence"].append("retrieval/ranking work in career history")
    if ev_eval:
        facts["evidence"].append("ranking-evaluation experience (NDCG/A-B style)")
    if ev_llm:
        facts["evidence"].append("LLM/fine-tuning exposure")
    if ai_native_exp:
        facts["evidence"].append("AI-native startup experience")
    if product_ratio > 0.5:
        facts["evidence"].append("mostly product-company career")
    if sk_retrieval:
        facts["sk_retrieval_names"] = sorted(names & RETRIEVAL_SKILLS)[:4]
    if consulting_only:
        facts["concerns"].append("consulting-only career")
    if job_hopper:
        facts["concerns"].append(f"short average stint ({avg_stint:.0f} mo)")
    if cv_only:
        facts["concerns"].append("CV/speech focus without NLP/IR")

    return {
        "ml_title": ml_title, "search_title": search_title,
        "tech_title": tech_title, "non_tech": non_tech,
        "hist_ml": hist_ml, "hist_search": hist_search,
        "ev_retrieval": ev_retrieval, "ev_production": ev_production,
        "ev_eval": ev_eval, "ev_llm": ev_llm,
        "sk_retrieval": sk_retrieval, "sk_ml": sk_ml, "sk_llm": sk_llm,
        "has_python": has_python, "product_ratio": product_ratio,
        "ai_native": ai_native_exp, "faang": faang_exp,
        "consulting_only": consulting_only, "research_only": research_only,
        "job_hopper": job_hopper, "cv_only": cv_only,
        "langchain_only": langchain_only, "yoe_score": yoe_score,
        "facts": facts,
    }


# ----------------------------------------------------------- availability

def availability(c: dict) -> tuple[float, dict]:
    """Behavioral multiplier in [0.25, 1.0]. Multiplicative per the
    signals doc ('as a multiplier or modifier on top of skill-match
    scoring'). Floor kept low so behavioral twins actually separate."""
    sig = c.get("redrob_signals", {})
    p = c["profile"]
    facts = {}

    la = parse_date(sig.get("last_active_date"))
    days_inactive = (REFERENCE_DATE - la).days if la else 240
    f_recency = 2.718 ** (-days_inactive / 120.0)          # 1.0 fresh -> .13 @8mo
    facts["days_inactive"] = days_inactive

    rr = sig.get("recruiter_response_rate")
    f_response = 0.25 + 0.75 * (rr if rr is not None else 0.3)
    facts["response_rate"] = rr

    f_open = 1.0 if sig.get("open_to_work_flag") else 0.75

    notice = sig.get("notice_period_days", 60)
    if notice <= 30:
        f_notice = 1.0          # JD: can buy out up to 30 days
    elif notice <= 60:
        f_notice = 0.85
    elif notice <= 90:
        f_notice = 0.72
    else:
        f_notice = 0.6
    facts["notice"] = notice

    loc = (p.get("location") or "").lower()
    country = p.get("country", "")
    relocate = sig.get("willing_to_relocate", False)
    if country == "India" and any(x in loc for x in PREFERRED_CITIES):
        f_loc = 1.0
    elif country == "India" and any(x in loc for x in WELCOME_CITIES):
        f_loc = 0.95
    elif country == "India":
        f_loc = 0.9 if relocate else 0.78
    else:
        f_loc = 0.62 if relocate else 0.45  # no visa sponsorship
    facts["location"] = f"{p.get('location')}, {country}"
    facts["relocate"] = relocate

    icr = sig.get("interview_completion_rate")
    f_icr = 0.7 + 0.3 * icr if icr is not None else 0.9

    # Weighted geometric mean, not a raw product: six sub-unity factors
    # multiplied directly pin ~46% of the pool to the floor (measured),
    # destroying the dynamic range that separates behavioral twins.
    raw = (f_recency ** 0.25) * (f_response ** 0.25) * (f_open ** 0.15) \
        * (f_notice ** 0.15) * (f_loc ** 0.15) * (f_icr ** 0.05)
    return max(0.25, min(1.0, raw)), facts
