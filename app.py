"""Redrob candidate ranker — hosted sandbox (HuggingFace Spaces / Gradio).

Satisfies submission spec section 10.5: accepts a small candidate sample
(<=100), runs the SAME ranking pipeline end-to-end, and returns a ranked
CSV — within the CPU compute budget. Pipeline logic lives in
sandbox_core.py (shared with the test harness); this file is only the UI.

Run locally:  python app.py
On Spaces:    auto-launched by the Gradio SDK.
"""

from __future__ import annotations

import gradio as gr

from sandbox_core import SAMPLE, rank_from_source


def run(file_obj):
    # gr.File returns a path string (gradio>=4 default) or, on older
    # versions, an object with .name. None when nothing is uploaded.
    if file_obj is None:
        src = str(SAMPLE)
    elif isinstance(file_obj, str):
        src = file_obj
    else:
        src = file_obj.name
    status, df, csv_path = rank_from_source(src)
    return status, df.head(25), csv_path


with gr.Blocks(title="Starva — Redrob Candidate Ranker") as demo:
    gr.Markdown(
        "# Starva — Redrob Candidate Ranker\n"
        "Ranks candidates for the **Senior AI Engineer** JD: deterministic "
        "honeypot checks → corroboration-gated structured features → MiniLM "
        "semantic + BM25 fusion → behavioral availability multiplier → "
        "fact-grounded reasoning. **No LLM at runtime.**\n\n"
        "Upload a candidate `.jsonl` (≤100 rows) or just click **Rank** to "
        "use the bundled 100-candidate sample (55 ML fits, 12 honeypots, "
        "33 generic).")
    with gr.Row():
        up = gr.File(label="candidates.jsonl (optional — ≤100 rows)",
                     file_types=[".jsonl", ".json", ".txt"])
        btn = gr.Button("Rank candidates", variant="primary")
    status = gr.Markdown()
    out_df = gr.Dataframe(label="Top 25 (full top-100 in the CSV)", wrap=True)
    out_csv = gr.File(label="Download ranked_output.csv")
    btn.click(run, inputs=up, outputs=[status, out_df, out_csv])

if __name__ == "__main__":
    demo.launch()
