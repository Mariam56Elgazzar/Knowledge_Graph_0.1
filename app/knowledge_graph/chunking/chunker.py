from __future__ import annotations
from typing import List
from app.core.config import PipelineConfig
from app.knowledge_graph.preprocessing import split_by_sections, sliding_window_chunks, page_based_chunks

def prioritize_chunks(chunks: List[str]) -> List[str]:
    priority_keywords = [
        "method", "architecture", "model", "transformer", "attention",
        "experiment", "results", "evaluation", "dataset", "benchmark",
        "training", "loss", "approach", "encoder", "decoder",
        "et al", "proposed", "compared", "achieves", "outperforms",
        "baseline", "metric", "accuracy", "f1", "bleu",
        "ablation", "limitation", "hyperparameter", "contribution"
    ]
    scored = [(sum(k in c.lower() for k in priority_keywords), c) for c in chunks]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored]

def make_bounded_chunks(text: str, cfg: PipelineConfig) -> List[str]:
    page_chunks = page_based_chunks(text, min_page_chars=120)
    section_chunks = split_by_sections(text, max_chunk_size=2600, overlap=900)
    sw_chunks = sliding_window_chunks(text, window_size=2400, step=700, max_chunks=40)

    all_chunks = []
    seen = set()
    for c in (page_chunks + section_chunks + sw_chunks):
        cc = (c or "").strip()
        if len(cc) < 200:
            continue
        if cc in seen:
            continue
        seen.add(cc)
        all_chunks.append(cc)

    all_chunks = prioritize_chunks(all_chunks)
    all_chunks = all_chunks[: cfg.prioritize_top_k]
    return all_chunks[: cfg.max_total_chunks]
