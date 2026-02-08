import re
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RankedChunk:
    text: str
    score: float
    idx: int

SECTION_KEYWORDS = [
    "abstract", "introduction", "method", "methods", "approach",
    "experiment", "experiments", "results", "discussion", "conclusion",
    "architecture", "model", "dataset", "training", "evaluation"
]

def simple_chunks(text: str, max_chars: int = 2000, overlap: int = 250) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max_chars - overlap
    return chunks

def score_chunk(chunk: str, query_terms: List[str]) -> float:
    c = chunk.lower()
    score = 0.0

    # section/academic signals
    for k in SECTION_KEYWORDS:
        if k in c:
            score += 0.6

    # query signals (title-like / domain keywords)
    for t in query_terms:
        t = t.lower().strip()
        if t and t in c:
            score += 1.5

    # prefer chunks with citations/equations/fig references (often dense knowledge)
    if re.search(r"\bfig(ure)?\b|\btable\b|\beq\.?\b|\bsection\b", c):
        score += 0.8
    if re.search(r"\[\d+\]|\(\d{4}\)", c):  # citations / years
        score += 0.4

    # penalize very short / very long noise
    L = len(chunk)
    if L < 600:
        score -= 0.8

    return score

def rank_chunks(text: str, query_terms: List[str], max_chars: int, keep_k: int) -> List[str]:
    chunks = simple_chunks(text, max_chars=max_chars, overlap=max(150, max_chars//8))
    ranked: List[RankedChunk] = []
    for i, ch in enumerate(chunks):
        ranked.append(RankedChunk(text=ch, score=score_chunk(ch, query_terms), idx=i))

    ranked.sort(key=lambda x: x.score, reverse=True)
    top = ranked[:keep_k]

    # حافظي على ترتيبهم الطبيعي داخل الورقة (أفضل للـ coherence)
    top.sort(key=lambda x: x.idx)
    return [t.text for t in top]
