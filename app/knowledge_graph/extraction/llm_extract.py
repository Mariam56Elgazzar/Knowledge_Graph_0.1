import json
import time
import random
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from .schema import EntityResult, RelationResult

def _sleep_backoff(base: float, attempt: int):
    # exponential backoff + jitter
    delay = base * (2 ** attempt) + random.uniform(0.0, 0.5)
    time.sleep(delay)

def _extract_json(text: str) -> str:
    # حاول تلتقط أول JSON object في الرد
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text.strip()

def extract_entities(llm: ChatGroq, chunks: List[str], max_retries: int = 4, backoff: float = 2.0) -> EntityResult:
    system = SystemMessage(content=
        "You extract ONLY entities from scientific text.\n"
        "Return STRICT JSON with key: nodes.\n"
        "Each node: {\"id\": string, \"type\": string}.\n"
        "Rules:\n"
        "- Prefer canonical names (e.g., 'Transformer', 'Self-Attention').\n"
        "- Merge obvious duplicates.\n"
        "- Keep 40-120 nodes max.\n"
        "- No extra keys.\n"
    )
    user = HumanMessage(content="TEXT:\n" + "\n\n---\n\n".join(chunks))

    for attempt in range(max_retries):
        try:
            resp = llm.invoke([system, user]).content
            j = _extract_json(resp)
            data = json.loads(j)
            return EntityResult(**data)
        except Exception:
            if attempt == max_retries - 1:
                raise
            _sleep_backoff(backoff, attempt)

    raise RuntimeError("Entity extraction failed")

def extract_relations(llm: ChatGroq, chunks: List[str], entities: EntityResult,
                      max_retries: int = 4, backoff: float = 2.0) -> RelationResult:
    # ندي للـ model قائمة entities عشان يقلل هلوسة العلاقات
    ent_list = [f"{e.id}::{e.type}" for e in entities.nodes]
    system = SystemMessage(content= """
        You are an information extraction system.

        Given:
        (1) A list of ENTITIES (with types)
        (2) A TEXT PASSAGE

        Task:
        Extract relationships ONLY between the provided entities.
        Return ONLY a JSON array. No markdown. No explanations.

        Each item:
        {"head":"...","head_type":"...","relation":"...","tail":"...","tail_type":"..."}

        Rules:
        - Use ONLY entities from the list (match by name, case-insensitive).
        - relation must be one of:
        ["RELATED_TO","USES","CONTAINS","COMPARED_TO","TRAINED_ON","USED_FOR","IMPLEMENTS",
        "EVALUATES","ACHIEVES","ADDRESSES","RESULTS_IN","PART_OF","CONTRIBUTES_TO",
        "IMPROVES","SUPPORTS","DEPENDS_ON","DESCRIBED_IN","PROPOSES","OBSERVED_IN",
        "EXTENDS","LIMITS","INTRODUCES","CITES"]
        - Output 15-50 relations if possible.
        - Do NOT call any tools or functions.
        """)

    user = HumanMessage(content=
        "ENTITIES:\n" + "\n".join(ent_list) +
        "\n\nTEXT:\n" + "\n\n---\n\n".join(chunks)
    )

    for attempt in range(max_retries):
        try:
            resp = llm.invoke([system, user]).content
            j = _extract_json(resp)
            data = json.loads(j)
            return RelationResult(**data)
        except Exception:
            if attempt == max_retries - 1:
                raise
            _sleep_backoff(backoff, attempt)

    raise RuntimeError("Relation extraction failed")
