from __future__ import annotations

import re
import json
import logging
from typing import List, Tuple
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_community.graphs.graph_document import Node, Relationship

from app.core.config import PipelineConfig
from app.knowledge_graph.llm.prompts import DIRECT_PROMPT

LOGGER = logging.getLogger("insightgraph")

def _extract_json_array(raw: str) -> List[dict]:
    raw = (raw or "").strip()

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1).strip()

    start = raw.find("[")
    if start < 0:
        return []

    depth = 0
    end = None
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return []

    candidate = raw[start:end]
    try:
        out = json.loads(candidate)
        return out if isinstance(out, list) else []
    except Exception:
        return []

def direct_extract_to_graph(llm: ChatGroq, text: str, cfg: PipelineConfig) -> Tuple[List[Node], List[Relationship]]:
    try:
        msg = llm.invoke([HumanMessage(content=DIRECT_PROMPT + "\n\nText:\n" + text[: cfg.max_chunk_chars_for_llm])])
        raw = msg.content if hasattr(msg, "content") else str(msg)
        parsed = _extract_json_array(raw)

        nodes_set = set()
        rels: List[Relationship] = []

        for rel in parsed:
            if not isinstance(rel, dict):
                continue
            h = (rel.get("head") or "").strip()
            t = (rel.get("tail") or "").strip()
            r = (rel.get("relation") or "").strip()
            if not (h and t and r):
                continue

            ht = (rel.get("head_type") or "Concept").strip()
            tt = (rel.get("tail_type") or "Concept").strip()
            r = r.upper().replace(" ", "_")

            nodes_set.add((h, ht))
            nodes_set.add((t, tt))
            rels.append(Relationship(
                source=Node(id=h, type=ht),
                target=Node(id=t, type=tt),
                type=r,
            ))

        nodes = [Node(id=n, type=t) for (n, t) in nodes_set]
        return nodes, rels

    except Exception as e:
        LOGGER.warning("Direct extraction failed: %s", e)
        return [], []
