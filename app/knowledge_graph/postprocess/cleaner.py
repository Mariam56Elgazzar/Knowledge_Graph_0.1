import re
from typing import Tuple, Dict
from app.knowledge_graph.extraction.schema import EntityResult, RelationResult, Entity, Relation

def _norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def dedupe_entities(entities: EntityResult) -> EntityResult:
    seen = {}
    for e in entities.nodes:
        key = _norm(e.id).lower()
        if key not in seen:
            seen[key] = Entity(id=_norm(e.id), type=_norm(e.type))
    return EntityResult(nodes=list(seen.values()))

def dedupe_relations(rel: RelationResult, entities: EntityResult) -> RelationResult:
    valid_ids = set(e.id for e in entities.nodes)
    seen = set()
    cleaned = []
    for r in rel.relationships:
        s = _norm(r.source); t = _norm(r.target); ty = _norm(r.type).upper()
        if s not in valid_ids or t not in valid_ids:
            continue
        if s == t:
            continue
        key = (s, t, ty)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(Relation(source=s, target=t, type=ty, evidence=(r.evidence or None)))
    return RelationResult(relationships=cleaned)
