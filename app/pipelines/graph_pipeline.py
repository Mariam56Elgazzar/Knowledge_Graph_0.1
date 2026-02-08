# app/pipelines/graph_pipeline.py
from __future__ import annotations

import os
import logging
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from app.core.config import PipelineConfig

# ✅ IMPORTANT: your module path has a typo "chunck_ranker"
# Keep it if that's how your folder/file is named.
from app.knowledge_graph.chunking.chunck_ranker import rank_chunks

from app.knowledge_graph.llm.groq_client import build_llm
from app.knowledge_graph.extraction.llm_extract import extract_entities, extract_relations
from app.knowledge_graph.postprocess.cleaner import dedupe_entities, dedupe_relations

from langchain_experimental.graph_transformers.llm import GraphDocument


LOGGER = logging.getLogger("insightgraph")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


OUTPUT_DIR = os.path.join("outputs", "graphs")
DEFAULT_HTML = os.path.join(OUTPUT_DIR, "knowledge_graph.html")


def _read_source(source: str, is_path: bool) -> str:
    """Read PDF/TXT from path or return raw text."""
    if not is_path:
        return source or ""

    path = source
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source path not found: {path}")

    # PDF
    if path.lower().endswith(".pdf"):
        # Try your loader if present
        try:
            from app.knowledge_graph.ingestion.pdf_loader import load_pdf_text
            return load_pdf_text(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF via load_pdf_text: {e}")

    # TXT or others
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def _safe_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ensure_html_with_pyvis(
    nodes: List,
    relationships: List,
    html_path: str = DEFAULT_HTML,
) -> str:
    """Fallback HTML writer (Neo4j-like interactive graph) using PyVis."""
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    try:
        from pyvis.network import Network
    except Exception as e:
        raise RuntimeError(
            "PyVis is required for fallback visualization. Install: pip install pyvis"
        ) from e

    net = Network(height="750px", width="100%", directed=True, bgcolor="#ffffff")

    # Add nodes
    for n in nodes or []:
        node_id = _safe_get(n, "id") or _safe_get(n, "name") or str(n)
        node_type = _safe_get(n, "type", "Entity")
        label = str(node_id)
        title = f"{label}\nType: {node_type}"
        net.add_node(str(node_id), label=label, title=title, group=str(node_type))

    # Add edges
    for r in relationships or []:
        s = _safe_get(r, "source_node_id") or _safe_get(r, "source") or _safe_get(r, "from")
        t = _safe_get(r, "target_node_id") or _safe_get(r, "target") or _safe_get(r, "to")
        rel_type = _safe_get(r, "type") or _safe_get(r, "relation") or "RELATED_TO"
        if not s or not t:
            continue
        net.add_edge(str(s), str(t), label=str(rel_type), title=str(rel_type))

    net.set_options("""
    var options = {
      "physics": { "enabled": true, "stabilization": {"iterations": 200} },
      "interaction": { "hover": true, "navigationButtons": true, "keyboard": true },
      "edges": { "smooth": false, "arrows": { "to": { "enabled": true } } }
    }
    """)

    net.write_html(html_path)
    return html_path


def generate_knowledge_graph(
    source: str,
    *,
    is_path: bool = False,
    sync_neo4j: bool = False,
    cfg: Optional[PipelineConfig] = None,
    query_terms=None,
) -> Tuple[Optional[str], List[GraphDocument], bool]:
    """
    ✅ Streamlit-compatible signature:
    Returns: (html_path, graph_docs, sync_status)
    """
    cfg = cfg or PipelineConfig()
    query_terms = query_terms or ["method", "results", "model", "dataset"]

    # 0) read content
    text = _read_source(source, is_path=is_path).strip()
    if not text:
        LOGGER.warning("Empty input text after reading source.")
        return None, [], False

    # Optional preprocess if you have it
    try:
        from app.knowledge_graph.preprocessing import preprocess_text
        text = preprocess_text(text)
    except Exception:
        pass

    # 1) rank/select chunks
    selected_chunks = rank_chunks(
        text,
        query_terms=query_terms,
        max_chars=cfg.max_chunk_chars_for_llm,
        keep_k=min(cfg.prioritize_top_k, cfg.max_total_chunks),
    )
    LOGGER.info(f"Selected {len(selected_chunks)} ranked chunks for LLM")
    if not selected_chunks:
        LOGGER.warning("No chunks selected (rank_chunks returned empty).")
        return None, [], False

    # 2) build LLM
    llm = build_llm(cfg)

    # 3) entities
    ent = extract_entities(
        llm,
        selected_chunks,
        max_retries=cfg.max_retries,
        backoff=cfg.retry_base_delay,
    )
    ent = dedupe_entities(ent)

    # 4) relations
    rel = extract_relations(
        llm,
        selected_chunks,
        ent,
        max_retries=cfg.max_retries,
        backoff=cfg.retry_base_delay,
    )
    rel = dedupe_relations(rel, ent)

    # 5) supplement pass (only 1 extra chunk) if too few relations
    if len(getattr(rel, "relationships", []) or []) < cfg.min_relationships_target:
        LOGGER.info(
            f"Relations ({len(rel.relationships)}) below target ({cfg.min_relationships_target}) -> supplement pass"
        )
        extra_chunks = rank_chunks(
            text,
            query_terms=query_terms + ["conclusion", "experiment", "evaluation"],
            max_chars=cfg.max_chunk_chars_for_llm,
            keep_k=min(min(cfg.prioritize_top_k + 1, 8), cfg.max_total_chunks + 1),
        )
        extra = extra_chunks[-1:] if extra_chunks else []
        if extra:
            rel2 = extract_relations(
                llm,
                extra,
                ent,
                max_retries=cfg.max_retries,
                backoff=cfg.retry_base_delay,
            )
            rel2 = dedupe_relations(rel2, ent)
            rel.relationships.extend(rel2.relationships)
            rel = dedupe_relations(rel, ent)

    nodes = getattr(ent, "nodes", []) or []
    relationships = getattr(rel, "relationships", []) or []
    LOGGER.info(f"Final Nodes: {len(nodes)} | Final Relations: {len(relationships)}")

    # 6) build GraphDocument (what Streamlit expects as graph_docs)

    doc = Document(
        page_content=text,
        metadata={"source": "uploaded_file" if is_path else "manual_text"}
    )

    graph_docs = [GraphDocument(nodes=nodes, relationships=relationships, source=doc)]

    # 7) visualization (try your visualize_graph, else fallback PyVis)
    html_path = None
    try:
        from app.knowledge_graph.visualization import visualize_graph
        # expected to return path OR write default
        html_path = visualize_graph(graph_docs, out_path=DEFAULT_HTML)  # if your func supports it
        if not html_path:
            html_path = DEFAULT_HTML if os.path.exists(DEFAULT_HTML) else None
    except TypeError:
        # visualize_graph signature different
        try:
            from app.knowledge_graph.visualization import visualize_graph
            html_path = visualize_graph(graph_docs)  # old signature
            if not html_path:
                html_path = DEFAULT_HTML if os.path.exists(DEFAULT_HTML) else None
        except Exception as e:
            LOGGER.warning(f"visualize_graph failed, using fallback PyVis. Reason: {e}")
            html_path = _ensure_html_with_pyvis(nodes, relationships, DEFAULT_HTML)
    except Exception as e:
        LOGGER.warning(f"visualize_graph failed, using fallback PyVis. Reason: {e}")
        html_path = _ensure_html_with_pyvis(nodes, relationships, DEFAULT_HTML)

    # 8) Neo4j sync (optional)
    sync_status = False
    if sync_neo4j:
        try:
            from app.knowledge_graph.store.neo4j_store import sync_to_neo4j
            sync_status = bool(sync_to_neo4j(graph_docs, cfg))
        except Exception as e:
            LOGGER.warning(f"Neo4j sync failed: {e}")
            sync_status = False

    return html_path, graph_docs, sync_status
