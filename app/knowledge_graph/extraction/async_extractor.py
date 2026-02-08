from __future__ import annotations

import asyncio
import random
import logging
from typing import List
from langchain_core.documents import Document
from langchain_experimental.graph_transformers.llm import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq

from app.core.config import PipelineConfig
from app.knowledge_graph.extraction.direct_extractor import direct_extract_to_graph

LOGGER = logging.getLogger("insightgraph")

async def _process_one_chunk(i: int, chunk: str, transformer: LLMGraphTransformer, cfg: PipelineConfig, sem: asyncio.Semaphore):
    async with sem:
        from app.knowledge_graph.llm.prompts import RESEARCH_PAPER_INSTRUCTIONS
        docs = [Document(page_content=RESEARCH_PAPER_INSTRUCTIONS + "\n\nTEXT:\n" + chunk[: cfg.max_chunk_chars_for_llm])]
        for attempt in range(cfg.max_retries):
            try:
                gdocs = await transformer.aconvert_to_graph_documents(docs)
                return i, gdocs
            except Exception as e:
                if attempt == cfg.max_retries - 1:
                    LOGGER.warning("Chunk %d failed after %d retries: %s", i + 1, cfg.max_retries, e)
                    return i, []
                delay = cfg.retry_base_delay * (2 ** attempt) + random.random() * 0.2
                await asyncio.sleep(delay)
    return i, []

async def extract_graph_data_from_chunks(
    chunks: List[str],
    transformer: LLMGraphTransformer,
    llm: ChatGroq,
    cfg: PipelineConfig,
) -> List[GraphDocument]:
    sem = asyncio.Semaphore(cfg.max_concurrent_chunks)
    tasks = [_process_one_chunk(i, ch, transformer, cfg, sem) for i, ch in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    nodes = []
    rels = []

    for r in results:
        if isinstance(r, Exception):
            LOGGER.warning("Task exception: %s", r)
            continue
        _, gdocs = r
        for g in gdocs or []:
            nodes.extend(getattr(g, "nodes", []) or [])
            rels.extend(getattr(g, "relationships", []) or [])

    combined = "\n\n".join(chunks)

    if len(rels) < cfg.min_relationships_target and len(combined) > 800:
        LOGGER.info("Few relations (%d). Running combined-pass transformer fallback...", len(rels))
        for offset in range(0, min(len(combined), 2 * cfg.max_chunk_chars_for_llm), cfg.max_chunk_chars_for_llm):
            passage = combined[offset: offset + cfg.max_chunk_chars_for_llm]
            try:
                gdocs = await transformer.aconvert_to_graph_documents([Document(page_content=passage)])
                for g in gdocs or []:
                    nodes.extend(g.nodes)
                    rels.extend(g.relationships)
            except Exception as e:
                LOGGER.warning("Transformer fallback failed: %s", e)

    if len(combined) > 500:
        LOGGER.info("Running bounded direct LLM extraction supplement...")
        step = 2200
        size = 5400
        passes = 0
        for offset in range(0, len(combined), step):
            if passes >= cfg.max_direct_passes:
                break
            passage = combined[offset: offset + size]
            if len(passage.strip()) < 250:
                continue
            dn, dr = direct_extract_to_graph(llm, passage, cfg)
            nodes.extend(dn)
            rels.extend(dr)
            passes += 1

    return [GraphDocument(nodes=nodes, relationships=rels, source=Document(page_content="Merged chunks"))]
