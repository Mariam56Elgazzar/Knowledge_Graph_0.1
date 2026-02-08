from __future__ import annotations

import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer

from app.core.config import PipelineConfig
from app.knowledge_graph.llm.prompts import (
    ALLOWED_NODES, ALLOWED_RELATIONSHIPS, RESEARCH_PAPER_INSTRUCTIONS
)

LOGGER = logging.getLogger("insightgraph")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    LOGGER.warning("GROQ_API_KEY is missing. Set it in .env or environment variables.")

def build_llm(cfg: PipelineConfig) -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        temperature=cfg.temperature,
        model_name=cfg.model_name,
    )

def build_transformer(llm: ChatGroq) -> LLMGraphTransformer:
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        strict_mode=False,
    )

