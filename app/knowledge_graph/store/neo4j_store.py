from __future__ import annotations
import logging
from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers.llm import GraphDocument
from app.core.config import PipelineConfig

LOGGER = logging.getLogger("insightgraph")

def sync_to_neo4j(graph_documents: List[GraphDocument], cfg: PipelineConfig) -> bool:
    try:
        graph = Neo4jGraph(url=cfg.neo4j_url, username=cfg.neo4j_user, password=cfg.neo4j_password)
        graph.add_graph_documents(graph_documents)
        return True
    except Exception as e:
        LOGGER.warning("Neo4j Error: %s", e)
        return False
