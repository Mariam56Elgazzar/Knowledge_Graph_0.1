from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_chunk_chars_for_llm: int = 2000  

    max_total_chunks: int = 12
    prioritize_top_k: int = 12

    max_concurrent_chunks: int = 1
    max_retries: int = 4
    retry_base_delay: float = 2.0

    max_direct_passes: int = 2
    min_relationships_target: int = 25    
