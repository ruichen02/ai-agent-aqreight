from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class IngestResponse(BaseModel):
    indexed_docs: int
    indexed_chunks: int

class AskRequest(BaseModel):
    query: str
    k: int | None = 2

class Citation(BaseModel):
    title: str
    section: str | None = None

class Chunk(BaseModel):
    title: str
    section: str | None = None
    text: str

class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    chunks: List[Chunk]
    metrics: Dict[str, Any]

class MetricsResponse(BaseModel):
    total_docs: int
    total_chunks: int
    embedding_model: str
    llm_model: str
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    last_query_timestamp: float | None = None
    last_ingest_timestamp: float | None = None
