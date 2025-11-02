from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from .models import IngestResponse, AskRequest, AskResponse, MetricsResponse, Citation, Chunk
from .settings import settings
from .ingest import load_documents
from .rag import RAGEngine, build_chunks_from_docs

app = FastAPI(title="AI Policy & Product Helper")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RAG engine ---
engine = RAGEngine()

# --- Health endpoint ---
@app.get("/api/health")
def health():
    """Check service health and dependencies."""
    try:
        # Check vector store connectivity
        if hasattr(engine.store, "client"):
            engine.store.client.get_collection(engine.store.collection)
        status = "ok"
    except Exception as e:
        status = f"error: {e}"
    return {"status": status}

# --- Metrics endpoint ---
@app.get("/api/metrics", response_model=MetricsResponse)
def metrics():
    """Return RAG engine metrics, latencies, and counters."""
    stats = engine.stats()
    return MetricsResponse(**stats)

# --- Ingest endpoint ---
@app.post("/api/ingest", response_model=IngestResponse)
def ingest():
    """Load documents, chunk them, and ingest into vector store."""
    docs = load_documents(settings.data_dir)
    chunks = build_chunks_from_docs(docs, settings.chunk_size, settings.chunk_overlap)
    new_docs, new_chunks = engine.ingest_chunks(chunks)
    return IngestResponse(indexed_docs=new_docs, indexed_chunks=new_chunks)

# --- Ask endpoint ---
@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Retrieve relevant chunks and generate an answer."""
    try:
        ctx = engine.retrieve(req.query, k=req.k or 4)
        answer = engine.generate(req.query, ctx)
        engine.metrics.add_query(success=True)
    except Exception:
        engine.metrics.add_query(success=False)
        raise

    citations = [Citation(title=c.get("title"), section=c.get("section")) for c in ctx]
    chunks = [Chunk(title=c.get("title"), section=c.get("section"), text=c.get("text")) for c in ctx]
    stats = engine.stats()
    return AskResponse(
        query=req.query,
        answer=answer,
        citations=citations,
        chunks=chunks,
        metrics={
            "retrieval_ms": stats["avg_retrieval_latency_ms"],
            "generation_ms": stats["avg_generation_latency_ms"],
        }
    )
