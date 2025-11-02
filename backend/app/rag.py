import time, os, math, json, hashlib
from typing import List, Dict, Tuple
import numpy as np
from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm

import re

# ---- Simple local embedder (deterministic) ----
def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.split()]

class LocalEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Hash-based repeatable pseudo-embedding
        h = hashlib.sha1(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "big") % (2**32-1)
        rng = np.random.default_rng(rng_seed)
        v = rng.standard_normal(self.dim).astype("float32")
        # L2 normalize
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

# ---- Vector store abstraction ----
class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)  # [N, d]
        q = query.reshape(1, -1)  # [1, d]
        # cosine similarity
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

class QdrantStore:
    def __init__(self, collection: str, dim: int = 384):
        self.client = QdrantClient(url="http://qdrant:6333", timeout=10.0)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
            )

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        import uuid

        points = []
        for i, (v, m) in enumerate(zip(vectors, metadatas)):
            raw_id = m.get("id") or m.get("hash") or i

            # ✅ Ensure ID is Qdrant-compatible (int or UUID)
            if isinstance(raw_id, str):
                try:
                    # Use if it's already a valid UUID
                    uuid.UUID(raw_id)
                    point_id = raw_id
                except ValueError:
                    # Deterministically convert any string/hash into UUIDv5
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
            else:
                point_id = int(raw_id)

            points.append(qm.PointStruct(id=point_id, vector=v.tolist(), payload=m))

        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True
        )
        out = []
        for r in res:
            out.append((float(r.score), dict(r.payload)))
        return out


# ---- LLM provider ----
class StubLLM:
    def generate(self, query: str, contexts: List[Dict]) -> str:
        # Sort by combined_score descending
        contexts = sorted(contexts, key=lambda x: x.get("combined_score", 0), reverse=True)
        top_contexts = contexts[:4]  # top 4 only

        lines = []
        lines.append("Stub Simulated Answer:\n")

        if not top_contexts:
            lines.append("⚠️ No relevant context found.")
            return "\n".join(lines)

        for i, c in enumerate(top_contexts, start=1):
            title = c.get("title", "Untitled")
            section = c.get("section") or ""
            text = c.get("text", "").strip()

            # Keep and show scores
            vector_score = c.get("score", 0)
            title_match = c.get("title_match", 0)
            section_match = c.get("section_match", 0)
            body_match = c.get("body_match", 0)
            combined_score = c.get("combined_score", 0)

            # ---- Convert Markdown lists to multi-line if not already
            text = re.sub(r"(?<!\n)- ", "\n- ", text)

            lines.append(f"[{i}] Title: {title}")
            if section:
                lines.append(f"Section: {section}")
            
            # Add scores display
            lines.append(f"Scores -> vector: {vector_score:.3f}, title: {title_match:.3f}, "
                         f"section: {section_match:.3f}, body: {body_match:.3f}, combined: {combined_score:.3f}\n")

            lines.append(text)
            lines.append("")  # spacing

        lines.append("📘 *End of Stub Output (simulated answer)*")
        return "\n".join(lines)
    
class OpenAILLM:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """
        🚀 OPTIMIZED: Shorter prompt, limited context, focused instructions
        """
        # Sort by combined_score and take only top 3 contexts (reduced from all)
        contexts = sorted(contexts, key=lambda x: x.get("combined_score", 0), reverse=True)[:4]
        
        # Build compact context string
        context_lines = []
        for i, c in enumerate(contexts, 1):
            title = c.get('title', 'Untitled')
            section = c.get('section') or 'General'
            text = c.get('text', '').strip()
            
            # Truncate context
            text_snippet = text[:1000]
            context_lines.append(f"{i}. [{title} | {section}]\n{text_snippet}")

        context_str = "\n\n".join(context_lines)

        prompt = f"""Answer this question using ONLY the sources below. Be concise (2-3 sentences max).

        Question: {query}

        Sources:
        {context_str}

        Instructions:
        - Answer directly and briefly
        - Cite sources as [Title | Section]
        - If uncertain, say "Not enough information in sources"

        Answer:"""

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Slightly higher for faster sampling
            max_tokens=300,   # Limit output length (was unlimited)
            top_p=0.9         # Nucleus sampling for speed
        )

        return resp.choices[0].message.content
    
# ---- RAG Orchestrator & Metrics ----
class Metrics:
    def __init__(self):
        self.t_retrieval = []
        self.t_generation = []
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.last_query_timestamp = None
        self.last_ingest_timestamp = None

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)

    def add_query(self, success: bool = True):
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        self.last_query_timestamp = time.time()

    def set_ingest_time(self):
        self.last_ingest_timestamp = time.time()

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval)/len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation)/len(self.t_generation) if self.t_generation else 0.0
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "last_query_timestamp": self.last_query_timestamp,
            "last_ingest_timestamp": self.last_ingest_timestamp,
        }

class RAGEngine:
    def __init__(self, auto_ingest: bool = True):
        self.embedder = LocalEmbedder(dim=384)

        # Vector store selection
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=384)
            except Exception:
                self.store = InMemoryStore(dim=384)
        else:
            self.store = InMemoryStore(dim=384)

        # LLM selection
        if settings.llm_provider == "openai" and settings.openai_api_key:
            try:
                self.llm = OpenAILLM(api_key=settings.openai_api_key)
                self.llm_name = "openai:4o-mini"
            except Exception:
                self.llm = StubLLM()
                self.llm_name = "stub"

        else:
            self.llm = StubLLM()
            self.llm_name = "stub"

        self.metrics = Metrics()
        self._doc_titles = set()
        self._chunk_count = 0
        
        # Auto-ingest on initialization
        if auto_ingest:
            self._auto_ingest_data()

    def _auto_ingest_data(self):
        """Automatically ingest data from configured sources on startup."""
        from .ingest import load_documents
        from .rag import build_chunks_from_docs

        print("🔄 Starting auto-ingestion...")

        # Load data directory path
        data_dir = getattr(settings, "data_directory", "./data")
        if not os.path.exists(data_dir):
            print(f"⚠️ Data directory '{data_dir}' not found. Skipping auto-ingestion.")
            return

        # Load documents
        docs = load_documents(data_dir)
        if not docs:
            print("⚠️ No documents found for ingestion.")
            return

        # Build chunks (same logic used in manual ingestion)
        chunk_size = getattr(settings, "chunk_size", 150)
        chunk_overlap = getattr(settings, "chunk_overlap", 30)
        chunks = build_chunks_from_docs(docs, chunk_size, chunk_overlap)

        # Use the exact same ingestion logic as manual upload
        new_docs, new_chunks = self.ingest_chunks(chunks)

        print(f"✅ Auto-ingestion complete: {new_docs} new documents, {new_chunks} chunks ingested.")



    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        vectors = []
        metas = []
        doc_titles_before = set(self._doc_titles)

        for ch in chunks:
            text = ch["text"]
            h = doc_hash(text)
            meta = {
            "id": h,
            "hash": h,
            "file": ch.get("file"),
            "title": ch["title"],
            "section": ch.get("section"),
            "text": text,
            }
            v = self.embedder.embed(text)
            vectors.append(v)
            metas.append(meta)
            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

        self.store.upsert(vectors, metas)
        return (len(self._doc_titles) - len(doc_titles_before), len(metas))

    def _score_context(self, query_words: List[str], meta: Dict) -> Dict:
        """Compute hierarchical match scores and combined score for one context."""
        title_text = meta.get("title", "").lower()
        section_text = (meta.get("section") or "").lower()
        body_text = (meta.get("text") or "").lower()

        title_match = sum(1 for w in query_words if w in title_text)
        section_match = sum(1 for w in query_words if w in section_text)
        body_match = sum(1 for w in query_words if w in body_text[:300])

        w_title, w_section, w_body, w_vector = 1.5, 1.5, 2.0, 1.0

        meta["title_match"] = title_match
        meta["section_match"] = section_match
        meta["body_match"] = body_match
        meta["combined_score"] = (
            w_vector * meta.get("score", 0) +
            w_title * title_match +
            w_section * section_match +
            w_body * body_match
        )
        return meta

    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        t0 = time.time()
        qv = self.embedder.embed(query)
        results = self.store.search(qv, k=20)

        self.metrics.add_retrieval((time.time()-t0)*1000.0)

        stopwords = {"the", "a", "an", "and", "or", "for", "of", "to", "after", "on", "in"}
        query_words = [w.lower() for w in re.findall(r"\w+", query) if w.lower() not in stopwords]

        scored = [self._score_context(query_words, dict(meta, score=float(score))) for score, meta in results]
        scored.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return scored[:k]

    def generate(self, query: str, contexts: List[Dict]) -> str:
        t0 = time.time()
        answer = self.llm.generate(query, contexts)
        self.metrics.add_generation((time.time()-t0)*1000.0)
        return answer

    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": settings.embedding_model,
            "llm_model": self.llm_name,
            **m
        }

# ---- Helpers ----
def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap):
            out.append({
                "file": d.get("file"),  # carry filename through
                "title": d["title"],
                "section": d["section"],
                "text": ch
            })
    return out