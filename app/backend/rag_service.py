"""
RAGService - wires the existing rag/ module (chunking is added here,
embeddings + Chroma vector store already existed) together with the
multi-provider LLMClient, so /rag/upload and /rag/query in routes.py have
something real to call.

Heavy ML deps (sentence-transformers, chromadb) are only imported the first
time a RAG endpoint is actually hit - same lazy-load pattern llm_client.py
uses for the local model - so the rest of the API keeps working even on a
requirements-web.txt-only install that hasn't pulled those in.

Isolation: each user gets their own Chroma collection, so one user's
uploads never show up in another user's /rag/query results.
"""

import os
import re
from typing import Dict, List

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 120    # characters shared between consecutive chunks


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window chunker over whitespace-normalized text. Good
    enough for .txt/.md/.pdf-extracted text without pulling in a heavier
    text-splitting dependency."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _collection_name(user: str) -> str:
    """One Chroma collection per user so documents/queries never cross users."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", user) or "anon"
    return f"user_{safe}_docs"


class RAGService:
    """Owns one shared embedding model + Chroma client for the process
    lifetime, and hands out a per-user vector store / retriever on demand."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self._embedder = None                    # lazy: rag.embedding.EmbeddingModel
        self._stores: Dict[str, object] = {}      # user -> ChromaVectorStore
        self._retrievers: Dict[str, object] = {}  # user -> VectorRetriever

    # ---------------- lazy loaders ----------------

    def _get_embedder(self):
        if self._embedder is None:
            from rag.embedding import EmbeddingModel
            self._embedder = EmbeddingModel()
        return self._embedder

    def _get_store(self, user: str):
        if user not in self._stores:
            from rag.chroma import ChromaVectorStore
            self._stores[user] = ChromaVectorStore(
                collection_name=_collection_name(user),
                persist_directory=CHROMA_DIR,
            )
        return self._stores[user]

    def _get_retriever(self, user: str):
        if user not in self._retrievers:
            from rag.retriever import VectorRetriever
            self._retrievers[user] = VectorRetriever(self._get_store(user), self._get_embedder())
        return self._retrievers[user]

    # ---------------- public API used by routes.py ----------------

    def add_document(self, user: str, filename: str, text: str) -> int:
        """Chunk + embed + store one document for this user. Returns how
        many chunks were stored (0 if the file had no usable text)."""
        chunks = _chunk_text(text)
        if not chunks:
            return 0

        embedder = self._get_embedder()
        embeddings = embedder.encode(chunks)
        store = self._get_store(user)

        base = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
        ids = [f"{base}_{i}" for i in range(len(chunks))]
        metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

        store.add_documents(chunks, embeddings, ids=ids, metadatas=metadatas)
        return len(chunks)

    def query(self, user: str, question: str, top_k: int = 5, preferred: str = None) -> dict:
        """Retrieve this user's most relevant chunks, then answer using the
        real multi-provider LLMClient - same fallback chain /chat uses, not
        a separate hardcoded call."""
        retriever = self._get_retriever(user)
        raw = retriever.retrieve(question, top_k=top_k)

        documents = raw.get("documents", [[]])[0] if isinstance(raw, dict) else []
        metadatas = raw.get("metadatas", [[]])[0] if isinstance(raw, dict) else []

        if not documents:
            return {
                "answer": (
                    "You haven't uploaded any documents yet, or nothing relevant to "
                    "your question was found. Upload something with /rag/upload first."
                ),
                "sources": [],
                "model": None,
            }

        context = "\n\n---\n\n".join(documents)
        prompt = (
            "Use ONLY the following context to answer the question. "
            "If the answer isn't in the context, say you don't know - don't make anything up.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        messages = [{"role": "user", "content": prompt}]
        answer, provider = self.llm_client.chat(messages, preferred=preferred)

        sources = []
        for meta in metadatas:
            fn = meta.get("filename") if isinstance(meta, dict) else None
            if fn and fn not in sources:
                sources.append(fn)

        return {"answer": answer, "sources": sources, "model": provider}

    def stats(self, user: str) -> dict:
        if user not in self._stores:
            return {"chunks_stored": 0}
        return {"chunks_stored": self._get_store(user).get_collection_stats()}
