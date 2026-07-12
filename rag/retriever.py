import numpy as np
from typing import List, Dict

class VectorRetriever:
    """Basic vector similarity retriever."""

    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query, top_k=5):
        """Retrieve top-k similar documents."""
        query_embedding = self.embedding_model.encode_queries([query])[0]
        results = self.vector_store.search(query_embedding, top_k)
        return results

    def batch_retrieve(self, queries, top_k=5):
        """Retrieve for multiple queries."""
        query_embeddings = self.embedding_model.encode_queries(queries)
        results = []
        for emb in query_embeddings:
            results.append(self.vector_store.search(emb, top_k))
        return results

class HybridRetriever:
    """Hybrid retriever combining vector and keyword search."""

    def __init__(self, vector_store, embedding_model, alpha=0.5):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.alpha = alpha  # Weight for vector search
        self.documents = []

    def add_documents(self, documents):
        """Store documents for keyword search."""
        self.documents = documents

    def keyword_search(self, query, top_k=5):
        """Simple keyword-based search."""
        query_words = set(query.lower().split())
        scores = []

        for doc in self.documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scores.append(overlap / max(len(query_words), 1))

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], scores[i]) for i in top_indices]

    def retrieve(self, query, top_k=5):
        """Hybrid retrieval combining both methods."""
        # Vector search
        query_embedding = self.embedding_model.encode_queries([query])[0]
        vector_results = self.vector_store.search(query_embedding, top_k)

        # Keyword search
        keyword_results = self.keyword_search(query, top_k)

        # Combine results (simplified)
        combined = {
            "vector_results": vector_results,
            "keyword_results": keyword_results
        }

        return combined
