import os
import chromadb
from chromadb.config import Settings
import numpy as np

class ChromaVectorStore:
    """Vector store using ChromaDB, persisted to disk.

    NOTE: chromadb.Client(Settings(persist_directory=...)) creates an
    EPHEMERAL in-memory client on chromadb>=0.4 - the Settings-based
    persist_directory field no longer does anything there, so every
    collection silently vanished on process restart. PersistentClient is
    the real disk-backed client and is what actually honors
    persist_directory.
    """

    def __init__(self, collection_name="default", persist_directory="./chroma_db"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents, embeddings, ids=None, metadatas=None):
        """Add documents with embeddings."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, query_embedding, n_results=5):
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding],
            n_results=n_results
        )
        return results

    def delete(self, ids):
        """Delete documents by ID."""
        self.collection.delete(ids=ids)

    def get_collection_stats(self):
        """Get collection statistics."""
        return self.collection.count()
