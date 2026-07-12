import faiss
import numpy as np

class FAISSVectorStore:
    """Vector store using FAISS."""

    def __init__(self, dimension=768, index_type="flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.documents = []
        self.metadatas = []

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity for normalized vectors)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def add_documents(self, documents, embeddings, metadatas=None):
        """Add documents with embeddings."""
        embeddings = np.array(embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.documents.extend(documents)

        if metadatas:
            self.metadatas.extend(metadatas)

    def search(self, query_embedding, k=5):
        """Search for similar documents."""
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)

        distances, indices = self.index.search(query, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(distances[0][i]),
                    "metadata": self.metadatas[idx] if idx < len(self.metadatas) else None
                })

        return results

    def save(self, path):
        """Save the index."""
        faiss.write_index(self.index, f"{path}/faiss.index")
        np.save(f"{path}/documents.npy", self.documents)

    def load(self, path):
        """Load the index."""
        self.index = faiss.read_index(f"{path}/faiss.index")
        self.documents = np.load(f"{path}/documents.npy", allow_pickle=True).tolist()
