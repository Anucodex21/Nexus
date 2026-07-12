from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """Wrapper for sentence embedding models."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=32, normalize=True):
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def encode_queries(self, queries):
        """Encode queries (may use different pooling)."""
        return self.encode(queries)

    def similarity(self, embedding1, embedding2):
        """Compute cosine similarity between embeddings."""
        return np.dot(embedding1, embedding2)
