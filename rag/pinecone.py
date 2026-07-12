from pinecone import Pinecone, ServerlessSpec
import os

class PineconeVectorStore:
    """Vector store using Pinecone."""

    def __init__(self, index_name="default", dimension=768, api_key=None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name

        # Create index if it doesn't exist
        if index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(index_name)

    def add_documents(self, documents, embeddings, ids=None, metadatas=None):
        """Add documents with embeddings."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        vectors = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            vector = {
                "id": ids[i],
                "values": emb.tolist() if hasattr(emb, 'tolist') else emb,
                "metadata": metadatas[i] if metadatas else {"text": doc}
            }
            vectors.append(vector)

        self.index.upsert(vectors=vectors)

    def search(self, query_embedding, top_k=5):
        """Search for similar documents."""
        results = self.index.query(
            vector=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results

    def delete(self, ids):
        """Delete documents by ID."""
        self.index.delete(ids=ids)

    def get_stats(self):
        """Get index statistics."""
        return self.index.describe_index_stats()
