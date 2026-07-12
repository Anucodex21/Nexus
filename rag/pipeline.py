from typing import List, Dict, Optional
import os

class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation."""

    def __init__(self, retriever, llm_client=None, api_key=None):
        self.retriever = retriever
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm_client = llm_client

    def retrieve(self, query, top_k=5):
        """Retrieve relevant documents."""
        return self.retriever.retrieve(query, top_k)

    def construct_prompt(self, query, context_documents):
        """Construct prompt with retrieved context."""
        context = "\n\n".join([doc for doc in context_documents])

        prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def generate(self, prompt, max_tokens=500):
        """Generate response using LLM."""
        if self.llm_client:
            response = self.llm_client.generate(prompt, max_tokens=max_tokens)
            return response
        else:
            # Fallback: return prompt for debugging
            return f"[DEBUG] Prompt constructed:\n{prompt}"

    def run(self, query, top_k=5):
        """Run the complete RAG pipeline."""
        # Retrieve relevant documents
        retrieved = self.retrieve(query, top_k)

        # Extract documents from results
        if isinstance(retrieved, dict):
            documents = retrieved.get("vector_results", {}).get("documents", [])
        else:
            documents = retrieved

        # Construct prompt
        prompt = self.construct_prompt(query, documents)

        # Generate response
        response = self.generate(prompt)

        return {
            "query": query,
            "retrieved_documents": documents,
            "prompt": prompt,
            "response": response
        }

    def batch_run(self, queries, top_k=5):
        """Run pipeline for multiple queries."""
        results = []
        for query in queries:
            results.append(self.run(query, top_k))
        return results
