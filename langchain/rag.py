from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

class LangChainRAG:
    """Retrieval-Augmented Generation with LangChain."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.vectorstore = None
        self.qa_chain = None

        self.custom_prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know.

            Context:
            {context}

            Question: {question}
            Answer:""",
            input_variables=["context", "question"]
        )

    def add_documents(self, documents):
        """Add documents to the vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
        else:
            self.vectorstore.add_documents(texts)

        self._build_chain()
        return len(texts)

    def _build_chain(self):
        """Build the QA chain."""
        llm = ChatOpenAI(api_key=self.api_key, temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.custom_prompt}
        )

    def query(self, question):
        """Query the RAG system."""
        if not self.qa_chain:
            raise ValueError("No documents added. Call add_documents() first.")

        response = self.qa_chain.invoke({"query": question})
        return {
            "answer": response["result"],
            "sources": [doc.page_content for doc in response.get("source_documents", [])]
        }

    def similarity_search(self, query, k=5):
        """Perform similarity search."""
        if not self.vectorstore:
            raise ValueError("No documents added.")

        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
