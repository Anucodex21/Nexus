from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os

class PDFChat:
    """Chat with PDF documents using RAG."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.vectorstore = None
        self.qa_chain = None

    def load_pdf(self, pdf_path):
        """Load and process a PDF file."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings
        )

        llm = ChatOpenAI(api_key=self.api_key, temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )

        return len(texts)

    def ask(self, question):
        """Ask a question about the loaded PDF."""
        if not self.qa_chain:
            raise ValueError("No PDF loaded. Call load_pdf() first.")

        response = self.qa_chain.invoke({"query": question})
        return response["result"]

    def get_relevant_chunks(self, query, k=3):
        """Get relevant text chunks for a query."""
        if not self.vectorstore:
            raise ValueError("No PDF loaded.")

        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
