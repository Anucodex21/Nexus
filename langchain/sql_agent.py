from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
import os

class SQLAgent:
    """Natural language to SQL agent."""

    def __init__(self, database_url=None, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.database_url = database_url or os.getenv("DATABASE_URL")

        self.db = SQLDatabase.from_uri(self.database_url)
        self.llm = ChatOpenAI(api_key=self.api_key, temperature=0)
        self.chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=self.db,
            verbose=True
        )

    def query(self, question):
        """Ask a natural language question about the database."""
        response = self.chain.invoke(question)
        return response["result"]

    def get_schema(self):
        """Get database schema information."""
        return self.db.get_table_info()

    def execute_sql(self, sql_query):
        """Execute raw SQL query."""
        return self.db.run(sql_query)
