from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

class ChatBot:
    """Conversational AI chatbot using LangChain."""

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=self.api_key
        )
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

    def chat(self, message):
        """Send a message and get a response."""
        response = self.chain.predict(input=message)
        return response

    def chat_with_system(self, message, system_prompt):
        """Chat with a custom system prompt."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        response = self.llm.invoke(messages)
        return response.content

    def get_history(self):
        """Get conversation history."""
        return self.memory.load_memory_variables({})

    def clear_history(self):
        """Clear conversation history."""
        self.memory.clear()
