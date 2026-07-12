from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
import os

class ConversationMemory:
    """Manage conversation memory for chatbots."""

    def __init__(self, memory_type="buffer", window_size=5, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.memory_type = memory_type

        if memory_type == "buffer":
            self.memory = ConversationBufferMemory()
        elif memory_type == "window":
            self.memory = ConversationBufferWindowMemory(k=window_size)
        elif memory_type == "summary":
            llm = ChatOpenAI(api_key=self.api_key)
            self.memory = ConversationSummaryMemory(llm=llm)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    def add_user_message(self, message):
        """Add user message to memory."""
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message):
        """Add AI message to memory."""
        self.memory.chat_memory.add_ai_message(message)

    def get_memory(self):
        """Get current memory state."""
        return self.memory.load_memory_variables({})

    def clear(self):
        """Clear memory."""
        self.memory.clear()
