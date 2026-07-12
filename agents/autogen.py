from typing import Dict, List

class AutoGenAgent:
    """Conversational agent inspired by AutoGen."""

    def __init__(self, name, system_message, llm_config=None):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config or {}
        self.chat_history = []

    def send(self, message, recipient):
        """Send a message to another agent."""
        print(f"[{self.name}] -> [{recipient.name}]: {message}")

        self.chat_history.append({
            "from": self.name,
            "to": recipient.name,
            "message": message
        })

        # Simulate response
        response = recipient.receive(message, self)
        return response

    def receive(self, message, sender):
        """Receive a message from another agent."""
        print(f"[{self.name}] Received from [{sender.name}]: {message}")

        # Generate response based on system message
        response = f"[{self.name}] Processing your request about: {message[:50]}..."

        self.chat_history.append({
            "from": sender.name,
            "to": self.name,
            "message": message
        })

        return response

    def get_chat_history(self):
        """Get conversation history."""
        return self.chat_history

class GroupChat:
    """Group chat for multiple agents."""

    def __init__(self, agents: List[AutoGenAgent], max_round=10):
        self.agents = agents
        self.max_round = max_round
        self.messages = []

    def initiate_chat(self, initiator, message):
        """Initiate a group chat."""
        self.messages.append({"from": initiator.name, "message": message})

        current_agent = initiator
        for _ in range(self.max_round):
            for agent in self.agents:
                if agent != current_agent:
                    response = agent.receive(message, current_agent)
                    self.messages.append({"from": agent.name, "message": response})
                    message = response
                    current_agent = agent

        return self.messages
