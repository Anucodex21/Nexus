from typing import List, Dict
import os

class CrewAIAgent:
    """Multi-agent system inspired by CrewAI."""

    def __init__(self, name, role, goal, backstory, tools=None):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.memory = []

    def execute_task(self, task_description):
        """Execute a given task."""
        print(f"[{self.name}] Executing: {task_description}")

        # Simulate task execution
        result = f"Task completed by {self.name}: {task_description}"
        self.memory.append({"task": task_description, "result": result})
        return result

    def get_memory(self):
        """Get agent memory."""
        return self.memory

class Crew:
    """A crew of agents working together."""

    def __init__(self, agents: List[CrewAIAgent], tasks: List[Dict]):
        self.agents = agents
        self.tasks = tasks

    def kickoff(self):
        """Start the crew execution."""
        results = []
        for task in self.tasks:
            agent_name = task.get("agent")
            agent = next((a for a in self.agents if a.name == agent_name), None)
            if agent:
                result = agent.execute_task(task["description"])
                results.append(result)
        return results
