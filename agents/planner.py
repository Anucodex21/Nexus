from typing import List, Dict
import json

class TaskPlanner:
    """Plan and decompose tasks for agents."""

    def __init__(self):
        self.plans = []

    def decompose_task(self, task: str) -> List[Dict]:
        """Decompose a complex task into subtasks."""
        # Simple heuristic-based decomposition
        subtasks = []

        if "research" in task.lower():
            subtasks.append({"step": 1, "action": "search", "description": f"Search for information about: {task}"})
            subtasks.append({"step": 2, "action": "summarize", "description": "Summarize findings"})

        if "write" in task.lower() or "create" in task.lower():
            subtasks.append({"step": len(subtasks)+1, "action": "draft", "description": f"Create draft for: {task}"})
            subtasks.append({"step": len(subtasks)+1, "action": "review", "description": "Review and refine"})

        if "analyze" in task.lower():
            subtasks.append({"step": len(subtasks)+1, "action": "gather_data", "description": "Gather relevant data"})
            subtasks.append({"step": len(subtasks)+1, "action": "analyze", "description": "Perform analysis"})

        if not subtasks:
            subtasks.append({"step": 1, "action": "execute", "description": task})

        return subtasks

    def create_plan(self, goal: str, constraints: List[str] = None) -> Dict:
        """Create a structured plan."""
        plan = {
            "goal": goal,
            "constraints": constraints or [],
            "steps": self.decompose_task(goal),
            "estimated_time": len(self.decompose_task(goal)) * 10  # minutes
        }
        self.plans.append(plan)
        return plan

    def optimize_plan(self, plan: Dict) -> Dict:
        """Optimize a plan by removing redundant steps."""
        seen_actions = set()
        optimized_steps = []

        for step in plan["steps"]:
            if step["action"] not in seen_actions:
                seen_actions.add(step["action"])
                optimized_steps.append(step)

        plan["steps"] = optimized_steps
        return plan

    def get_all_plans(self):
        """Get all created plans."""
        return self.plans
