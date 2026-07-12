"""
agents package - each submodule is imported lazily so a missing optional
dependency (e.g. selenium for browser_agent, crewai for crewai.py) doesn't
break importing the whole package. Import what you need directly, e.g.:

    from agents.coding_agent import CodingAgent
"""

__all__ = [
    'CrewAIAgent', 'AutoGenAgent', 'BrowserAgent',
    'TaskPlanner', 'TaskExecutor', 'AgentMemory', 'CodingAgent',
]


def __getattr__(name):
    mapping = {
        'CrewAIAgent': ('.crewai', 'CrewAIAgent'),
        'AutoGenAgent': ('.autogen', 'AutoGenAgent'),
        'BrowserAgent': ('.browser_agent', 'BrowserAgent'),
        'TaskPlanner': ('.planner', 'TaskPlanner'),
        'TaskExecutor': ('.executor', 'TaskExecutor'),
        'AgentMemory': ('.memory', 'AgentMemory'),
        'CodingAgent': ('.coding_agent', 'CodingAgent'),
    }
    if name not in mapping:
        raise AttributeError(f"module 'agents' has no attribute {name!r}")
    module_path, attr = mapping[name]
    import importlib
    try:
        module = importlib.import_module(module_path, __name__)
    except ImportError as e:
        raise ImportError(
            f"Could not load agents{module_path} ({attr}): {e}. "
            "It likely needs an optional dependency installed."
        ) from e
    return getattr(module, attr)
