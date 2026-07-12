from langchain.tools import Tool
from langchain.agents import load_tools
from langchain.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_experimental.tools import PythonREPLTool

class ToolManager:
    """Manage and create tools for agents."""

    def __init__(self):
        self.tools = {}
        self._load_default_tools()

    def _load_default_tools(self):
        """Load default tools."""
        # Wikipedia search
        wiki = WikipediaAPIWrapper()
        self.tools["wikipedia"] = Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Search Wikipedia for information."
        )

        # DuckDuckGo search
        search = DuckDuckGoSearchAPIWrapper()
        self.tools["search"] = Tool(
            name="Web Search",
            func=search.run,
            description="Search the web for current information."
        )

        # Python REPL
        python = PythonREPLTool()
        self.tools["python"] = Tool(
            name="Python",
            func=python.run,
            description="Execute Python code."
        )

    def add_tool(self, name, func, description):
        """Add a custom tool."""
        self.tools[name] = Tool(
            name=name,
            func=func,
            description=description
        )

    def get_tool(self, name):
        """Get a tool by name."""
        return self.tools.get(name)

    def get_all_tools(self):
        """Get all available tools."""
        return list(self.tools.values())

    def list_tools(self):
        """List all tool names."""
        return list(self.tools.keys())
