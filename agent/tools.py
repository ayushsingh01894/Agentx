"""
tools.py
Module 5 - Tool Engineering  +  Module 11 - MCP (Model Context Protocol)

Tools are defined once as plain Python functions, then exposed two ways:
 1. As Groq function-calling JSON schemas (`TOOL_SCHEMAS`)
 2. Through a small MCP-style server (`MCPToolServer`) that mimics how an
    MCP server exposes/executes tools for a client — swap this for the real
    `mcp` SDK in production without changing the agent logic.
"""
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Individual tools
# ---------------------------------------------------------------------------
def web_search(query: str) -> str:
    """Mock web search tool. Replace with a real API (Tavily/Serper/Bing) in production."""
    mock_kb = {
        "ai agents": "AI agents are autonomous systems combining LLMs with planning, tools and memory to complete multi-step goals.",
        "groq": "Groq provides ultra-fast LLM inference hardware (LPU) and an OpenAI-compatible API for open models like Llama.",
        "mcp": "Model Context Protocol (MCP) is an open standard letting AI apps connect to external tools and data sources uniformly.",
    }
    for key, val in mock_kb.items():
        if key in query.lower():
            return val
    return f"No strong match found for '{query}'. (mock search — plug a real search API here)"


def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Invalid expression."
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def save_report(filename: str, content: str) -> str:
    """Save the final report to disk."""
    path = REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Report saved to {path}"


# ---------------------------------------------------------------------------
# Groq function-calling schemas
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web / knowledge base for information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "web_search": web_search,
    "calculator": calculator,
}


# ---------------------------------------------------------------------------
# Minimal MCP-style tool server (concept demo, Module 11)
# ---------------------------------------------------------------------------
class MCPToolServer:
    """Mimics an MCP server: exposes a tool registry and executes calls on request.
    In production, replace this with a real MCP server (`mcp` Python SDK) running
    over STDIO/HTTP/SSE, discovered dynamically by an MCP client."""

    def __init__(self, registry: dict):
        self.registry = registry

    def list_tools(self):
        return list(self.registry.keys())

    def call_tool(self, name: str, arguments: dict):
        if name not in self.registry:
            raise ValueError(f"Tool '{name}' not found on this MCP server.")
        return self.registry[name](**arguments)


mcp_server = MCPToolServer(TOOL_REGISTRY)
