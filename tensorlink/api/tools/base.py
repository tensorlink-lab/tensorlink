"""
Types shared by the agentic tool loop (agents.py) and individual tools
(websearch.py, and later file fetch etc.). Kept free of imports from the rest
of the API so both sides can depend on it without cycles.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    # Source-style results: [{"title", "url", "snippet"}]. The loop numbers these
    # across all tool calls so the model can cite them as [n].
    items: List[Dict[str, str]] = field(default_factory=list)
    # Free-form output for tools that don't return sources.
    text: str = ""
    # Set instead of raising; the loop records it and still answers.
    error: Optional[str] = None


@dataclass
class Tool:
    name: str
    signature: str  # e.g. "web_search(query)"
    description: str  # one line shown to the model
    example_arguments: Dict[str, Any]  # used in the prompt's JSON example
    run: Callable[[Dict[str, Any]], Awaitable[ToolResult]]
    # Arguments to use when the user forced the tool on but the model's
    # decision couldn't be parsed. Receives the user's message.
    default_arguments: Callable[[str], Dict[str, Any]]
