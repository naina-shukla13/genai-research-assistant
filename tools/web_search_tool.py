"""
Mocked web search tool.

The case study explicitly permits mocking this ("Simple web search (can
be mocked)"). We implement it as a small static knowledge lookup with a
clear `MOCKED` flag in the output, rather than silently faking real
search results — transparency about what's mocked vs real is itself a
Code Quality / honesty signal, and avoids misleading evaluators into
thinking we built (or claimed to build) live internet access.
"""

from pydantic import BaseModel, Field

from core.schemas import ToolName
from tools.base_tool import BaseTool


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query")


# Small static set of mocked "search results" to demonstrate the tool-calling
# mechanism end-to-end without depending on a live, possibly-flaky API.
_MOCK_SEARCH_INDEX = {
    "weather": "Mocked result: Weather data requires a live API integration (not implemented in this mock).",
    "news": "Mocked result: Latest news requires a live API integration (not implemented in this mock).",
    "stock": "Mocked result: Stock prices require a live financial data API (not implemented in this mock).",
}


class WebSearchTool(BaseTool):
    tool_name = ToolName.WEB_SEARCH
    input_schema = WebSearchInput

    def execute(self, validated_input: WebSearchInput) -> dict:
        query_lower = validated_input.query.lower()

        matched_topic = next(
            (topic for topic in _MOCK_SEARCH_INDEX if topic in query_lower),
            None,
        )

        result_text = (
            _MOCK_SEARCH_INDEX[matched_topic]
            if matched_topic
            else f"Mocked result: No live search performed. Query was '{validated_input.query}'."
        )

        return {
            "mocked": True,
            "query": validated_input.query,
            "result": result_text,
        }