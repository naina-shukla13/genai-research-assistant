"""
General Agent — handles reasoning and tool use (Calculator, Web Search).

This agent decides FOR ITSELF whether a tool is needed, using a structured
LLM call that outputs a tool decision before generating a final answer —
satisfying "Agents should decide when to call tools" rather than the
Coordinator hardcoding "if COMPUTATION intent, always call calculator."
That distinction matters: intent classification routes which AGENT
handles the query; the agent itself still reasons about WHICH tool (if
any) to invoke, and with what input.
"""

from typing import Optional
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from config.settings import settings
from core.schemas import AgentResponse, ToolCallResult, ToolName
from llm.openrouter_client import openrouter_client
from tools.calculator_tool import CalculatorTool
from tools.web_search_tool import WebSearchTool
from utils.logger import trace_step, log_event


class ToolDecision(BaseModel):
    needs_tool: bool
    tool_name: Optional[ToolName] = None
    tool_input: Optional[dict] = None
    reasoning: str = Field(..., description="Why a tool is or isn't needed")


TOOL_DECISION_PROMPT = """You are deciding whether a tool is needed to answer a query.

Available tools:
- calculator: input {"expression": "<arithmetic expression>"} — use for any math.
- web_search: input {"query": "<search query>"} — use for queries needing
  external/current information not derivable from reasoning alone.

If neither tool is needed (the query can be answered through reasoning
alone), set needs_tool to false.

Respond with your decision and brief reasoning."""

GENERAL_ANSWER_PROMPT = """You are a helpful, precise reasoning assistant.
Answer the user's query directly and concisely. If tool output is provided,
incorporate it accurately into your answer."""


class GeneralAgent(BaseAgent):
    agent_name = "GeneralAgent"
    model_name = settings.general_agent_model

    def __init__(self) -> None:
        super().__init__()
        self._tools = {
            ToolName.CALCULATOR: CalculatorTool(),
            ToolName.WEB_SEARCH: WebSearchTool(),
        }

    def run(self, query: str) -> AgentResponse:
        tool_calls: list[ToolCallResult] = []

        with trace_step("GeneralAgent", "decide_tool_use", query=query):
            decision = self.llm.chat_structured(
                model=self.model_name,
                system_prompt=TOOL_DECISION_PROMPT,
                user_prompt=f"Query: {query}",
                output_schema=ToolDecision,
            )

        log_event(
            "GeneralAgent", "tool_decision_made",
            query=query, needs_tool=decision.needs_tool,
            tool_name=decision.tool_name.value if decision.tool_name else None,
            reasoning=decision.reasoning,
        )

        tool_output_context = ""
        if decision.needs_tool and decision.tool_name:
            tool = self._tools.get(decision.tool_name)
            if tool:
                result = tool.run(decision.tool_input or {})
                tool_calls.append(result)
                if result.success:
                    tool_output_context = f"\n\nTool '{decision.tool_name.value}' returned: {result.output}"
                else:
                    tool_output_context = f"\n\nTool '{decision.tool_name.value}' failed: {result.error}"

        with trace_step("GeneralAgent", "generate_answer", query=query):
            answer = self.llm.chat(
                model=self.model_name,
                system_prompt=GENERAL_ANSWER_PROMPT,
                user_prompt=f"Query: {query}{tool_output_context}",
            )

        return AgentResponse(
            agent_name=self.agent_name,
            answer=answer,
            tool_calls=tool_calls,
            reasoning_trace=decision.reasoning,
        )