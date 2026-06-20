"""
Abstract base class for all agents (Retriever, General).

Why a base class: Coordinator is NOT a subclass of this — it's a router,
not a domain agent — which is itself a design statement. Retriever and
General share: an LLM client, a model name, structured logging, and a
common `run()` contract returning AgentResponse. This avoids duplicating
boilerplate while keeping each agent's actual *logic* fully separate.
"""

from abc import ABC, abstractmethod

from core.schemas import AgentResponse
from llm.openrouter_client import openrouter_client
from utils.logger import trace_step, log_event
from utils.exceptions import AgentExecutionError


class BaseAgent(ABC):
    """Every concrete agent must implement `run(query)` and return an AgentResponse."""

    agent_name: str = "BaseAgent"
    model_name: str = ""

    def __init__(self) -> None:
        self.llm = openrouter_client

    @abstractmethod
    def run(self, query: str) -> AgentResponse:
        """Execute the agent's logic for a given query and return a structured response."""
        raise NotImplementedError

    def safe_run(self, query: str) -> AgentResponse:
        """
        Wraps `run()` with consistent tracing and error translation, so the
        Coordinator only ever has to handle one exception type
        (AgentExecutionError) regardless of which concrete agent failed
        or why.
        """
        with trace_step(self.agent_name, "agent_run", query=query):
            try:
                return self.run(query)
            except Exception as exc:
                log_event(
                    self.agent_name, "agent_run_failed",
                    query=query, error=str(exc), error_type=type(exc).__name__,
                )
                raise AgentExecutionError(
                    f"{self.agent_name} failed on query '{query}': {exc}"
                ) from exc