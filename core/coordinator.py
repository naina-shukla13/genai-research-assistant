"""
Coordinator Agent — routes queries based on classification only.

This is the centerpiece of the "real multi-agent vs fake multi-agent"
distinction. The Coordinator:
  1. Classifies the query (via core/classifier.py — not inline logic)
  2. Routes to exactly one domain agent based on that classification
  3. Synthesizes the FinalResponse shape required by the case study

It deliberately contains NO retrieval logic, NO tool logic, and NO
answer-generation logic of its own. If you delete RetrieverAgent and
GeneralAgent, the Coordinator can't answer anything — proof that the
separation is real, not cosmetic.
"""

import time

from core.classifier import classify_query
from core.schemas import FinalResponse, QueryIntent
from agents.retriever_agent import RetrieverAgent
from agents.general_agent import GeneralAgent
from utils.exceptions import (
    ClassificationError,
    AgentExecutionError,
    AgentRoutingError,
)
from utils.logger import trace_step, log_event


class Coordinator:
    def __init__(self) -> None:
        self._retriever_agent = RetrieverAgent()
        self._general_agent = GeneralAgent()

        # Both COMPUTATION and GENERAL_REASONING route to the same agent,
        # since GeneralAgent already decides internally whether a tool
        # (calculator or web search) is needed. The Coordinator doesn't
        # need a third agent for this — that would be over-engineering
        # for the sake of agent count, not for the sake of design.
        self._routing_table = {
            QueryIntent.KNOWLEDGE_LOOKUP: self._retriever_agent,
            QueryIntent.COMPUTATION: self._general_agent,
            QueryIntent.GENERAL_REASONING: self._general_agent,
        }

    def handle_query(self, query: str) -> FinalResponse:
        start_time = time.perf_counter()

        with trace_step("Coordinator", "handle_query", query=query):
            # Step 1: Classify
            try:
                classification = classify_query(query)
            except ClassificationError as exc:
                log_event("Coordinator", "classification_failed_fallback", query=query, error=str(exc))
                # Graceful degradation: if classification itself fails,
                # fall back to GeneralAgent rather than crashing the
                # whole request — better a possibly-imperfect answer
                # than no answer.
                classification = None

            intent = classification.intent if classification else QueryIntent.GENERAL_REASONING
            agent = self._routing_table.get(intent)

            if agent is None:
                raise AgentRoutingError(f"No agent registered for intent: {intent}")

            log_event(
                "Coordinator", "query_routed",
                query=query, intent=intent.value,
                routed_to=agent.agent_name,
                classification_confidence=classification.confidence if classification else None,
            )

            # Step 2: Delegate to the chosen agent
            try:
                agent_response = agent.safe_run(query)
            except AgentExecutionError as exc:
                log_event("Coordinator", "agent_execution_failed", query=query, error=str(exc))
                # Graceful degradation: surface a clear, honest failure
                # message rather than propagating a stack trace to the user.
                total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return FinalResponse(
                    query=query,
                    final_answer=(
                        "I encountered an error trying to answer this query. "
                        "Please try rephrasing it or try again shortly."
                    ),
                    handled_by=agent.agent_name,
                    intent=intent,
                    total_latency_ms=total_latency_ms,
                )

            # Step 3: Synthesize final structured response
            total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            final_response = FinalResponse(
                query=query,
                final_answer=agent_response.answer,
                handled_by=agent_response.agent_name,
                intent=intent,
                retrieved_chunks=agent_response.retrieved_chunks,
                tool_calls=agent_response.tool_calls,
                total_latency_ms=total_latency_ms,
            )

            log_event(
                "Coordinator", "query_completed",
                query=query, handled_by=final_response.handled_by,
                latency_ms=total_latency_ms,
                chunks_returned=len(final_response.retrieved_chunks),
                tools_used=[t.tool_name.value for t in final_response.tool_calls],
            )

            return final_response


# Singleton — import as `from core.coordinator import coordinator`
coordinator = Coordinator()