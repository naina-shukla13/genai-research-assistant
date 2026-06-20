"""
Query classification — the core of "avoid unnecessary RAG calls."

This is a standalone module (not buried inside the Coordinator class)
so it can be unit-tested and reasoned about independently: given a query,
what intent does it produce, and why. The Coordinator imports this but
contains no classification logic itself.
"""

from config.settings import settings
from core.schemas import ClassificationResult, QueryIntent
from llm.openrouter_client import openrouter_client
from utils.exceptions import ClassificationError
from utils.logger import trace_step, log_event


CLASSIFICATION_SYSTEM_PROMPT = """You are a query classification system for an AI research assistant.

The knowledge base contains a GenAI Intern case study document describing:
- Project requirements (RAG pipeline, multi-agent architecture, tool usage)
- Evaluation criteria for the project
- Deliverables and common pitfalls

Classify the user's query into EXACTLY ONE of these intents:

- KNOWLEDGE_LOOKUP: The query asks about specific information that would
  be found in the knowledge base described above (requirements, evaluation
  criteria, deliverables, project specifications).

- COMPUTATION: The query requires a mathematical calculation
  (arithmetic, percentages, conversions, numeric reasoning).

- GENERAL_REASONING: The query is conversational, opinion-based, requires
  general world knowledge not tied to the specific knowledge base, or
  requires looking up current/external information (e.g. news, web facts).

Be conservative with KNOWLEDGE_LOOKUP — only choose it if the query plausibly
relates to documents in the knowledge base described above. Do not default to it.

Respond with your classification, a confidence score (0.0-1.0), and a
brief one-sentence reasoning for your choice."""


def classify_query(query: str) -> ClassificationResult:
    """
    Calls the LLM with a strict JSON schema to classify the query.
    Raises ClassificationError if the model cannot produce a valid result
    even after the client's internal retries.
    """
    with trace_step("Classifier", "classify_query", query=query):
        try:
            result = openrouter_client.chat_structured(
                model=settings.coordinator_model,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                user_prompt=f"Query: {query}",
                output_schema=ClassificationResult,
            )
            log_event(
                "Classifier", "query_classified",
                query=query, intent=result.intent.value,
                confidence=result.confidence, reasoning=result.reasoning,
            )
            return result
        except Exception as exc:
            raise ClassificationError(
                f"Failed to classify query '{query}': {exc}"
            ) from exc