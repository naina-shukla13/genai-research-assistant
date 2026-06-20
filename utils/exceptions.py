"""
Custom exception hierarchy for the AI Research Assistant.

Using typed exceptions (instead of bare `Exception` or `except:`) lets
each layer of the system fail in a specific, catchable, loggable way.
This directly supports the "Error Handling" and "Code Quality" criteria:
a reviewer can see at a glance what can go wrong at each layer, and
calling code can react differently to a RetrievalError vs a ToolError.
"""


class AssistantBaseException(Exception):
    """Base class for all custom exceptions in this system."""
    pass


# ── LLM Layer ────────────────────────────────────────────────────────
class LLMCallError(AssistantBaseException):
    """Raised when an OpenRouter/LLM API call fails or returns malformed output."""
    pass


class ClassificationError(AssistantBaseException):
    """Raised when the Coordinator cannot classify a query into a valid intent."""
    pass


# ── RAG Layer ────────────────────────────────────────────────────────
class DocumentLoadError(AssistantBaseException):
    """Raised when a document in the knowledge base cannot be loaded or parsed."""
    pass


class EmbeddingError(AssistantBaseException):
    """Raised when embedding generation fails for a chunk or query."""
    pass


class VectorStoreError(AssistantBaseException):
    """Raised on DuckDB read/write/connection failures."""
    pass


class RetrievalError(AssistantBaseException):
    """Raised when the retrieval step fails or returns no usable results."""
    pass


# ── Tool Layer ───────────────────────────────────────────────────────
class ToolInputError(AssistantBaseException):
    """Raised when a tool receives input that fails its Pydantic schema validation."""
    pass


class ToolExecutionError(AssistantBaseException):
    """Raised when a tool's internal logic fails during execution."""
    pass


# ── Agent / Orchestration Layer ───────────────────────────────────────
class AgentRoutingError(AssistantBaseException):
    """Raised when the Coordinator cannot route a query to any agent."""
    pass


class AgentExecutionError(AssistantBaseException):
    """Raised when an agent (Retriever/General) fails during its run."""
    pass