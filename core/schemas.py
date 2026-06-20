"""
Shared Pydantic schemas used across agents, tools, and the RAG pipeline.

Centralizing these schemas (rather than letting each module define its
own ad-hoc dicts) is what makes "structured inputs/outputs" a real,
enforced property of the system rather than a description in the README.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field


# ── Query Classification ─────────────────────────────────────────────

class QueryIntent(str, Enum):
    KNOWLEDGE_LOOKUP = "KNOWLEDGE_LOOKUP"      # Needs RAG over knowledge base
    COMPUTATION = "COMPUTATION"                 # Needs calculator tool
    GENERAL_REASONING = "GENERAL_REASONING"     # General agent, may use web search tool


class ClassificationResult(BaseModel):
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Why this intent was chosen")


# ── RAG ───────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_document: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class RetrievalResult(BaseModel):
    query: str
    chunks: List[Chunk]
    retrieval_count: int

    @property
    def has_relevant_chunks(self) -> bool:
        return len(self.chunks) > 0


# ── Tools ──────────────────────────────────────────────────────────────

class ToolName(str, Enum):
    CALCULATOR = "calculator"
    WEB_SEARCH = "web_search"


class ToolCallRequest(BaseModel):
    tool_name: ToolName
    tool_input: dict


class ToolCallResult(BaseModel):
    tool_name: ToolName
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None


# ── Agent-Level I/O ──────────────────────────────────────────────────

class AgentResponse(BaseModel):
    """Standard output shape every agent must return to the Coordinator."""
    agent_name: str
    answer: str
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    tool_calls: List[ToolCallResult] = Field(default_factory=list)
    reasoning_trace: Optional[str] = None


# ── Final System Output ───────────────────────────────────────────────

class FinalResponse(BaseModel):
    """
    The complete structured response returned to the user/UI.
    Satisfies the case study's explicit requirement to show:
    - final answer
    - which agent handled the query
    - retrieved chunks (for RAG transparency)
    """
    query: str
    final_answer: str
    handled_by: str
    intent: QueryIntent
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    tool_calls: List[ToolCallResult] = Field(default_factory=list)
    total_latency_ms: float