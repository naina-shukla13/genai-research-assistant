"""
Retriever Agent — RAG-only. Its sole job is to retrieve relevant chunks
and synthesize an answer grounded strictly in that retrieved context.

Critical design point: this agent NEVER answers from general knowledge.
If retrieval returns no chunks above threshold, it says so explicitly
rather than falling back to the LLM's own knowledge — that fallback
belongs to the Coordinator's routing logic, not to this agent silently
blurring its responsibility boundary.
"""

from agents.base_agent import BaseAgent
from config.settings import settings
from core.schemas import AgentResponse, Chunk
from llm.openrouter_client import openrouter_client
from rag.retriever import retrieve_top_k
from utils.exceptions import RetrievalError
from utils.logger import trace_step, log_event


RETRIEVER_SYSTEM_PROMPT = """You are a Retrieval-Augmented research assistant.

You will be given a user query and a set of retrieved context chunks from
a document knowledge base. Answer the query using ONLY the information in
the provided chunks. Do not use outside knowledge.

If the chunks do not contain enough information to answer the query,
say so explicitly rather than guessing.

Cite which chunk(s) you used by their chunk_id where relevant."""


class RetrieverAgent(BaseAgent):
    agent_name = "RetrieverAgent"
    model_name = settings.retriever_agent_model

    def run(self, query: str) -> AgentResponse:
        try:
            chunks = retrieve_top_k(query)
        except RetrievalError as exc:
            log_event("RetrieverAgent", "retrieval_unavailable", query=query, error=str(exc))
            chunks = []

        if not chunks:
            return AgentResponse(
                agent_name=self.agent_name,
                answer=(
                    "I couldn't find relevant information in the knowledge base "
                    "for this query."
                ),
                retrieved_chunks=[],
                reasoning_trace="No chunks cleared the similarity threshold.",
            )

        context_block = "\n\n".join(
            f"[{c.chunk_id}] (score={c.similarity_score}): {c.text}" for c in chunks
        )
        user_prompt = f"Query: {query}\n\nRetrieved context:\n{context_block}"

        with trace_step("RetrieverAgent", "synthesize_answer", query=query, chunk_count=len(chunks)):
            answer = self.llm.chat(
                model=self.model_name,
                system_prompt=RETRIEVER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

        log_event(
            "RetrieverAgent", "answer_synthesized",
            query=query, chunks_used=[c.chunk_id for c in chunks],
        )

        return AgentResponse(
            agent_name=self.agent_name,
            answer=answer,
            retrieved_chunks=chunks,
            reasoning_trace=f"Synthesized from {len(chunks)} retrieved chunk(s).",
        )