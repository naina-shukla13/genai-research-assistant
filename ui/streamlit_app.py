"""
Optional Streamlit UI — surfaces the same transparency the CLI does
(final answer, handling agent, retrieved chunks, tool calls) in a
visual form. This satisfies the case study's "(Optional but Recommended)
UI" requirement and its explicit sub-bullets: final answer, which agent
handled the query, retrieved chunks.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from core.coordinator import coordinator
from utils.exceptions import AssistantBaseException

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from core.coordinator import coordinator
from utils.exceptions import AssistantBaseException

# ── Auto-ingest on first load (deployed environments start with an empty
# vector store, since storage/vectors.duckdb is gitignored) ──────────────
from rag.vector_store import vector_store

if vector_store.is_empty():
    with st.spinner("Building knowledge base index (first-time setup)..."):
        from scripts.ingest import run_ingestion
        run_ingestion()

st.set_page_config(page_title="AI Research Assistant", page_icon="🧭", layout="wide")

st.set_page_config(page_title="AI Research Assistant", page_icon="🧭", layout="wide")

st.title("🧭 AI Research Assistant")
st.caption("Multi-agent system: Coordinator → Retriever / General Agent")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question...")

if query:
    try:
        with st.spinner("Routing and processing..."):
            response = coordinator.handle_query(query)
        st.session_state.history.append(response)
    except AssistantBaseException as exc:
        st.error(f"System error: {exc}")

for response in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(response.query)

    with st.chat_message("assistant"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Handled by", response.handled_by)
        col2.metric("Intent", response.intent.value)
        col3.metric("Latency", f"{response.total_latency_ms} ms")

        st.markdown(f"**Answer:** {response.final_answer}")

        if response.retrieved_chunks:
            with st.expander(f"📚 Retrieved Context ({len(response.retrieved_chunks)} chunks)"):
                for chunk in response.retrieved_chunks:
                    st.markdown(
                        f"**`{chunk.chunk_id}`** — similarity: `{chunk.similarity_score}`\n\n"
                        f"> {chunk.text}"
                    )

        if response.tool_calls:
            with st.expander(f"🔧 Tool Calls ({len(response.tool_calls)})"):
                for tc in response.tool_calls:
                    icon = "✅" if tc.success else "❌"
                    st.markdown(f"{icon} **{tc.tool_name.value}** → `{tc.output if tc.success else tc.error}`")

st.sidebar.header("About")
st.sidebar.markdown(
    """
    **Architecture:** Star topology — all routing flows through the Coordinator.

    - **Coordinator**: classifies intent, routes, synthesizes response
    - **Retriever Agent**: RAG-only, grounded strictly in retrieved chunks
    - **General Agent**: reasoning + tool use (calculator, mocked web search)

    Vector store: DuckDB (brute-force cosine similarity)
    Embeddings: sentence-transformers (local, free)
    LLM: OpenRouter (free-tier model)
    """
)