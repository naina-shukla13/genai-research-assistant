"""
CLI entry point for the AI Research Assistant.

Provides an interactive loop showing, per the case study's explicit
requirement, the final answer, which agent handled the query, and any
retrieved chunks — not just a bare text response.
"""

from core.coordinator import coordinator
from utils.exceptions import AssistantBaseException, AgentRoutingError


def print_response(response) -> None:
    print("\n" + "─" * 70)
    print(f"🧭 Intent: {response.intent.value}")
    print(f"🤖 Handled by: {response.handled_by}")
    print(f"⏱  Latency: {response.total_latency_ms} ms")

    if response.retrieved_chunks:
        print(f"\n📚 Retrieved Context ({len(response.retrieved_chunks)} chunk(s)):")
        for chunk in response.retrieved_chunks:
            preview = chunk.text[:150].replace("\n", " ")
            print(f"   [{chunk.chunk_id}] score={chunk.similarity_score} — {preview}...")

    if response.tool_calls:
        print(f"\n🔧 Tool Calls:")
        for tc in response.tool_calls:
            status = "✅" if tc.success else "❌"
            print(f"   {status} {tc.tool_name.value} → {tc.output if tc.success else tc.error}")

    print(f"\n💬 Final Answer:\n{response.final_answer}")
    print("─" * 70)


def main() -> None:
    print("=" * 70)
    print("  AI Research Assistant — Multi-Agent CLI")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    while True:
        try:
            query = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            response = coordinator.handle_query(query)
            print_response(response)
        except AgentRoutingError as exc:
            print(f"\n❌ Routing error: {exc}")
        except AssistantBaseException as exc:
            print(f"\n❌ System error: {exc}")
        except Exception as exc:
            print(f"\n❌ Unexpected error: {exc}")


if __name__ == "__main__":
    main()