import os
import sqlite3
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from state import SchedulerState
from nodes import (
    document_processor,
    task_ingester,
    scheduler,
    critic,
    human_review,
    tool_node,
    resolve_timezone,
)


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_scheduler(state: SchedulerState):
    """
    After the Scheduler runs, check if the LLM requested tool calls.
    If yes -> route to the ToolNode to execute them.
    If no  -> the schedule is ready for the Critic.
    """
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
    return "critic"


def route_after_critic(state: SchedulerState):
    """
    After the Critic runs, decide the next step:
    - If the critique says PERFECT or we've hit 3 revisions -> human review
    - Otherwise -> loop back to the Scheduler for another attempt
    """
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    if "PERFECT" in critique.upper() or revision_count >= 3:
        return "human_review"
    return "scheduler"


def route_after_human(state: SchedulerState):
    """
    After the human reviews:
    - 'approved' -> END (we're done)
    - anything else -> back to Scheduler to incorporate tweaks
    """
    if state.get("human_feedback", "") == "approved":
        return END
    return "scheduler"


# ---------------------------------------------------------------------------
# Graph construction — reusable builder for both terminal and Streamlit
# ---------------------------------------------------------------------------

DB_DIR = "/app/data" if os.path.isdir("/app/data") else "."
DB_PATH = os.path.join(DB_DIR, "scheduler_memory.db")


def build_graph(sqlite_conn=None):
    """
    Build and compile the scheduling graph.
    Pass a sqlite3.Connection to enable persistence; omit for in-memory only.
    """
    workflow = StateGraph(SchedulerState)

    # Register all nodes
    workflow.add_node("document_processor", document_processor)
    workflow.add_node("task_ingester", task_ingester)
    workflow.add_node("scheduler", scheduler)
    workflow.add_node("tools", tool_node)
    workflow.add_node("critic", critic)
    workflow.add_node("human_review", human_review)

    # Fixed edges
    workflow.add_edge(START, "document_processor")
    workflow.add_edge("document_processor", "task_ingester")
    workflow.add_edge("task_ingester", "scheduler")
    workflow.add_edge("tools", "scheduler")  # Tool results feed back into Scheduler

    # Conditional edges
    workflow.add_conditional_edges("scheduler", route_after_scheduler, {
        "tools": "tools",
        "critic": "critic",
    })
    workflow.add_conditional_edges("critic", route_after_critic, {
        "human_review": "human_review",
        "scheduler": "scheduler",
    })
    workflow.add_conditional_edges("human_review", route_after_human, {
        END: END,
        "scheduler": "scheduler",
    })

    if sqlite_conn:
        memory = SqliteSaver(sqlite_conn)
        return workflow.compile(checkpointer=memory)
    return workflow.compile()


# ---------------------------------------------------------------------------
# Main execution loop with HITL interrupt handling (terminal mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("   AI DAILY SCHEDULE OPTIMIZER")
    print("   Powered by Claude Haiku + LangGraph")
    print("=" * 55)

    sqlite_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    app = build_graph(sqlite_conn)

    # Thread ID is date-based so today's session persists across restarts
    thread_id = f"schedule-{datetime.now().strftime('%Y-%m-%d')}"
    config = {"configurable": {"thread_id": thread_id}}

    # Check if there is an interrupted session to resume
    snapshot = app.get_state(config)

    if snapshot.next:
        # --- Resume a paused session (human_review interrupt) ---
        print(f"\nResuming session: {thread_id}")
        values = snapshot.values
        schedule = values.get("draft_schedule", "")
        revisions = values.get("revision_count", 0)

        print(f"\n{'=' * 50}")
        print("  YOUR PROPOSED SCHEDULE")
        print(f"{'=' * 50}")
        print(schedule)
        print(f"\n  (Revisions used: {revisions} of 3)")

        feedback = input(
            "\nType 'approve' to accept, or describe your tweaks:\n> "
        ).strip()

        result = app.invoke(Command(resume=feedback), config)
    else:
        # --- Fresh scheduling session ---
        now = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        print(f"\n  Today is {now}")

        user_location = input(
            "\nWhat is your current location? (e.g. '123 Main St, Austin, TX')\n> "
        ).strip() or "Home"

        user_input = input(
            "\nWhat do you need to get done today?\n> "
        ).strip()

        if not user_input:
            print("No tasks provided. Exiting.")
            exit()

        print("\nResolving timezone...")
        user_tz = resolve_timezone(user_location)
        print(f"  Timezone: {user_tz}")

        print("\nThinking...\n")
        result = app.invoke(
            {
                "raw_tasks": user_input,
                "user_location": user_location,
                "user_timezone": user_tz,
            },
            config,
        )

    # Handle additional interrupt cycles (user tweaks -> re-draft -> review)
    snapshot = app.get_state(config)
    while snapshot.next:
        values = snapshot.values
        schedule = values.get("draft_schedule", "")
        revisions = values.get("revision_count", 0)

        print(f"\n{'=' * 50}")
        print("  REVISED SCHEDULE")
        print(f"{'=' * 50}")
        print(schedule)
        print(f"\n  (Revisions used: {revisions} of 3)")

        feedback = input(
            "\nType 'approve' to accept, or describe more tweaks:\n> "
        ).strip()

        result = app.invoke(Command(resume=feedback), config)
        snapshot = app.get_state(config)

    # --- Final output: always read from the authoritative state snapshot ---
    final_state = app.get_state(config).values

    print(f"\n{'=' * 55}")
    print("   FINAL OPTIMIZED SCHEDULE")
    print(f"{'=' * 55}")
    print(final_state.get("draft_schedule", "No schedule generated."))

    critique = final_state.get("critique", "")
    if critique:
        print(f"\n{'=' * 55}")
        print("   FINAL ASSESSMENT")
        print(f"{'=' * 55}")
        print(critique)

    print(f"\n  Total revisions: {final_state.get('revision_count', 0)}")
    print(f"  Session thread:  {thread_id}")
    print(f"  Persistence:     scheduler_memory.db")
