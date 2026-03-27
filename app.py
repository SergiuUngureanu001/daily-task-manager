"""
Streamlit frontend for the AI Daily Schedule Optimizer.
Run with:  streamlit run app.py
"""

import sqlite3
from datetime import datetime

import streamlit as st
from langgraph.types import Command

from graph import build_graph, DB_PATH
from nodes import process_uploaded_file

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Daily Schedule Optimizer",
    page_icon="\U0001f4c5",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached graph instance — shared across all Streamlit reruns
# ---------------------------------------------------------------------------

@st.cache_resource
def get_app():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return build_graph(conn)

graph_app = get_app()

# ---------------------------------------------------------------------------
# Friendly labels shown during streaming
# ---------------------------------------------------------------------------

NODE_LABELS = {
    "document_processor": "Processing uploaded documents...",
    "task_ingester": "Analyzing your tasks...",
    "scheduler": "Drafting your schedule...",
    "tools": "Fetching weather & commute data...",
    "critic": "Reviewing schedule quality...",
    "human_review": "Ready for your review!",
}

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "phase" not in st.session_state:
    st.session_state.phase = "input"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
if "location" not in st.session_state:
    st.session_state.location = "Home"


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def new_session():
    st.session_state.phase = "input"
    st.session_state.thread_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    st.session_state.location = st.text_input(
        "Your current location",
        value=st.session_state.location,
        placeholder="e.g. Piata Unirii, Bucuresti",
    )

    st.divider()
    if st.button("New Session", use_container_width=True):
        new_session()
        st.rerun()

    st.divider()
    st.caption("Powered by Claude Haiku + LangGraph")
    st.caption(f"Session: `{st.session_state.thread_id}`")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("AI Daily Schedule Optimizer")
st.caption(f"Today is {datetime.now().strftime('%A, %B %d, %Y at %H:%M')}")


# ---------------------------------------------------------------------------
# Helper: stream graph execution with live status updates
# ---------------------------------------------------------------------------

def run_graph_streaming(inputs, config, *, is_resume=False):
    """
    Stream graph execution and display node-level progress.
    Returns True on success, False on error.
    """
    with st.status("Working on your schedule...", expanded=True) as status:
        try:
            stream = graph_app.stream(inputs, config, stream_mode="updates")
            for chunk in stream:
                for node_name in chunk:
                    label = NODE_LABELS.get(node_name, f"Running: {node_name}")
                    st.write(f"- {label}")
            status.update(label="Done!", state="complete")
            return True
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
            st.error(f"An error occurred: {e}")
            return False


# ---------------------------------------------------------------------------
# Phase: INPUT — collect tasks, location, and optional file uploads
# ---------------------------------------------------------------------------

if st.session_state.phase == "input":
    st.subheader("What do you need to get done today?")

    raw_tasks = st.text_area(
        "Type your tasks (natural language is fine!)",
        placeholder=(
            "e.g., Finish homework, go to the gym at 5pm, "
            "call mom, pick up groceries from Mega Image"
        ),
        height=120,
    )

    uploaded_files = st.file_uploader(
        "Or upload task documents (handwritten lists, PDFs, screenshots)",
        accept_multiple_files=True,
        type=["txt", "pdf", "png", "jpg", "jpeg"],
    )

    can_submit = bool(raw_tasks and raw_tasks.strip()) or bool(uploaded_files)

    if st.button("Generate My Schedule", type="primary", disabled=not can_submit):
        # ---- Step 1: process uploaded files (outside the graph) ----
        extracted_text = ""
        if uploaded_files:
            with st.status("Processing uploaded files...", expanded=True) as fstatus:
                for f in uploaded_files:
                    st.write(f"Reading: **{f.name}**")
                    try:
                        text = process_uploaded_file(f.name, f.read())
                        extracted_text += f"\n--- From {f.name} ---\n{text}\n"
                        st.write(f"  {f.name} processed")
                    except Exception as e:
                        st.write(f"  {f.name} failed: {e}")
                fstatus.update(label="Files processed!", state="complete")

        # ---- Step 2: invoke the graph ----
        initial_state = {
            "raw_tasks": raw_tasks or "See uploaded documents below.",
            "user_location": st.session_state.location,
            "uploaded_files_text": extracted_text,
        }

        config = get_config()
        success = run_graph_streaming(initial_state, config)

        if success:
            snapshot = graph_app.get_state(config)
            st.session_state.phase = "review" if snapshot.next else "done"
            st.rerun()


# ---------------------------------------------------------------------------
# Phase: REVIEW — human-in-the-loop approval or tweaks
# ---------------------------------------------------------------------------

elif st.session_state.phase == "review":
    config = get_config()
    try:
        snapshot = graph_app.get_state(config)
    except Exception:
        st.error("Session expired. Please start a new session.")
        new_session()
        st.rerun()

    values = snapshot.values
    schedule = values.get("draft_schedule", "No schedule available.")
    critique = values.get("critique", "")
    revisions = values.get("revision_count", 0)

    st.subheader("Proposed Schedule")
    st.markdown(schedule)

    if critique:
        with st.expander("Critic's Assessment", expanded=False):
            st.markdown(critique)

    st.info(f"Revisions used: {revisions} of 3")
    st.divider()

    st.subheader("Your Decision")
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Approve Schedule", type="primary", use_container_width=True):
            success = run_graph_streaming(
                Command(resume="approved"), config, is_resume=True
            )
            if success:
                snap = graph_app.get_state(config)
                st.session_state.phase = "review" if snap.next else "done"
                st.rerun()

    with col2:
        tweaks = st.text_input(
            "Or describe your tweaks:",
            placeholder="e.g. Move gym to morning, add lunch break",
        )
        if st.button("Submit Tweaks", use_container_width=True) and tweaks:
            success = run_graph_streaming(
                Command(resume=tweaks), config, is_resume=True
            )
            if success:
                snap = graph_app.get_state(config)
                st.session_state.phase = "review" if snap.next else "done"
                st.rerun()


# ---------------------------------------------------------------------------
# Phase: DONE — display final schedule
# ---------------------------------------------------------------------------

elif st.session_state.phase == "done":
    config = get_config()
    final_state = graph_app.get_state(config).values

    st.subheader("Final Optimized Schedule")
    st.markdown(final_state.get("draft_schedule", "No schedule generated."))

    critique = final_state.get("critique", "")
    if critique:
        with st.expander("Final Assessment", expanded=True):
            st.markdown(critique)

    st.success(
        f"Schedule complete! Total revisions: "
        f"{final_state.get('revision_count', 0)}"
    )

    st.divider()
    if st.button("Plan Another Day", type="primary"):
        new_session()
        st.rerun()
