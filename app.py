"""
Streamlit frontend for the AI Weekly Schedule Optimizer.
Run with:  streamlit run app.py
"""

import html as html_lib
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from langgraph.types import Command

from graph import build_graph, DB_PATH
from nodes import process_uploaded_file, resolve_timezone

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Weekly Schedule Optimizer",
    page_icon="\U0001f4c5",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS — injected once for card styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Schedule card */
.sched-card {
    border-radius: 0.6rem;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(250,250,250,0.08);
}
.sched-card .card-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.sched-card .card-title {
    font-size: 1.15rem;
    font-weight: 600;
}
.sched-card .card-time {
    color: rgba(250,250,250,0.45);
    font-size: 0.92rem;
    font-weight: 500;
}
.sched-card .card-meta {
    display: flex;
    gap: 1.2rem;
    color: rgba(250,250,250,0.45);
    font-size: 0.85rem;
    margin-top: 0.35rem;
    flex-wrap: wrap;
}
.sched-card .card-notes {
    color: rgba(250,250,250,0.55);
    font-size: 0.9rem;
    margin-top: 0.35rem;
}
.sched-card .card-badges {
    display: flex;
    gap: 0.45rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
}
.pri-badge {
    padding: 0.12rem 0.55rem;
    border-radius: 1rem;
    font-size: 0.78rem;
    font-weight: 600;
}
.tool-badge {
    padding: 0.12rem 0.5rem;
    border-radius: 0.45rem;
    font-size: 0.78rem;
}
/* HITL review banner */
.hitl-banner {
    background: linear-gradient(135deg, rgba(255,152,0,0.12), rgba(255,87,34,0.06));
    border: 1px solid rgba(255,152,0,0.3);
    border-radius: 0.75rem;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0 1.2rem 0;
}
.hitl-banner h3 {
    margin: 0 0 0.3rem 0;
    color: #ffa726;
    font-size: 1.15rem;
}
.hitl-banner p {
    color: rgba(250,250,250,0.6);
    margin: 0;
    font-size: 0.92rem;
}
/* Day header in tabs */
.day-header {
    font-size: 0.85rem;
    color: rgba(250,250,250,0.4);
    margin-bottom: 0.5rem;
}
/* Weekly overview cards */
.week-overview {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.day-pill {
    padding: 0.4rem 0.8rem;
    border-radius: 0.5rem;
    font-size: 0.82rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(250,250,250,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached graph instance
# ---------------------------------------------------------------------------

@st.cache_resource
def get_app():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return build_graph(conn)

graph_app = get_app()

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

NODE_LABELS = {
    "document_processor": "Processing uploaded documents...",
    "task_ingester": "Analyzing your tasks for the week...",
    "scheduler": "Drafting your weekly schedule...",
    "tools": "Fetching weather forecast & commute data...",
    "critic": "Reviewing schedule quality...",
    "human_review": "Ready for your review!",
}

TYPE_CONFIG = {
    "work":    {"icon": "\U0001f4bb", "label": "Focus Work"},
    "break":   {"icon": "\u2615",     "label": "Break"},
    "travel":  {"icon": "\U0001f697", "label": "Travel"},
    "fitness": {"icon": "\U0001f4aa", "label": "Fitness"},
    "call":    {"icon": "\U0001f4de", "label": "Call"},
    "errand":  {"icon": "\U0001f6d2", "label": "Errand"},
    "meal":    {"icon": "\U0001f37d\ufe0f", "label": "Meal"},
    "shower":  {"icon": "\U0001f6bf", "label": "Freshen Up"},
}

DEFAULT_TYPE = {"icon": "\U0001f4cb", "label": "Task"}


def _priority_style(p):
    """Return color/bg/label for a priority value."""
    if p is None:
        return {"color": "#78909c", "bg": "rgba(120,144,156,0.15)", "label": "\u2014"}
    if p <= 3:
        return {"color": "#ff6b6b", "bg": "rgba(255,75,75,0.15)", "label": f"P{p}"}
    if p <= 6:
        return {"color": "#ffa726", "bg": "rgba(255,167,38,0.15)", "label": f"P{p}"}
    return {"color": "#66bb6a", "bg": "rgba(102,187,106,0.15)", "label": f"P{p}"}


def _border_color(p):
    if p is None:
        return "#78909c"
    if p <= 3:
        return "#ff4b4b"
    if p <= 6:
        return "#ffa726"
    return "#66bb6a"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_card(entry: dict):
    """Render a single schedule entry as a styled HTML card."""
    t = TYPE_CONFIG.get(entry.get("type", ""), DEFAULT_TYPE)
    p = entry.get("priority")
    ps = _priority_style(p)
    bc = _border_color(p)
    title = html_lib.escape(entry.get("title", "Untitled"))
    start = html_lib.escape(str(entry.get("start", "?")))
    end = html_lib.escape(str(entry.get("end", "?")))
    dur = entry.get("duration_min", "?")
    loc = entry.get("location")
    notes = entry.get("notes")
    weather = entry.get("weather")
    commute = entry.get("commute")

    # Build metadata spans
    meta_parts = [f"\u23f1 {dur} min"]
    if loc:
        meta_parts.append(f"\U0001f4cd {html_lib.escape(loc)}")
    meta_html = "".join(f"<span>{m}</span>" for m in meta_parts)

    # Build badge spans
    badges = []
    if weather:
        badges.append(
            f'<span class="tool-badge" style="background:rgba(30,136,229,0.12);'
            f'color:#64b5f6;">\U0001f324\ufe0f {html_lib.escape(weather)}</span>'
        )
    if commute:
        badges.append(
            f'<span class="tool-badge" style="background:rgba(255,167,38,0.12);'
            f'color:#ffb74d;">\U0001f6a6 {html_lib.escape(commute)}</span>'
        )
    badges_html = "".join(badges)

    # Notes
    notes_html = ""
    if notes:
        notes_html = f'<div class="card-notes">{html_lib.escape(notes)}</div>'

    # Priority badge
    pri_html = (
        f'<span class="pri-badge" style="background:{ps["bg"]};'
        f'color:{ps["color"]};">{ps["label"]}</span>'
    )

    card = f"""
    <div class="sched-card" style="border-left:4px solid {bc};">
        <div class="card-row">
            <span class="card-title">{t["icon"]} {title}</span>
            <div style="display:flex;gap:0.6rem;align-items:center;">
                {pri_html}
                <span class="card-time">{start} \u2013 {end}</span>
            </div>
        </div>
        <div class="card-meta">{meta_html}</div>
        {notes_html}
        <div class="card-badges">{badges_html}</div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)


def render_weekly_metrics(schedule_data: dict):
    """Render top-level weekly overview metrics from the structured schedule dict."""
    if not schedule_data:
        return

    total_min = 0
    total_tasks = 0
    total_focus = 0
    days_with_tasks = 0
    weather_note = None

    for day_name, entries in schedule_data.items():
        if not isinstance(entries, list):
            continue
        real_tasks = [e for e in entries if e.get("type") not in ("break", "travel", "shower")]
        if real_tasks:
            days_with_tasks += 1
        total_tasks += len(entries)
        total_focus += len(real_tasks)
        total_min += sum(e.get("duration_min", 0) for e in entries)
        if not weather_note:
            weather_note = next(
                (e["weather"] for e in entries if e.get("weather")), None
            )

    hours, mins = divmod(total_min, 60)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Planned", f"{hours}h {mins}m", delta=f"{total_tasks} blocks")
    c2.metric("Focus Tasks", str(total_focus), delta=f"across {days_with_tasks} days")
    c3.metric("Days Planned", f"{days_with_tasks}/7")
    if weather_note:
        c4.metric("Weather", weather_note[:30])
    else:
        c4.metric("Weather", "N/A")


def render_day_metrics(entries: list):
    """Render compact metrics for a single day."""
    if not entries:
        st.caption("No tasks scheduled for this day.")
        return

    total_min = sum(e.get("duration_min", 0) for e in entries)
    hours, mins = divmod(total_min, 60)
    focus = [e for e in entries if e.get("type") not in ("break", "travel", "shower")]
    end_time = entries[-1].get("end", "?") if entries else "?"

    c1, c2, c3 = st.columns(3)
    c1.metric("Day Total", f"{hours}h {mins}m")
    c2.metric("Tasks", str(len(focus)))
    c3.metric("Ends At", end_time)


def render_day_schedule(entries: list, fallback_text: str = ""):
    """Render a single day's schedule with cards."""
    if entries:
        render_day_metrics(entries)
        st.markdown("---")
        for entry in entries:
            render_card(entry)
    elif fallback_text:
        st.markdown(fallback_text)
    else:
        st.info("No tasks scheduled for this day.")


def render_weekly_schedule(schedule_data: dict, fallback_text: str):
    """Render the full weekly schedule with tabs per day."""
    if not schedule_data:
        st.markdown(fallback_text)
        return

    render_weekly_metrics(schedule_data)
    st.markdown("---")

    # Create tabs for each day
    day_names = list(schedule_data.keys())
    if not day_names:
        st.markdown(fallback_text)
        return

    tabs = st.tabs(day_names)
    for tab, day_name in zip(tabs, day_names):
        with tab:
            entries = schedule_data.get(day_name, [])
            if isinstance(entries, list):
                render_day_schedule(entries)
            else:
                st.markdown(str(entries))


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "phase" not in st.session_state:
    st.session_state.phase = "input"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
if "location" not in st.session_state:
    st.session_state.location = "Home"
if "user_timezone" not in st.session_state:
    st.session_state.user_timezone = "UTC"
if "timeline" not in st.session_state:
    st.session_state.timeline = []


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def new_session():
    st.session_state.phase = "input"
    st.session_state.thread_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    st.session_state.user_timezone = "UTC"
    st.session_state.timeline = []


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

st.title("AI Weekly Schedule Optimizer")
try:
    _tz = ZoneInfo(st.session_state.user_timezone)
except Exception:
    _tz = ZoneInfo("UTC")
_now_dt = datetime.now(tz=_tz)
_now_str = _now_dt.strftime("%A, %B %d, %Y at %H:%M")
_week_end = (_now_dt + timedelta(days=6)).strftime("%A, %B %d")
st.caption(f"Planning: {_now_str} through {_week_end} ({st.session_state.user_timezone})")


# ---------------------------------------------------------------------------
# Helper: stream graph execution with live status
# ---------------------------------------------------------------------------

def run_graph_streaming(inputs, config, *, is_resume=False):
    """Stream graph execution and show node-level progress."""
    with st.status("Working on your weekly schedule...", expanded=True) as status:
        try:
            for chunk in graph_app.stream(inputs, config, stream_mode="updates"):
                for node_name in chunk:
                    label = NODE_LABELS.get(node_name, f"Running: {node_name}")
                    st.write(f"- {label}")
            status.update(label="Done!", state="complete")
            return True
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
            st.error(f"An error occurred: {e}")
            return False


# ===========================================================================
# Phase: INPUT
# ===========================================================================

if st.session_state.phase == "input":
    st.subheader("What do you need to get done this week?")

    raw_tasks = st.text_area(
        "Type your tasks (natural language is fine!)",
        placeholder=(
            "e.g., Finish project report by Thursday, gym Mon/Wed/Fri at 6pm, "
            "dentist appointment Tuesday 10am, groceries, call mom, "
            "study for exam on Saturday"
        ),
        height=140,
    )

    uploaded_files = st.file_uploader(
        "Or upload task documents (handwritten lists, PDFs, screenshots)",
        accept_multiple_files=True,
        type=["txt", "pdf", "png", "jpg", "jpeg"],
    )

    can_submit = bool(raw_tasks and raw_tasks.strip()) or bool(uploaded_files)

    if st.button("Plan My Week", type="primary", disabled=not can_submit):
        # Step 1: process uploaded files
        extracted_text = ""
        if uploaded_files:
            with st.status("Processing uploaded files...", expanded=True) as fs:
                for f in uploaded_files:
                    st.write(f"Reading: **{f.name}**")
                    try:
                        text = process_uploaded_file(f.name, f.read())
                        extracted_text += f"\n--- From {f.name} ---\n{text}\n"
                        st.write(f"  {f.name} processed")
                    except Exception as e:
                        st.write(f"  {f.name} failed: {e}")
                fs.update(label="Files processed!", state="complete")

        # Step 2: resolve timezone
        with st.status("Detecting timezone...") as tz_s:
            user_tz = resolve_timezone(st.session_state.location)
            st.session_state.user_timezone = user_tz
            tz_s.update(label=f"Timezone: {user_tz}", state="complete")

        # Step 3: invoke the graph
        initial_state = {
            "raw_tasks": raw_tasks or "See uploaded documents below.",
            "user_location": st.session_state.location,
            "user_timezone": user_tz,
            "uploaded_files_text": extracted_text,
        }

        config = get_config()
        success = run_graph_streaming(initial_state, config)

        if success:
            snapshot = graph_app.get_state(config)
            st.session_state.phase = "review" if snapshot.next else "done"
            st.rerun()


# ===========================================================================
# Phase: REVIEW — premium HITL interface with weekly tabbed view
# ===========================================================================

elif st.session_state.phase == "review":
    config = get_config()
    try:
        snapshot = graph_app.get_state(config)
    except Exception:
        st.error("Session expired. Please start a new session.")
        new_session()
        st.rerun()

    values = snapshot.values
    draft_text = values.get("draft_schedule", "No schedule available.")
    structured = values.get("structured_schedule", {})
    critique = values.get("critique", "")
    revisions = values.get("revision_count", 0)

    # Track timeline entries
    timeline_key = f"{revisions}:{draft_text[:80]}"
    if (not st.session_state.timeline
            or st.session_state.timeline[-1].get("_key") != timeline_key):
        st.session_state.timeline.append({
            "type": "schedule",
            "_key": timeline_key,
            "text": draft_text,
            "structured": structured,
            "critique": critique,
            "revision": revisions,
        })

    # --- Render previous timeline entries (collapsed) ---
    has_history = any(
        e["type"] == "schedule"
        for e in st.session_state.timeline[:-1]
    )
    if has_history:
        with st.expander("Previous Revisions", expanded=False):
            for entry in st.session_state.timeline[:-1]:
                if entry["type"] == "tweak":
                    st.markdown(
                        f'> **Your tweak:** *"{entry["content"]}"*'
                    )
                elif entry["type"] == "schedule":
                    st.caption(f"Schedule v{entry['revision']}")
                    st.text(entry["text"][:400] + "..." if len(entry["text"]) > 400 else entry["text"])
                    st.divider()

    # --- Render latest schedule with premium weekly tabs ---
    st.subheader(f"Weekly Schedule v{revisions}")
    render_weekly_schedule(structured, draft_text)

    if critique:
        with st.expander("Critic's Assessment", expanded=False):
            st.markdown(critique)

    # --- HITL review banner ---
    st.markdown("""
    <div class="hitl-banner">
        <h3>\u23f8\ufe0f Review Required</h3>
        <p>The AI has drafted your weekly schedule and is waiting for your approval.
           Approve it or describe what you'd like changed (e.g. "move gym to Thursday",
           "add lunch break on Wednesday").</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"Auto-revisions used: {revisions} of 7 (your tweaks are always allowed)")

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
            "Describe your tweaks:",
            placeholder="e.g. Move dentist to Wednesday, add gym on Thursday, free up Friday evening",
        )
        if st.button("Submit Tweaks", use_container_width=True) and tweaks:
            st.session_state.timeline.append({
                "type": "tweak",
                "content": tweaks,
            })
            success = run_graph_streaming(
                Command(resume=tweaks), config, is_resume=True
            )
            if success:
                snap = graph_app.get_state(config)
                st.session_state.phase = "review" if snap.next else "done"
                st.rerun()


# ===========================================================================
# Phase: DONE — final weekly dashboard
# ===========================================================================

elif st.session_state.phase == "done":
    config = get_config()
    final = graph_app.get_state(config).values

    draft_text = final.get("draft_schedule", "No schedule generated.")
    structured = final.get("structured_schedule", {})
    critique = final.get("critique", "")
    rev_count = final.get("revision_count", 0)

    st.subheader("Final Optimized Weekly Schedule")
    render_weekly_schedule(structured, draft_text)

    if critique:
        with st.expander("Final Assessment", expanded=True):
            st.markdown(critique)

    st.success(f"Weekly schedule complete! Total revisions: {rev_count}")

    # Revision history
    if len(st.session_state.timeline) > 1:
        st.divider()
        with st.expander("Revision History", expanded=False):
            for entry in st.session_state.timeline:
                if entry["type"] == "tweak":
                    st.markdown(f'> **Your tweak:** *"{entry["content"]}"*')
                elif entry["type"] == "schedule":
                    st.caption(f"Schedule v{entry['revision']}")
                    old = entry.get("structured", {})
                    if old:
                        for day_name, day_entries in old.items():
                            if isinstance(day_entries, list) and day_entries:
                                st.text(f"  {day_name}:")
                                for e in day_entries[:3]:
                                    tc = TYPE_CONFIG.get(e.get("type", ""), DEFAULT_TYPE)
                                    st.text(
                                        f"    {e.get('start','?')}-{e.get('end','?')}  "
                                        f"{tc['icon']} {e.get('title','?')}"
                                    )
                                if len(day_entries) > 3:
                                    st.text(f"    ... +{len(day_entries) - 3} more")
                    else:
                        st.text(entry["text"][:300])
                    st.divider()

    st.divider()
    if st.button("Plan Another Week", type="primary"):
        new_session()
        st.rerun()
