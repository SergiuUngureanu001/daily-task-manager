"""
Streamlit frontend for the AI Weekly Schedule Optimizer.
Run with:  streamlit run app.py
"""

import io
import json
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from langgraph.types import Command
from streamlit_autorefresh import st_autorefresh

from graph import build_graph, build_reschedule_graph, DB_PATH
from nodes import process_uploaded_file, resolve_timezone, _schedule_to_text
import session_store

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Weekly Schedule Optimizer",
    page_icon="\U0001f4c5",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
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
/* Tighter spacing for task metadata under checkboxes */
div[data-testid="stCaptionContainer"] {
    margin-top: -0.6rem;
    padding-left: 1.75rem;
}
/* Session history sidebar items */
.session-item {
    padding: 0.4rem 0.6rem;
    border-radius: 0.4rem;
    margin-bottom: 0.3rem;
    border-left: 3px solid rgba(250,250,250,0.15);
    font-size: 0.85rem;
}
.session-item.complete {
    border-left-color: #66bb6a;
}
.session-item .session-date {
    color: rgba(250,250,250,0.4);
    font-size: 0.75rem;
}
/* Focus mode big timer */
.focus-timer {
    font-size: 3.5rem;
    font-weight: 700;
    text-align: center;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.1em;
}
.focus-task-title {
    font-size: 1.6rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 0.5rem;
}
/* Goal progress cards */
.goal-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0.6rem;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached graph instances
# ---------------------------------------------------------------------------

@st.cache_resource
def get_app():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return build_graph(conn)


@st.cache_resource
def get_reschedule_app():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return build_reschedule_graph(conn)


graph_app = get_app()
reschedule_app = get_reschedule_app()

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


# ---------------------------------------------------------------------------
# Checkbox key helpers
# ---------------------------------------------------------------------------

def _task_key(day: str, entry: dict, index: int) -> str:
    """Generate a unique, stable session_state key for a task checkbox."""
    title = entry.get("title", "task").replace(" ", "_")[:30]
    start = entry.get("start", "00")
    return f"chk_{day}_{index}_{start}_{title}"


def _init_checkbox_states(schedule_data: dict):
    """Ensure every task checkbox has a session_state entry (default False)."""
    for day_name, entries in schedule_data.items():
        if not isinstance(entries, list):
            continue
        for i, entry in enumerate(entries):
            key = _task_key(day_name, entry, i)
            if key not in st.session_state:
                st.session_state[key] = False


def _clear_checkbox_states():
    """Remove all checkbox keys from session_state (used on new session)."""
    to_remove = [k for k in st.session_state if k.startswith("chk_")]
    for k in to_remove:
        del st.session_state[k]


# ---------------------------------------------------------------------------
# Priority formatting
# ---------------------------------------------------------------------------

def _priority_label(p):
    if p is None:
        return ""
    if p <= 3:
        return f"\U0001f534 P{p}"
    if p <= 6:
        return f"\U0001f7e0 P{p}"
    return f"\U0001f7e2 P{p}"


# ---------------------------------------------------------------------------
# Timezone helper
# ---------------------------------------------------------------------------

def _get_user_tz():
    tz_name = st.session_state.get("user_timezone", "UTC")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("UTC")


def _get_today_name():
    return datetime.now(tz=_get_user_tz()).strftime("%A")


def _get_tomorrow_name():
    return (datetime.now(tz=_get_user_tz()) + timedelta(days=1)).strftime("%A")


# ---------------------------------------------------------------------------
# Rendering helpers — checkbox-based
# ---------------------------------------------------------------------------

def render_task_checkbox(entry: dict, day_name: str, index: int):
    """Render a single task as an interactive checkbox with clean metadata."""
    t = TYPE_CONFIG.get(entry.get("type", ""), DEFAULT_TYPE)
    title = entry.get("title", "Untitled")
    start = entry.get("start", "?")
    end = entry.get("end", "?")

    label = f"{t['icon']} {start} \u2013 {end} | {title}"

    key = _task_key(day_name, entry, index)
    checked = st.checkbox(label, value=st.session_state.get(key, False), key=key)

    meta_parts = []
    p = entry.get("priority")
    if p is not None:
        meta_parts.append(_priority_label(p))
    dur = entry.get("duration_min")
    if dur:
        meta_parts.append(f"\u23f1 {dur} min")
    loc = entry.get("location")
    if loc:
        meta_parts.append(f"\U0001f4cd {loc}")
    weather = entry.get("weather")
    if weather:
        meta_parts.append(f"\U0001f324\ufe0f {weather}")
    commute = entry.get("commute")
    if commute:
        meta_parts.append(f"\U0001f6a6 {commute}")
    notes = entry.get("notes")
    if notes:
        meta_parts.append(f"\U0001f4dd {notes}")
    goal_id = entry.get("goal_id")
    if goal_id is not None:
        goals = st.session_state.get("macro_goals", [])
        if 0 <= goal_id < len(goals):
            meta_parts.append(f"\U0001f3af {goals[goal_id]}")

    if meta_parts:
        meta_text = " | ".join(meta_parts)
        if checked:
            st.caption(f"~~{meta_text}~~")
        else:
            st.caption(meta_text)


def render_day_progress(entries: list, day_name: str):
    """Show a progress bar for how many tasks are checked off in a day."""
    if not entries:
        return
    total = len(entries)
    done = sum(
        1 for i, e in enumerate(entries)
        if st.session_state.get(_task_key(day_name, e, i), False)
    )
    st.progress(done / total if total else 0, text=f"{done}/{total} completed")


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


def render_weekly_metrics(schedule_data: dict):
    """Render top-level weekly overview metrics."""
    if not schedule_data:
        return

    total_min = 0
    total_tasks = 0
    total_focus = 0
    total_checked = 0
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
        total_checked += sum(
            1 for i, e in enumerate(entries)
            if st.session_state.get(_task_key(day_name, e, i), False)
        )
        if not weather_note:
            weather_note = next(
                (e["weather"] for e in entries if e.get("weather")), None
            )

    hours, mins = divmod(total_min, 60)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Planned", f"{hours}h {mins}m", delta=f"{total_tasks} blocks")
    c2.metric("Focus Tasks", str(total_focus), delta=f"across {days_with_tasks} days")
    c3.metric("Completed", f"{total_checked}/{total_tasks}")
    if weather_note:
        c4.metric("Weather", weather_note[:30])
    else:
        c4.metric("Weather", "N/A")


def render_day_schedule(entries: list, day_name: str, fallback_text: str = ""):
    """Render a single day's schedule as an interactive checklist."""
    if entries:
        render_day_metrics(entries)
        render_day_progress(entries, day_name)
        st.markdown("---")
        for i, entry in enumerate(entries):
            render_task_checkbox(entry, day_name, i)
    elif fallback_text:
        st.markdown(fallback_text)
    else:
        st.info("No tasks scheduled for this day.")


def render_weekly_schedule(schedule_data: dict, fallback_text: str):
    """Render the full weekly schedule with tabs per day and interactive checkboxes."""
    if not schedule_data:
        st.markdown(fallback_text)
        return

    _init_checkbox_states(schedule_data)

    render_weekly_metrics(schedule_data)
    st.markdown("---")

    day_names = list(schedule_data.keys())
    if not day_names:
        st.markdown(fallback_text)
        return

    tab_labels = []
    for day_name in day_names:
        entries = schedule_data.get(day_name, [])
        if isinstance(entries, list) and entries:
            done = sum(
                1 for i, e in enumerate(entries)
                if st.session_state.get(_task_key(day_name, e, i), False)
            )
            total = len(entries)
            if done == total and total > 0:
                tab_labels.append(f"\u2705 {day_name}")
            elif done > 0:
                tab_labels.append(f"{day_name} ({done}/{total})")
            else:
                tab_labels.append(day_name)
        else:
            tab_labels.append(day_name)

    tabs = st.tabs(tab_labels)
    for tab, day_name in zip(tabs, day_names):
        with tab:
            entries = schedule_data.get(day_name, [])
            if isinstance(entries, list):
                render_day_schedule(entries, day_name)
            else:
                st.markdown(str(entries))


# ---------------------------------------------------------------------------
# Focus Mode helpers
# ---------------------------------------------------------------------------

def _get_current_task(schedule_data: dict):
    """
    Find the task scheduled for the current time.
    Returns (entry, index, today_entries, today_name) or (None, None, entries, name).
    """
    if not schedule_data:
        return None, None, [], ""

    tz = _get_user_tz()
    now = datetime.now(tz=tz)
    today_name = now.strftime("%A")
    now_minutes = now.hour * 60 + now.minute

    today_entries = schedule_data.get(today_name, [])
    if not isinstance(today_entries, list):
        return None, None, [], today_name

    for i, entry in enumerate(today_entries):
        try:
            sh, sm = map(int, entry.get("start", "0:0").split(":"))
            eh, em = map(int, entry.get("end", "0:0").split(":"))
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            if start_min <= now_minutes < end_min:
                return entry, i, today_entries, today_name
        except (ValueError, TypeError):
            continue

    return None, None, today_entries, today_name


def _get_next_task(today_entries: list, current_index: int | None):
    """Find the next upcoming task after current_index."""
    if current_index is None:
        # Find the first future task
        tz = _get_user_tz()
        now_minutes = datetime.now(tz=tz).hour * 60 + datetime.now(tz=tz).minute
        for i, entry in enumerate(today_entries):
            try:
                sh, sm = map(int, entry.get("start", "0:0").split(":"))
                if sh * 60 + sm > now_minutes:
                    return entry, i
            except (ValueError, TypeError):
                continue
        return None, None

    next_idx = current_index + 1
    if next_idx < len(today_entries):
        return today_entries[next_idx], next_idx
    return None, None


def render_focus_mode(schedule_data: dict):
    """Render the single-task focus view with live countdown timer."""
    # Auto-refresh every second while in focus mode
    st_autorefresh(interval=1000, key="focus_timer_refresh")

    entry, idx, today_entries, today_name = _get_current_task(schedule_data)

    if entry is None:
        next_entry, _ = _get_next_task(today_entries, None)
        st.markdown("### No active task right now")
        if next_entry:
            t = TYPE_CONFIG.get(next_entry.get("type", ""), DEFAULT_TYPE)
            st.info(
                f"**Next up:** {t['icon']} {next_entry.get('title', 'Untitled')} "
                f"at {next_entry.get('start', '?')}"
            )
        else:
            st.info("No more tasks scheduled for today. Great job!")
        return

    # Current task info
    t = TYPE_CONFIG.get(entry.get("type", ""), DEFAULT_TYPE)
    title = entry.get("title", "Untitled")
    start_str = entry.get("start", "00:00")
    end_str = entry.get("end", "00:00")
    priority = entry.get("priority")

    st.markdown(
        f'<div class="focus-task-title">{t["icon"]} {title}</div>',
        unsafe_allow_html=True,
    )

    # Calculate time remaining
    tz = _get_user_tz()
    now = datetime.now(tz=tz)
    try:
        eh, em = map(int, end_str.split(":"))
        sh, sm = map(int, start_str.split(":"))
        end_dt = now.replace(hour=eh, minute=em, second=0, microsecond=0)
        start_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        remaining = end_dt - now
        total_duration = end_dt - start_dt
        elapsed = now - start_dt

        remaining_secs = max(0, int(remaining.total_seconds()))
        total_secs = max(1, int(total_duration.total_seconds()))
        elapsed_secs = max(0, int(elapsed.total_seconds()))

        # Progress: how much of the task is done (0.0 to 1.0)
        progress = min(1.0, elapsed_secs / total_secs)
    except (ValueError, TypeError):
        remaining_secs = 0
        progress = 0.0

    # Timer display
    mins_left, secs_left = divmod(remaining_secs, 60)
    hours_left, mins_left = divmod(mins_left, 60)

    if hours_left > 0:
        timer_str = f"{hours_left:02d}:{mins_left:02d}:{secs_left:02d}"
    else:
        timer_str = f"{mins_left:02d}:{secs_left:02d}"

    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.metric("Started", start_str)
        if priority:
            st.metric("Priority", _priority_label(priority))

    with col2:
        if remaining_secs > 0:
            st.markdown(
                f'<div class="focus-timer">{timer_str}</div>',
                unsafe_allow_html=True,
            )
            st.progress(progress, text=f"{int(progress * 100)}% elapsed")
        else:
            st.markdown(
                '<div class="focus-timer" style="color:#66bb6a;">DONE!</div>',
                unsafe_allow_html=True,
            )
            st.progress(1.0, text="Task time completed!")
            st.balloons()

    with col3:
        st.metric("Ends", end_str)
        loc = entry.get("location")
        if loc:
            st.caption(f"\U0001f4cd {loc}")

    # Task metadata
    notes = entry.get("notes")
    if notes:
        st.caption(f"\U0001f4dd {notes}")

    # Next task preview
    st.markdown("---")
    next_entry, _ = _get_next_task(today_entries, idx)
    if next_entry:
        nt = TYPE_CONFIG.get(next_entry.get("type", ""), DEFAULT_TYPE)
        st.markdown(
            f"**Up next:** {nt['icon']} {next_entry.get('title', 'Untitled')} "
            f"({next_entry.get('start', '?')} \u2013 {next_entry.get('end', '?')})"
        )
    else:
        st.caption("This is your last task for today!")


# ---------------------------------------------------------------------------
# "I'm Behind" rescheduler helpers
# ---------------------------------------------------------------------------

def _get_remaining_tasks(schedule_data: dict):
    """Get unchecked, not-yet-past tasks for today."""
    tz = _get_user_tz()
    now = datetime.now(tz=tz)
    today_name = now.strftime("%A")
    now_minutes = now.hour * 60 + now.minute

    today_entries = schedule_data.get(today_name, [])
    if not isinstance(today_entries, list):
        return [], today_name

    remaining = []
    for i, entry in enumerate(today_entries):
        # Skip checked tasks
        key = _task_key(today_name, entry, i)
        if st.session_state.get(key, False):
            continue
        # Skip breaks/travel (not reschedulable user tasks)
        if entry.get("type") in ("break",):
            continue
        # Skip tasks whose end time has already passed
        try:
            eh, em = map(int, entry.get("end", "0:0").split(":"))
            if eh * 60 + em <= now_minutes:
                continue
        except (ValueError, TypeError):
            pass
        remaining.append(entry)

    return remaining, today_name


def run_reschedule(schedule_data: dict, parsed_tasks: list):
    """Execute the reschedule graph and return updated schedule."""
    tz = _get_user_tz()
    now = datetime.now(tz=tz)
    today_name = now.strftime("%A")
    tomorrow_name = (now + timedelta(days=1)).strftime("%A")
    current_time = now.strftime("%H:%M")

    remaining, _ = _get_remaining_tasks(schedule_data)
    tomorrow_schedule = schedule_data.get(tomorrow_name, [])
    if not isinstance(tomorrow_schedule, list):
        tomorrow_schedule = []

    thread_id = f"resched-{st.session_state.thread_id}-{now.strftime('%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}

    result = reschedule_app.invoke({
        "current_time": current_time,
        "user_timezone": st.session_state.user_timezone,
        "user_location": st.session_state.location,
        "today_name": today_name,
        "tomorrow_name": tomorrow_name,
        "remaining_tasks": remaining,
        "tomorrow_schedule": tomorrow_schedule,
        "parsed_tasks": parsed_tasks,
    }, config)

    return result


# ---------------------------------------------------------------------------
# Macro-Goal Progress rendering
# ---------------------------------------------------------------------------

def render_goal_progress(schedule_data: dict, macro_goals: list):
    """Render progress bars for each weekly macro-goal."""
    if not macro_goals or not schedule_data:
        return

    st.subheader("\U0001f3af Weekly Goal Progress")

    for goal_idx, goal_name in enumerate(macro_goals):
        total = 0
        done = 0
        for day_name, entries in schedule_data.items():
            if not isinstance(entries, list):
                continue
            for i, entry in enumerate(entries):
                if entry.get("goal_id") == goal_idx:
                    total += 1
                    key = _task_key(day_name, entry, i)
                    if st.session_state.get(key, False):
                        done += 1

        if total > 0:
            pct = done / total
            st.progress(pct, text=f"**{goal_name}** \u2014 {done}/{total} tasks ({int(pct*100)}%)")
        else:
            st.progress(0.0, text=f"**{goal_name}** \u2014 No linked tasks found")


# ---------------------------------------------------------------------------
# Streak widget
# ---------------------------------------------------------------------------

def _check_today_p1_status(schedule_data: dict) -> bool:
    """Check if all P1-P3 tasks for today are checked off."""
    today_name = _get_today_name()
    today_entries = schedule_data.get(today_name, [])
    if not isinstance(today_entries, list):
        return True  # No tasks = technically complete

    p1_tasks = [
        (i, e) for i, e in enumerate(today_entries)
        if e.get("priority") is not None and e.get("priority") <= 3
        and e.get("type") not in ("break", "travel")
    ]

    if not p1_tasks:
        return True  # No P1 tasks today

    return all(
        st.session_state.get(_task_key(today_name, e, i), False)
        for i, e in p1_tasks
    )


def render_streak_widget(schedule_data: dict | None = None):
    """Render the current streak metric in the sidebar."""
    streak = session_store.get_current_streak()

    # Update today's status if we have schedule data
    if schedule_data:
        tz = _get_user_tz()
        today_str = datetime.now(tz=tz).strftime("%Y-%m-%d")
        all_p1_done = _check_today_p1_status(schedule_data)
        session_store.record_day_completion(today_str, all_p1_done)

        # If today is done, include it in the count
        if all_p1_done:
            today_status = session_store.is_today_p1_complete(today_str)
            if today_status:
                # Recount including today
                streak = session_store.get_current_streak()

    if streak > 0:
        st.metric(
            "\U0001f525 P1 Streak",
            f"{streak} day{'s' if streak != 1 else ''}",
            delta="Keep it going!",
        )
    else:
        st.metric(
            "\U0001f525 P1 Streak",
            "0 days",
            delta="Complete all P1 tasks to start!",
            delta_color="off",
        )


# ---------------------------------------------------------------------------
# Export helpers — flatten weekly schedule to DataFrame for CSV/Excel download
# ---------------------------------------------------------------------------

def schedule_to_dataframe(schedule_data: dict) -> pd.DataFrame:
    """Flatten the weekly schedule dict into a clean, export-ready DataFrame."""
    rows = []
    for day_name, entries in schedule_data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            t = TYPE_CONFIG.get(entry.get("type", ""), DEFAULT_TYPE)
            notes_parts = []
            if entry.get("notes"):
                notes_parts.append(entry["notes"])
            if entry.get("weather"):
                notes_parts.append(entry["weather"])
            if entry.get("commute"):
                notes_parts.append(entry["commute"])
            goal_name = ""
            goal_id = entry.get("goal_id")
            if goal_id is not None:
                goals = st.session_state.get("macro_goals", [])
                if 0 <= goal_id < len(goals):
                    goal_name = goals[goal_id]
            rows.append({
                "Day": day_name,
                "Start": entry.get("start", ""),
                "End": entry.get("end", ""),
                "Task": f"{t['icon']} {entry.get('title', 'Untitled')}",
                "Type": t["label"],
                "Priority": entry.get("priority", ""),
                "Duration (min)": entry.get("duration_min", ""),
                "Location": entry.get("location", ""),
                "Goal": goal_name,
                "Notes / Weather": " | ".join(notes_parts) if notes_parts else "",
            })
    return pd.DataFrame(rows)


def render_export_section(schedule_data: dict):
    """Render CSV and Excel download buttons for the schedule."""
    df = schedule_to_dataframe(schedule_data)
    if df.empty:
        return

    st.subheader("\U0001f4e5 Export Schedule")

    col_csv, col_xlsx = st.columns(2)

    with col_csv:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="\U0001f4c4 Download CSV",
            data=csv_bytes,
            file_name="AI_Weekly_Schedule.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_xlsx:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Weekly Schedule")
        buf.seek(0)
        st.download_button(
            label="\U0001f4ca Download Excel",
            data=buf.getvalue(),
            file_name="AI_Weekly_Schedule.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Session title helper
# ---------------------------------------------------------------------------

def _derive_title(raw_tasks: str, max_len: int = 40) -> str:
    """Extract a short title from the user's raw task input."""
    text = raw_tasks.strip().replace("\n", ", ")
    if len(text) <= max_len:
        return text or "Untitled schedule"
    return text[:max_len].rsplit(" ", 1)[0] + "..."


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
if "focus_mode" not in st.session_state:
    st.session_state.focus_mode = False
if "macro_goals" not in st.session_state:
    st.session_state.macro_goals = []


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def new_session():
    st.session_state.phase = "input"
    st.session_state.thread_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    st.session_state.user_timezone = "UTC"
    st.session_state.timeline = []
    st.session_state.focus_mode = False
    st.session_state.macro_goals = []
    _clear_checkbox_states()


def load_session(thread_id: str):
    """Rehydrate a past session from LangGraph's checkpoint store."""
    st.session_state.thread_id = thread_id
    st.session_state.timeline = []
    st.session_state.focus_mode = False
    _clear_checkbox_states()

    # Try to read saved state from LangGraph
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph_app.get_state(config)
        values = snapshot.values or {}
    except Exception:
        values = {}

    # Restore location, timezone, and goals if available
    st.session_state.location = values.get("user_location", st.session_state.location)
    st.session_state.user_timezone = values.get("user_timezone", "UTC")
    st.session_state.macro_goals = values.get("macro_goals", [])

    # Determine the right phase
    session_meta = session_store.get_session(thread_id)
    if session_meta and session_meta.get("is_complete"):
        st.session_state.phase = "history"
    elif snapshot.next:
        st.session_state.phase = "review"
    elif values.get("structured_schedule"):
        st.session_state.phase = "history"
    else:
        st.session_state.phase = "input"


# ---------------------------------------------------------------------------
# Sidebar — settings + session history + streak
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    st.session_state.location = st.text_input(
        "Your current location",
        value=st.session_state.location,
        placeholder="e.g. Piata Unirii, Bucuresti",
    )

    st.divider()
    if st.button("\u2795 New Session", use_container_width=True):
        new_session()
        st.rerun()

    # --- Streak Widget (updated with schedule data if available) ---
    st.divider()
    _sidebar_schedule = None
    if st.session_state.phase in ("done", "review", "history"):
        try:
            _snap = graph_app.get_state(get_config())
            _sidebar_schedule = (_snap.values or {}).get("structured_schedule")
        except Exception:
            pass
    render_streak_widget(_sidebar_schedule)

    # --- Past Sessions ---
    st.divider()
    st.subheader("\U0001f4cb Past Sessions")

    past_sessions = session_store.list_sessions(limit=20)

    if not past_sessions:
        st.caption("No past sessions yet.")
    else:
        for sess in past_sessions:
            tid = sess["thread_id"]
            title = sess["title"]
            is_current = tid == st.session_state.thread_id
            is_complete = sess.get("is_complete", 0)

            # Format the date
            try:
                dt = datetime.fromisoformat(sess["updated_at"])
                date_str = dt.strftime("%b %d, %H:%M")
            except Exception:
                date_str = sess.get("updated_at", "")[:16]

            status_icon = "\u2705" if is_complete else "\U0001f504"
            btn_label = f"{status_icon} {title}"

            col_btn, col_del = st.columns([5, 1])

            with col_btn:
                if is_current:
                    st.markdown(
                        f"**\u25b6 {title}**  \n"
                        f"<span style='color:rgba(250,250,250,0.4);font-size:0.75rem;'>"
                        f"{date_str}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(btn_label, key=f"load_{tid}", use_container_width=True):
                        load_session(tid)
                        st.rerun()

            with col_del:
                if st.button("\U0001f5d1", key=f"del_{tid}", help="Delete session"):
                    session_store.delete_session(tid)
                    if is_current:
                        new_session()
                    st.rerun()

            # Show date under button
            if not is_current:
                st.caption(date_str)

    st.divider()
    st.caption("Powered by Gemini 2.5 Pro + LangGraph")
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

    # --- Macro-Goals input ---
    st.subheader("\U0001f3af Weekly Goals (optional)")
    st.caption("Set 1-3 macro-goals to track progress across the week.")
    goal_cols = st.columns(3)
    macro_goals = []
    for i, col in enumerate(goal_cols):
        with col:
            g = st.text_input(
                f"Goal {i + 1}",
                key=f"input_macro_goal_{i}",
                placeholder=["e.g. Finish Physics Project",
                              "e.g. Hit the gym 3 times",
                              "e.g. Clear email backlog"][i],
            )
            if g.strip():
                macro_goals.append(g.strip())

    can_submit = bool(raw_tasks and raw_tasks.strip()) or bool(uploaded_files)

    if st.button("Plan My Week", type="primary", disabled=not can_submit):
        # Save macro goals to session state
        st.session_state.macro_goals = macro_goals

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

        # Step 3: save session metadata
        title = _derive_title(raw_tasks or "Uploaded documents")
        session_store.save_session(
            thread_id=st.session_state.thread_id,
            title=title,
            location=st.session_state.location,
        )

        # Step 4: invoke the graph
        initial_state = {
            "raw_tasks": raw_tasks or "See uploaded documents below.",
            "user_location": st.session_state.location,
            "user_timezone": user_tz,
            "uploaded_files_text": extracted_text,
            "macro_goals": macro_goals,
        }

        config = get_config()
        success = run_graph_streaming(initial_state, config)

        if success:
            snapshot = graph_app.get_state(config)
            st.session_state.phase = "review" if snapshot.next else "done"
            st.rerun()


# ===========================================================================
# Phase: REVIEW — interactive checklist with weekly tabbed view
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
        _clear_checkbox_states()
        st.session_state.timeline.append({
            "type": "schedule",
            "_key": timeline_key,
            "text": draft_text,
            "structured": structured,
            "critique": critique,
            "revision": revisions,
        })

    # --- Goal Progress (if goals set) ---
    if st.session_state.macro_goals and structured:
        render_goal_progress(structured, st.session_state.macro_goals)
        st.markdown("---")

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

    # --- Render latest schedule as interactive checklist ---
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
                if snap.next:
                    st.session_state.phase = "review"
                else:
                    st.session_state.phase = "done"
                    session_store.mark_complete(st.session_state.thread_id)
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
# Phase: DONE — final weekly checklist dashboard
# ===========================================================================

elif st.session_state.phase == "done":
    config = get_config()
    final = graph_app.get_state(config).values

    draft_text = final.get("draft_schedule", "No schedule generated.")
    structured = final.get("structured_schedule", {})
    critique = final.get("critique", "")
    rev_count = final.get("revision_count", 0)
    parsed_tasks = final.get("parsed_tasks", [])

    # Mark session complete in the store
    session_store.mark_complete(st.session_state.thread_id)

    # --- Focus Mode toggle + I'm Behind button ---
    if structured:
        top_col1, top_col2, top_col3 = st.columns([1, 1, 2])

        with top_col1:
            focus_on = st.toggle(
                "\U0001f3af Focus Mode",
                value=st.session_state.focus_mode,
                key="focus_toggle",
            )
            st.session_state.focus_mode = focus_on

        with top_col2:
            remaining, _ = _get_remaining_tasks(structured)
            if remaining:
                if st.button(
                    "\U0001f6a8 I'm Behind",
                    type="primary",
                    help="Compress breaks, push low-priority to tomorrow",
                    use_container_width=True,
                ):
                    with st.status("Rescheduling your day...", expanded=True) as rs:
                        try:
                            result = run_reschedule(structured, parsed_tasks)
                            rescheduled_today = result.get("rescheduled_today", [])
                            rescheduled_tomorrow = result.get("rescheduled_tomorrow", [])

                            today_name = _get_today_name()
                            tomorrow_name = _get_tomorrow_name()

                            # Update the schedule in place
                            updated = dict(structured)
                            if rescheduled_today:
                                updated[today_name] = rescheduled_today
                            if rescheduled_tomorrow:
                                updated[tomorrow_name] = rescheduled_tomorrow

                            # Persist back to LangGraph checkpoint
                            graph_app.update_state(
                                config,
                                {
                                    "structured_schedule": updated,
                                    "draft_schedule": _schedule_to_text(updated),
                                },
                                as_node="human_review",
                            )

                            _clear_checkbox_states()
                            rs.update(label="Day rescheduled!", state="complete")
                            st.rerun()
                        except Exception as e:
                            rs.update(label=f"Error: {e}", state="error")
                            st.error(f"Rescheduling failed: {e}")

        with top_col3:
            pass  # Spacer

        st.markdown("---")

    # --- Goal Progress ---
    if st.session_state.macro_goals and structured:
        render_goal_progress(structured, st.session_state.macro_goals)
        st.markdown("---")

    # --- Main schedule view (Focus Mode or Full Calendar) ---
    if st.session_state.focus_mode and structured:
        st.subheader("\U0001f3af Focus Mode")
        render_focus_mode(structured)
    else:
        st.subheader("Final Optimized Weekly Schedule")
        render_weekly_schedule(structured, draft_text)

    if critique and not st.session_state.focus_mode:
        with st.expander("Final Assessment", expanded=True):
            st.markdown(critique)

    if not st.session_state.focus_mode:
        st.success(f"Weekly schedule complete! Total revisions: {rev_count}")

        # Export buttons
        if structured:
            st.divider()
            render_export_section(structured)

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


# ===========================================================================
# Phase: HISTORY — read-only view of a past session loaded from checkpoint
# ===========================================================================

elif st.session_state.phase == "history":
    config = get_config()
    try:
        snapshot = graph_app.get_state(config)
        values = snapshot.values or {}
    except Exception:
        st.warning("Could not load this session's data. The checkpoint may have been cleared.")
        values = {}

    draft_text = values.get("draft_schedule", "No schedule data available.")
    structured = values.get("structured_schedule", {})
    critique = values.get("critique", "")
    rev_count = values.get("revision_count", 0)

    # Show session metadata
    session_meta = session_store.get_session(st.session_state.thread_id)
    if session_meta:
        try:
            created = datetime.fromisoformat(session_meta["created_at"]).strftime("%B %d, %Y at %H:%M")
        except Exception:
            created = session_meta.get("created_at", "Unknown")
        st.caption(f"\U0001f4c6 Created: {created}  |  \U0001f4cd {session_meta.get('location', 'N/A')}")

    # --- Goal Progress (if this session had goals) ---
    goals = values.get("macro_goals", [])
    if goals and structured:
        st.session_state.macro_goals = goals
        render_goal_progress(structured, goals)
        st.markdown("---")

    st.subheader("\U0001f4c2 Past Schedule")

    if structured:
        render_weekly_schedule(structured, draft_text)
    else:
        st.markdown(draft_text)

    if critique:
        with st.expander("Critic's Assessment", expanded=False):
            st.markdown(critique)

    if rev_count:
        st.info(f"This schedule was finalized after {rev_count} revision(s).")

    # Export buttons for past schedules too
    if structured:
        st.divider()
        render_export_section(structured)

    st.divider()
    if st.button("Plan Another Week", type="primary"):
        new_session()
        st.rerun()
