"""
Lightweight SQLite metadata store for sessions, templates, goals, and streaks.

Uses the same DB directory as the LangGraph checkpointer so a single
Docker volume persists both graph state and session history.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta

# Same DB directory logic as graph.py
_DB_DIR = "/app/data" if os.path.isdir("/app/data") else "."
DB_PATH = os.path.join(_DB_DIR, "session_history.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id   TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    location    TEXT NOT NULL DEFAULT 'Home',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    is_complete INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_STREAKS_TABLE = """
CREATE TABLE IF NOT EXISTS streaks (
    date        TEXT PRIMARY KEY,
    all_p1_done INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_TEMPLATES_TABLE = """
CREATE TABLE IF NOT EXISTS templates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    tasks_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""

_CREATE_GOALS_TABLE = """
CREATE TABLE IF NOT EXISTS long_term_goals (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_name               TEXT NOT NULL,
    deadline_date           TEXT NOT NULL,
    total_hours_estimated   REAL NOT NULL DEFAULT 0,
    hours_completed         REAL NOT NULL DEFAULT 0,
    status                  TEXT NOT NULL DEFAULT 'active',
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
)
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_STREAKS_TABLE)
    conn.execute(_CREATE_TEMPLATES_TABLE)
    conn.execute(_CREATE_GOALS_TABLE)
    conn.commit()
    return conn


_conn = _get_conn()


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def save_session(thread_id: str, title: str, location: str = "Home",
                 is_complete: bool = False):
    """Insert or update a session record."""
    now = datetime.utcnow().isoformat()
    _conn.execute(
        """
        INSERT INTO sessions (thread_id, title, location, created_at, updated_at, is_complete)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            title = excluded.title,
            location = excluded.location,
            updated_at = excluded.updated_at,
            is_complete = excluded.is_complete
        """,
        (thread_id, title, location, now, now, int(is_complete)),
    )
    _conn.commit()


def mark_complete(thread_id: str):
    """Mark a session as approved/done."""
    now = datetime.utcnow().isoformat()
    _conn.execute(
        "UPDATE sessions SET is_complete = 1, updated_at = ? WHERE thread_id = ?",
        (now, thread_id),
    )
    _conn.commit()


def list_sessions(limit: int = 30) -> list[dict]:
    """Return recent sessions, newest first."""
    rows = _conn.execute(
        "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_session(thread_id: str):
    """Remove a session record (does NOT delete LangGraph checkpoint data)."""
    _conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
    _conn.commit()


def get_session(thread_id: str) -> dict | None:
    """Fetch a single session by thread_id."""
    row = _conn.execute(
        "SELECT * FROM sessions WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Streak tracking — records daily P1 completion status
# ---------------------------------------------------------------------------

def record_day_completion(date_str: str, all_p1_done: bool):
    """Upsert a streak record for the given date (YYYY-MM-DD)."""
    _conn.execute(
        """
        INSERT INTO streaks (date, all_p1_done) VALUES (?, ?)
        ON CONFLICT(date) DO UPDATE SET all_p1_done = excluded.all_p1_done
        """,
        (date_str, int(all_p1_done)),
    )
    _conn.commit()


def get_current_streak() -> int:
    """
    Count consecutive days (ending yesterday or today) where all P1 tasks
    were completed. Returns 0 if yesterday was incomplete.
    """
    rows = _conn.execute(
        "SELECT date, all_p1_done FROM streaks ORDER BY date DESC LIMIT 60"
    ).fetchall()

    if not rows:
        return 0

    streak = 0
    today = datetime.utcnow().date()

    # Build a date -> done map
    done_map = {}
    for r in rows:
        done_map[r["date"]] = bool(r["all_p1_done"])

    # Count backwards from today
    check_date = today
    for _ in range(60):
        ds = check_date.strftime("%Y-%m-%d")
        if ds in done_map:
            if done_map[ds]:
                streak += 1
            else:
                break
        else:
            # No record for this date — if it's today that's ok (day not over)
            if check_date == today:
                check_date -= timedelta(days=1)
                continue
            break
        check_date -= timedelta(days=1)

    return streak


def is_today_p1_complete(date_str: str) -> bool | None:
    """Check if today's P1 completion has been recorded. Returns None if no record."""
    row = _conn.execute(
        "SELECT all_p1_done FROM streaks WHERE date = ?", (date_str,)
    ).fetchone()
    if row is None:
        return None
    return bool(row["all_p1_done"])


# ---------------------------------------------------------------------------
# Template CRUD — recurring weekly schedule templates
# ---------------------------------------------------------------------------

def save_template(name: str, tasks: list[dict]):
    """Create or update a named template. tasks is a list of task dicts."""
    now = datetime.utcnow().isoformat()
    tasks_json = json.dumps(tasks, ensure_ascii=False)
    _conn.execute(
        """
        INSERT INTO templates (name, tasks_json, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            tasks_json = excluded.tasks_json,
            updated_at = excluded.updated_at
        """,
        (name, tasks_json, now, now),
    )
    _conn.commit()


def list_templates() -> list[dict]:
    """Return all templates, newest first."""
    rows = _conn.execute(
        "SELECT id, name, tasks_json, created_at, updated_at FROM templates ORDER BY updated_at DESC"
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["tasks"] = json.loads(d.pop("tasks_json"))
        results.append(d)
    return results


def get_template(name: str) -> dict | None:
    """Fetch a single template by name."""
    row = _conn.execute(
        "SELECT id, name, tasks_json, created_at, updated_at FROM templates WHERE name = ?",
        (name,),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["tasks"] = json.loads(d.pop("tasks_json"))
    return d


def delete_template(name: str):
    """Remove a template by name."""
    _conn.execute("DELETE FROM templates WHERE name = ?", (name,))
    _conn.commit()


# ---------------------------------------------------------------------------
# Long-Term Goal CRUD
# ---------------------------------------------------------------------------

def save_goal(goal_name: str, deadline_date: str, total_hours: float,
              hours_completed: float = 0.0) -> int:
    """Create a new long-term goal. Returns the new row id."""
    now = datetime.utcnow().isoformat()
    cursor = _conn.execute(
        """
        INSERT INTO long_term_goals
            (goal_name, deadline_date, total_hours_estimated, hours_completed, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'active', ?, ?)
        """,
        (goal_name, deadline_date, total_hours, hours_completed, now, now),
    )
    _conn.commit()
    return cursor.lastrowid


def update_goal(goal_id: int, **kwargs):
    """Update specific fields of a goal. Accepts: goal_name, deadline_date,
    total_hours_estimated, hours_completed, status."""
    allowed = {"goal_name", "deadline_date", "total_hours_estimated",
               "hours_completed", "status"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    updates["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [goal_id]
    _conn.execute(
        f"UPDATE long_term_goals SET {set_clause} WHERE id = ?",
        values,
    )
    _conn.commit()


def list_goals(status: str | None = "active") -> list[dict]:
    """Return goals filtered by status (or all if status is None)."""
    if status:
        rows = _conn.execute(
            "SELECT * FROM long_term_goals WHERE status = ? ORDER BY deadline_date ASC",
            (status,),
        ).fetchall()
    else:
        rows = _conn.execute(
            "SELECT * FROM long_term_goals ORDER BY deadline_date ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_goal(goal_id: int) -> dict | None:
    """Fetch a single goal by id."""
    row = _conn.execute(
        "SELECT * FROM long_term_goals WHERE id = ?", (goal_id,)
    ).fetchone()
    return dict(row) if row else None


def delete_goal(goal_id: int):
    """Remove a goal by id."""
    _conn.execute("DELETE FROM long_term_goals WHERE id = ?", (goal_id,))
    _conn.commit()
