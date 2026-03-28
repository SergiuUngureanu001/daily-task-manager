"""
Lightweight SQLite session metadata store.

Uses the same DB directory as the LangGraph checkpointer so a single
Docker volume persists both graph state and session history.
"""

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


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_STREAKS_TABLE)
    conn.commit()
    return conn


_conn = _get_conn()


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
