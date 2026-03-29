import base64
import io
import json
import os
import re
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from pypdf import PdfReader

try:
    from google.api_core.exceptions import ResourceExhausted as GoogleRateLimitError
except ImportError:
    GoogleRateLimitError = None

from state import SchedulerState, RescheduleState


# ---------------------------------------------------------------------------
# Pydantic models for schema-validated schedule output
# ---------------------------------------------------------------------------

class ScheduledTask(BaseModel):
    """A single time-blocked entry in the weekly schedule."""
    start: str = Field(description="Start time in HH:MM format")
    end: str = Field(description="End time in HH:MM format")
    title: str = Field(description="Exact task name from the checklist")
    type: Literal["work", "break", "travel", "fitness", "call", "errand", "meal", "shower"]
    priority: Optional[int] = Field(default=None, description="1-10 for tasks, null for breaks/travel")
    duration_min: int = Field(description="Duration in minutes")
    location: Optional[str] = None
    notes: Optional[str] = None
    weather: Optional[str] = None
    commute: Optional[str] = None
    goal_id: Optional[str] = Field(
        default=None,
        description="String goal slug from the GOAL ID LOOKUP TABLE, or null if unrelated"
    )


class WeeklySchedule(BaseModel):
    """Complete 7-day schedule with schema-validated entries."""
    Monday: list[ScheduledTask] = Field(default_factory=list)
    Tuesday: list[ScheduledTask] = Field(default_factory=list)
    Wednesday: list[ScheduledTask] = Field(default_factory=list)
    Thursday: list[ScheduledTask] = Field(default_factory=list)
    Friday: list[ScheduledTask] = Field(default_factory=list)
    Saturday: list[ScheduledTask] = Field(default_factory=list)
    Sunday: list[ScheduledTask] = Field(default_factory=list)

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------------
# LLM setup — Gemini 2.5 Pro via Google AI API
# Two instances: one precise (temperature=0, no thinking) for structured JSON
# output, and one with thinking enabled for the critic's reasoning.
# ---------------------------------------------------------------------------
_gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")

# Structured output LLM — temperature=0 for reliable JSON, no thinking
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=_gemini_api_key,
    temperature=0,
    max_retries=0,
    timeout=180,
)

# Thinking LLM — used only for the critic node where reasoning helps
llm_thinking = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=_gemini_api_key,
    temperature=1,
    thinking_budget=10000,
    max_retries=0,
    timeout=180,
)

# Structured output LLM — available for Pydantic-validated JSON responses.
# NOT used in the main scheduler flow because Gemini's structured output mode
# truncates large responses (70+ entries), causing severe task dropping.
# Instead, the scheduler generates raw JSON and validates with Pydantic
# post-hoc in _validate_with_pydantic() + _enforce_goal_ids().
llm_structured = llm.with_structured_output(WeeklySchedule)


# ---------------------------------------------------------------------------
# Rate-limit-aware LLM invoke wrapper with exponential backoff
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
BASE_DELAY = 15  # seconds


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect whether an exception is a rate-limit error (429 / ResourceExhausted)."""
    # 1. Google AI SDK raises google.api_core.exceptions.ResourceExhausted
    if GoogleRateLimitError and isinstance(exc, GoogleRateLimitError):
        return True
    # 2. Check the HTTP status code if available (langchain wrappers)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    # 3. Fallback: string matching on error message
    error_str = str(exc).lower()
    return "429" in error_str or "rate_limit" in error_str or "resource_exhausted" in error_str


def _invoke_with_backoff(llm_instance, messages):
    """
    Call llm_instance.invoke(messages) with automatic retry on 429 rate-limit
    errors. Uses exponential backoff: 20s, 40s, 80s, 160s, 320s.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm_instance.invoke(messages)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** (attempt - 1))
                print(f"    [Rate limit] Attempt {attempt}/{MAX_RETRIES} — "
                      f"waiting {delay}s before retry...")
                time.sleep(delay)
                continue
            # Not a rate limit error, or out of retries — raise
            raise


def extract_text(content) -> str:
    """
    The LLM can return content as a plain string OR as a list of content blocks
    (e.g. [{'type': 'text', 'text': '...', 'extras': {...}}]).
    This helper always returns a clean string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Timezone resolution — derives IANA timezone from a location name
# ---------------------------------------------------------------------------

def resolve_timezone(location: str) -> str:
    """
    Geocode a location via Open-Meteo and return its IANA timezone
    (e.g. 'Europe/Bucharest'). Falls back to UTC on failure.
    """
    for query in [location, location.split(",")[0].strip()]:
        try:
            url = (
                "https://geocoding-api.open-meteo.com/v1/search"
                f"?name={requests.utils.quote(query)}&count=1"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results and "timezone" in results[0]:
                tz = results[0]["timezone"]
                ZoneInfo(tz)  # validate it
                return tz
        except Exception:
            continue
    return "UTC"


def _now(state) -> str:
    """Return the current time formatted for prompts, using the user's timezone."""
    tz_name = state.get("user_timezone", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz=tz).strftime("%A, %B %d, %Y at %H:%M")


def _week_dates(state) -> list[dict]:
    """Return a list of {day_name, date_str} for the next 7 days starting today."""
    tz_name = state.get("user_timezone", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    today = datetime.now(tz=tz).date()
    days = []
    for i in range(7):
        d = today + timedelta(days=i)
        days.append({
            "day_name": d.strftime("%A"),
            "date_str": d.strftime("%Y-%m-%d"),
            "display": d.strftime("%A, %B %d"),
        })
    return days


# ---------------------------------------------------------------------------
# Schedule JSON parsing helpers (weekly format)
# ---------------------------------------------------------------------------

def _find_all_json_objects(text: str) -> list[dict]:
    """
    Scan text for every top-level JSON object ({...}) and return all that
    parse successfully. Handles LLM outputs with reasoning text mixed
    between multiple JSON blocks.
    """
    results = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            # Track brace nesting to find the matching close brace
            depth = 0
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    if in_string:
                        escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j + 1]
                        try:
                            data = json.loads(candidate)
                            if isinstance(data, dict):
                                results.append(data)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
            i = j + 1 if depth == 0 else i + 1
        else:
            i += 1
    return results


def _is_weekly_schedule(data: dict) -> bool:
    """Check if a dict looks like a weekly schedule (day names mapping to lists)."""
    if not data:
        return False
    day_names = {"monday", "tuesday", "wednesday", "thursday", "friday",
                 "saturday", "sunday", "today"}
    matching = sum(1 for k in data if k.lower() in day_names)
    return matching >= 1 and all(isinstance(v, list) for v in data.values())


def _parse_schedule_response(response_content):
    """
    Parse the scheduler's LLM response into (structured_dict, text_version).
    The structured_dict is keyed by day name: {"Monday": [...], "Tuesday": [...]}.
    Handles messy LLM outputs with reasoning text mixed between JSON blocks.
    Falls back to ({}, raw_text) if all extraction attempts fail.
    """
    raw = extract_text(response_content)

    # Try direct JSON parse (clean response — no surrounding text)
    for attempt in [raw, raw.strip().strip("`").removeprefix("json").strip()]:
        try:
            data = json.loads(attempt)
            if isinstance(data, dict):
                return data, _schedule_to_text(data)
            if isinstance(data, list):
                wrapped = {"Today": data}
                return wrapped, _schedule_to_text(wrapped)
        except (json.JSONDecodeError, TypeError):
            continue

    # Extract all JSON objects from mixed text and pick the best one.
    # The LLM often outputs reasoning, a first draft, corrections, then
    # a final JSON — so we prefer the LAST valid weekly schedule object,
    # and among those the LARGEST (most days/entries).
    candidates = _find_all_json_objects(raw)
    weekly_candidates = [c for c in candidates if _is_weekly_schedule(c)]
    if weekly_candidates:
        # Prefer the last one (the corrected/final version)
        best = weekly_candidates[-1]
        print(f"  Extracted weekly schedule: {list(best.keys())} "
              f"({sum(len(v) for v in best.values() if isinstance(v, list))} entries)")
        return best, _schedule_to_text(best)

    # If no weekly dict found, try any dict candidate
    if candidates:
        best = candidates[-1]
        return best, _schedule_to_text(best)

    # Try to extract a JSON array (legacy single-day format)
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        data = json.loads(raw[start:end])
        if isinstance(data, list):
            wrapped = {"Today": data}
            return wrapped, _schedule_to_text(wrapped)
    except (ValueError, json.JSONDecodeError):
        pass

    print("  Warning: could not parse schedule JSON, using raw text.")
    return {}, raw


def _schedule_to_text(week_data):
    """Convert structured weekly schedule dict to readable text for the critic."""
    lines = []
    for day_name, entries in week_data.items():
        lines.append(f"\n{'='*40}")
        lines.append(f"  {day_name}")
        lines.append(f"{'='*40}")
        if not isinstance(entries, list):
            lines.append(str(entries))
            continue
        for e in entries:
            p = f" [Priority {e['priority']}]" if e.get("priority") else ""
            lines.append(
                f"{e.get('start','?')} - {e.get('end','?')} | "
                f"{e.get('title','Untitled')}{p} ({e.get('duration_min','?')} min)"
            )
            if e.get("location"):
                lines.append(f"  Location: {e['location']}")
            if e.get("notes"):
                lines.append(f"  {e['notes']}")
            if e.get("weather"):
                lines.append(f"  Weather: {e['weather']}")
            if e.get("commute"):
                lines.append(f"  Commute: {e['commute']}")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Goal slug generation — deterministic string IDs for macro goals
# ---------------------------------------------------------------------------

_SLUG_STOP = {"a", "the", "of", "to", "and", "for", "from", "in", "on", "at",
              "my", "get", "do", "all", "pages", "page", "hours", "hour",
              "finish", "complete", "start", "begin", "100", "50", "200", "300"}


def _goal_slug(goal_name: str) -> str:
    """
    Generate a deterministic, semantic string ID from a goal name.
    Examples:
        "Read 100 pages from Brothers Karamazov" → "goal_read_brothers_karamazov"
        "Get Driving License"                    → "goal_driving_license"
        "Build REST API for internship"          → "goal_build_rest_api"
    """
    words = re.findall(r'[a-z]+', goal_name.lower())
    meaningful = [w for w in words if w not in _SLUG_STOP and len(w) > 2][:3]
    core = "_".join(meaningful) if meaningful else "task"
    return f"goal_{core}"


def _build_goal_slug_map(macro_goals: list[str]) -> dict[str, str]:
    """
    Build a slug → goal_name mapping from the user's macro goals.
    Handles collisions by appending a counter.
    Returns: {"goal_read_brothers_karamazov": "Read 100 pages from Brothers Karamazov", ...}
    """
    slug_map: dict[str, str] = {}
    for goal_name in macro_goals:
        slug = _goal_slug(goal_name)
        base_slug = slug
        counter = 2
        while slug in slug_map:
            slug = f"{base_slug}_{counter}"
            counter += 1
        slug_map[slug] = goal_name
    return slug_map


# ---------------------------------------------------------------------------
# Goal-ID enforcement & Pydantic validation
# ---------------------------------------------------------------------------

# Entry types that NEVER get a goal_id (even if the LLM assigns one)
_NULL_GOAL_TYPES = {"break", "travel", "meal", "shower"}


def _build_goal_id_map(parsed_tasks: list[dict]) -> dict[str, str]:
    """
    Build an authoritative task_name -> goal_id (string slug) lookup from
    the ingested tasks. Keys are lowercased task names.
    Only tasks with a non-null goal_id appear.
    """
    mapping: dict[str, str] = {}
    for t in parsed_tasks:
        name = t.get("task", "").strip().lower()
        gid = t.get("goal_id")
        if name and gid is not None:
            mapping[name] = str(gid)
    return mapping


def _naive_stem(word: str) -> str:
    """
    Minimal English stemmer — strips common suffixes to normalize verb/noun forms.
    E.g. reading→read, studies→studi, completed→complet, practices→practic.
    Not perfect, but good enough for keyword overlap matching.
    """
    if len(word) <= 3:
        return word
    for suffix in ("ying", "ting", "ning", "ring", "ping",
                   "ding", "king", "ling", "ming", "sing"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix) + 1]  # keep the base consonant
    for suffix in ("ing", "tion", "sion", "ment", "ness", "ence", "ance"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    for suffix in ("ed", "es", "er", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _fuzzy_match(a: str, b: str) -> bool:
    """
    Check if two words are close enough to be considered a match.
    Uses prefix matching to handle spelling variants like
    'dostoyevsky' vs 'dostoviesky'. Requires both words to be long
    enough and share a substantial prefix to avoid false positives
    like 'comput' matching 'compete'.
    """
    if a == b:
        return True
    # Both words must be at least 5 chars to fuzzy match (short words need exact)
    if len(a) < 5 or len(b) < 5:
        return False
    # Require a 5-char prefix match and similar length
    if a[:5] == b[:5] and abs(len(a) - len(b)) <= 3:
        return True
    return False


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from a text string, filtering stop words."""
    stop = {"a", "the", "of", "to", "and", "for", "in", "on", "at", "from",
            "by", "with", "is", "it", "my", "i", "do", "an", "or", "be",
            "this", "that", "these", "those", "all", "each", "every", "no",
            "not", "but", "so", "if", "up", "out", "about", "into", "over",
            "after", "before", "between", "under", "through", "during",
            "pages", "page", "chapters", "chapter", "hours", "hour",
            "minutes", "min", "session", "sessions", "part", "week", "weekly"}
    words = set(text.lower().split()) - stop
    # Strip digits and punctuation-only tokens, then stem
    cleaned = {w.strip(".,;:!?()[]{}\"'-/—–") for w in words if len(w) > 1}
    # Remove pure numeric tokens
    cleaned = {w for w in cleaned if w and not w.isdigit()}
    return {_naive_stem(w) for w in cleaned}


def _fuzzy_keyword_overlap(kw_a: set[str], kw_b: set[str]) -> int:
    """
    Count the number of fuzzy-matching keywords between two sets.
    Uses exact match first, then falls back to _fuzzy_match for remaining words.
    """
    # Exact matches
    exact = kw_a & kw_b
    count = len(exact)
    # Fuzzy matches for remaining
    remaining_a = kw_a - exact
    remaining_b = kw_b - exact
    used_b = set()
    for wa in remaining_a:
        for wb in remaining_b:
            if wb not in used_b and _fuzzy_match(wa, wb):
                count += 1
                used_b.add(wb)
                break
    return count


def _enforce_goal_ids(structured: dict, parsed_tasks: list[dict],
                      macro_goals: list[str] | None = None) -> dict:
    """
    Programmatically enforce correct goal_id on every scheduled entry.

    Uses a THREE-LAYER matching strategy:
    Layer 1: Exact title match against parsed_tasks (task_name → goal_id)
    Layer 2: Substring/keyword match against parsed_tasks
    Layer 3: Keyword match against macro_goals names directly
             (catches cases where the scheduler renamed the task but the
             title still clearly relates to a goal's theme)

    Rules:
    1. break/travel/meal/shower → always null
    2. Title matches a parsed task with goal_id → copy that goal_id
    3. Title keywords overlap with a goal name → assign that goal's index
    4. Everything else → null
    """
    goal_map = _build_goal_id_map(parsed_tasks)

    # Build keyword sets for each macro goal for Layer 3 matching
    # Uses string slugs instead of integer indices
    macro_goal_keywords: list[tuple[str, set[str]]] = []
    if macro_goals:
        slug_map = _build_goal_slug_map(macro_goals)
        for slug, gname in slug_map.items():
            kw = _extract_keywords(gname)
            if kw:
                macro_goal_keywords.append((slug, kw))

    corrections = 0

    for day, entries in structured.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue

            old_gid = entry.get("goal_id")

            # Rule 1: non-work types always null
            if entry.get("type") in _NULL_GOAL_TYPES:
                entry["goal_id"] = None
                if old_gid is not None:
                    corrections += 1
                continue

            title_lower = entry.get("title", "").strip().lower()

            # Layer 1: exact match against parsed_tasks
            if title_lower in goal_map:
                correct_gid = goal_map[title_lower]
                if old_gid != correct_gid:
                    corrections += 1
                entry["goal_id"] = correct_gid
                continue

            # Layer 2: substring/keyword match against parsed_tasks
            matched_gid = None
            for task_name, gid in goal_map.items():
                # Substring check
                if task_name in title_lower or title_lower in task_name:
                    matched_gid = gid
                    break
                # Fuzzy keyword overlap — require at least 1 meaningful match
                task_kw = _extract_keywords(task_name)
                title_kw = _extract_keywords(title_lower)
                if task_kw and title_kw:
                    overlap_count = _fuzzy_keyword_overlap(task_kw, title_kw)
                    if overlap_count >= 1:
                        matched_gid = gid
                        break

            if matched_gid is not None:
                if old_gid != matched_gid:
                    corrections += 1
                entry["goal_id"] = matched_gid
                continue

            # Layer 3: keyword match against macro_goals names directly
            # This catches renamed tasks: e.g. title="Read Karamazov pages 1-25"
            # matches goal="Read 100 pages from Brothers Karamazov"
            title_kw = _extract_keywords(title_lower)
            if title_kw and macro_goal_keywords:
                best_slug = None
                best_score = 0
                for goal_slug, goal_kw in macro_goal_keywords:
                    overlap_count = _fuzzy_keyword_overlap(title_kw, goal_kw)
                    if overlap_count >= 1 and overlap_count > best_score:
                        best_score = overlap_count
                        best_slug = goal_slug
                if best_slug is not None:
                    if old_gid != best_slug:
                        corrections += 1
                    entry["goal_id"] = best_slug
                    continue

            # No match — null
            if old_gid is not None:
                corrections += 1
            entry["goal_id"] = None

    if corrections:
        print(f"  [Goal-ID Enforcement] Corrected {corrections} goal_id value(s)")
    return structured


def _validate_with_pydantic(structured: dict) -> dict:
    """
    Validate the schedule dict through the WeeklySchedule Pydantic model.
    Returns a clean dict with guaranteed correct types and structure.
    Falls back gracefully: invalid entries are kept as-is if they have
    the minimum required fields (start, title).
    """
    from pydantic import ValidationError

    # Normalize keys: ensure all 7 days exist
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

    # Map any case-insensitive keys to proper case
    normalized = {}
    lower_map = {d.lower(): d for d in day_names}
    for key, val in structured.items():
        proper = lower_map.get(key.lower(), key)
        normalized[proper] = val

    # Try full model validation
    try:
        schedule = WeeklySchedule(**normalized)
        result = schedule.model_dump()
        print("  [Pydantic] Full schedule validated successfully")
        return result
    except ValidationError as e:
        print(f"  [Pydantic] Full validation failed ({len(e.errors())} errors), validating per-entry...")

    # Fallback: validate each entry independently, keep valid ones
    result = {}
    for day in day_names:
        entries = normalized.get(day, [])
        if not isinstance(entries, list):
            result[day] = []
            continue
        valid_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                task = ScheduledTask(**entry)
                valid_entries.append(task.model_dump())
            except ValidationError:
                # Keep the entry if it has at least start + title (minimum for display)
                if entry.get("start") and entry.get("title"):
                    # Ensure goal_id is string or None
                    gid = entry.get("goal_id")
                    if gid is not None:
                        entry["goal_id"] = str(gid) if gid else None
                    valid_entries.append(entry)
        result[day] = valid_entries

    validated = sum(len(v) for v in result.values())
    print(f"  [Pydantic] Kept {validated} entries after per-entry validation")
    return result


# ---------------------------------------------------------------------------
# Document processing helpers — called from app.py before graph invocation
# ---------------------------------------------------------------------------

def process_uploaded_file(filename: str, file_bytes: bytes) -> str:
    """
    Process an uploaded file and extract actionable task text.
    Handles .txt, .pdf (via pypdf), and images (via Gemini vision).
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "txt":
        return file_bytes.decode("utf-8", errors="replace")

    elif ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return "(PDF contained no extractable text)"
        response = _invoke_with_backoff(llm, [HumanMessage(content=(
            "CRITICAL: Extract EVERY single task, event, class, appointment, "
            "deadline, and to-do item from this document.\n\n"
            "Rules:\n"
            "1. Preserve EXACT names — do NOT summarize, paraphrase, or merge.\n"
            "2. Include ALL time, day, date, and location information.\n"
            "3. Number each extracted item.\n"
            "4. Do NOT skip any items.\n\n"
            "Document text:\n" + text
        ))])
        return extract_text(response.content)

    elif ext in ("png", "jpg", "jpeg"):
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        media_type = "image/png" if ext == "png" else "image/jpeg"
        response = _invoke_with_backoff(llm, [HumanMessage(content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            {
                "type": "text",
                "text": (
                    "CRITICAL: Read this image VERY carefully and extract EVERY single item you can see.\n\n"
                    "Rules:\n"
                    "1. Extract EVERY task, event, class, appointment, deadline, and to-do item visible.\n"
                    "2. Preserve the EXACT names as written — do NOT summarize, paraphrase, or merge items.\n"
                    "   Example: if the image says 'Calculus II lecture' write exactly 'Calculus II lecture', NOT 'University class'.\n"
                    "3. Include ALL time information (days, hours, dates) exactly as shown.\n"
                    "4. Include ALL location information exactly as shown.\n"
                    "5. If it's a weekly schedule/timetable, list EVERY entry for EVERY day.\n"
                    "6. Number each extracted item.\n\n"
                    "Return a numbered list. Do NOT skip any items. Do NOT generalize."
                ),
            },
        ])])
        return extract_text(response.content)

    return f"(Unsupported file type: .{ext})"


# ---------------------------------------------------------------------------
# Node -1: Goal Decomposer — breaks long-term goals into weekly chunks
# ---------------------------------------------------------------------------

def goal_decomposer(state: SchedulerState):
    """
    Queries active long-term goals from the database, calculates weeks
    remaining until each deadline, and uses the LLM to produce a concrete,
    actionable weekly task chunk for each goal.
    """
    import session_store  # local import to avoid circular dependency at module level

    active_goals = session_store.list_goals(status="active")
    if not active_goals:
        print("\n--- GOAL DECOMPOSER --- (no active goals, skipping)")
        return {}

    print(f"\n--- GOAL DECOMPOSER --- ({len(active_goals)} active goal(s))")

    now = _now(state)
    tz_name = state.get("user_timezone", "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    today = datetime.now(tz=tz).date()

    goal_descriptions = []
    for g in active_goals:
        try:
            deadline = datetime.strptime(g["deadline_date"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            deadline = today + timedelta(days=30)

        weeks_left = max(1, (deadline - today).days // 7)
        hours_remaining = max(0, g["total_hours_estimated"] - g["hours_completed"])
        hours_per_week = round(hours_remaining / weeks_left, 1) if weeks_left > 0 else hours_remaining

        goal_descriptions.append(
            f"- Goal: \"{g['goal_name']}\"\n"
            f"  Deadline: {g['deadline_date']} ({weeks_left} weeks left)\n"
            f"  Total effort: {g['total_hours_estimated']}h, completed: {g['hours_completed']}h, "
            f"remaining: {hours_remaining}h\n"
            f"  Recommended this week: ~{hours_per_week}h\n"
            f"  Database ID: {g['id']}"
        )

    goals_text = "\n".join(goal_descriptions)

    prompt = f"""You are a strategic goal planner. The current date is: {now}

The user has these active long-term goals:
{goals_text}

For EACH goal, determine what specific, actionable chunk of work MUST be completed THIS WEEK to stay on track.

======================================================================
STRICT GOAL SEPARATION (CRITICAL)
======================================================================
Each goal is INDEPENDENT. You must produce DISTINCT, explicitly named tasks
for EACH goal. NEVER merge work from different goals into one task.

NAMING RULES:
- Every generated task name MUST clearly identify which goal it belongs to.
- A reading goal MUST produce tasks named like:
    "Read [Exact Book Name] — pages 1-25" or "Read [Book Name] (25 pages)"
- A driving license goal MUST produce tasks named like:
    "Study DRPCIV driving theory — Chapter 3" or "Practice mock driving tests"
- An internship goal MUST produce tasks named like:
    "Build REST API endpoint for To-Do app" or "Write cover letter for Company X"

BAD EXAMPLES (FORBIDDEN):
  - "Study for 2 hours" — which goal? Ambiguous. REJECTED.
  - "Read and study" — merges two goals. REJECTED.
  - "Work on goals" — tells the scheduler nothing. REJECTED.

GOOD EXAMPLES:
  - "Read Brothers Karamazov — pages 50-100 (50 pages)" — clearly for a reading goal
  - "Study DRPCIV driving theory manual — Chapters 6-8" — clearly for driving license
  - "Implement GET/DELETE endpoints for To-Do REST API" — clearly for an internship project

SEMANTIC GUARDRAIL — DO NOT CONFUSE GOAL THEMES:
  - "Studying a driving theory manual" is NOT the same as "reading a novel".
    A driving license goal produces tasks about traffic rules, road signs, mock tests.
    A reading goal produces tasks about reading pages from the specific book.
  - "Writing a cover letter" is NOT the same as "reading a book".
  - Each goal has a distinct DOMAIN. Keep tasks within their domain. If two goals
    sound vaguely similar (e.g. both involve "reading"), look at WHAT is being read:
    a driving manual ≠ a novel ≠ a textbook.
======================================================================

Be concrete — not "work on thesis" but "Write Introduction section draft (2000 words)" or "Complete experiment data analysis for Chapter 3".

Return ONLY a valid JSON list — no markdown, no code fences:
[{{"task": "Specific weekly chunk description", "duration": 120, "priority": 2, "goal_db_id": 1, "goal_name": "Original goal name", "preferred_day": null, "is_recurring": false, "recurrence_days": [], "date_deadline": null}}]

Rules:
- duration is in minutes — split large chunks into multiple tasks if > 180min
- priority should be 1-3 (these are important long-term goals)
- goal_db_id must match the Database ID listed above
- goal_name must be the EXACT goal name as listed above — do NOT paraphrase it
- If a goal needs ~6h this week, split into 2-3 focused sessions across different days
- Each session's task name must make it OBVIOUS which goal it serves"""

    response = _invoke_with_backoff(llm, [HumanMessage(content=prompt)])

    try:
        raw_text = extract_text(response.content)
        cleaned = raw_text.strip().strip("`").removeprefix("json").strip()
        goal_chunks = json.loads(cleaned)
        if not isinstance(goal_chunks, list):
            raise ValueError("Expected a list")
    except (json.JSONDecodeError, AttributeError, ValueError):
        try:
            start = raw_text.index("[")
            end = raw_text.rindex("]") + 1
            goal_chunks = json.loads(raw_text[start:end])
        except (ValueError, json.JSONDecodeError):
            objs = _find_all_json_objects(raw_text)
            goal_chunks = [o for o in objs if "goal_db_id" in o]
            if not goal_chunks:
                print("  Warning: could not parse goal chunks. Skipping.")
                goal_chunks = []

    print(f"  Generated {len(goal_chunks)} weekly goal chunk(s).")
    return {
        "goal_chunks": goal_chunks,
        "messages": [HumanMessage(
            content=f"[Goal Decomposer] Generated {len(goal_chunks)} weekly tasks from "
                    f"{len(active_goals)} long-term goal(s):\n"
                    f"{json.dumps(goal_chunks, indent=2)}"
        )],
    }


# ---------------------------------------------------------------------------
# Node 0: Document Processor — merges uploaded files + template + goal chunks
# ---------------------------------------------------------------------------

def document_processor(state: SchedulerState):
    """
    Combines text extracted from uploaded documents, template tasks, and
    goal chunks with the user's typed tasks so the Task Ingester has
    a complete picture.
    """
    uploaded_text = state.get("uploaded_files_text", "")
    raw_tasks = state.get("raw_tasks", "")
    template_tasks = state.get("template_tasks", [])
    goal_chunks = state.get("goal_chunks", [])

    parts = [raw_tasks]

    if uploaded_text.strip():
        print("\n--- DOCUMENT PROCESSOR ---")
        print(f"  Merging uploaded document text with user tasks.")
        parts.append(
            f"Additional tasks extracted from uploaded documents:\n{uploaded_text}"
        )

    if template_tasks:
        print(f"  Merging {len(template_tasks)} template tasks (fixed recurring schedule).")
        template_lines = []
        for t in template_tasks:
            line = t.get("task", "Untitled")
            if t.get("preferred_day"):
                line += f" on {t['preferred_day']}"
            if t.get("deadline"):
                line += f" at {t['deadline']}"
            if t.get("location"):
                line += f" at {t['location']}"
            if t.get("is_recurring") and t.get("recurrence_days"):
                line += f" (recurring: {', '.join(t['recurrence_days'])})"
            if t.get("duration"):
                line += f" [{t['duration']}min]"
            template_lines.append(f"  - {line}")
        parts.append(
            "FIXED RECURRING SCHEDULE (from saved template — these are non-negotiable, "
            "schedule them at their exact times):\n" + "\n".join(template_lines)
        )

    if goal_chunks:
        print(f"  Merging {len(goal_chunks)} goal-derived weekly tasks.")
        chunk_lines = []
        for c in goal_chunks:
            line = f"  - {c.get('task', 'Goal task')} [{c.get('duration', 60)}min, P{c.get('priority', 2)}]"
            if c.get("goal_name"):
                line += f" (for goal: {c['goal_name']})"
            chunk_lines.append(line)
        parts.append(
            "WEEKLY GOAL TASKS (AI-generated from long-term goals — treat as high priority):\n"
            + "\n".join(chunk_lines)
        )

    combined = "\n\n".join(parts)
    return {"raw_tasks": combined}


# ---------------------------------------------------------------------------
# Tool definitions — bound to the LLM so the Scheduler can call them
# ---------------------------------------------------------------------------

@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location to help plan outdoor activities."""
    try:
        results = []
        for query in [location, location.split(",")[0].strip()]:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(query)}&count=1"
            geo_resp = requests.get(geo_url, timeout=10)
            geo_resp.raise_for_status()
            results = geo_resp.json().get("results", [])
            if results:
                break

        if not results:
            return f"Could not find coordinates for '{location}'."

        lat = results[0]["latitude"]
        lng = results[0]["longitude"]
        resolved_name = results[0].get("name", location)
        country = results[0].get("country", "")

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lng}"
            f"&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m"
            f"&temperature_unit=celsius"
        )
        resp = requests.get(weather_url, timeout=10)
        resp.raise_for_status()
        current = resp.json().get("current", {})

        temp = current.get("temperature_2m", "?")
        feels_like = current.get("apparent_temperature", "?")
        humidity = current.get("relative_humidity_2m", "?")
        wind = current.get("wind_speed_10m", "?")
        code = current.get("weather_code", -1)

        wmo_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
        }
        description = wmo_descriptions.get(code, f"Unknown (code {code})")

        result = (
            f"Weather in {resolved_name}, {country}: {description}, "
            f"{temp}°C (feels like {feels_like}°C), "
            f"humidity {humidity}%, wind {wind} km/h"
        )
        print(f"    [Tool] get_weather({location}) -> {result}")
        return result

    except Exception as e:
        error_msg = f"Weather lookup failed for '{location}': {e}"
        print(f"    [Tool] get_weather -> ERROR: {error_msg}")
        return error_msg


@tool
def get_weekly_forecast(location: str) -> str:
    """Get a 7-day weather forecast for a location to help plan the week ahead.
    Returns daily high/low temperatures and conditions for each day."""
    try:
        results = []
        for query in [location, location.split(",")[0].strip()]:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(query)}&count=1"
            geo_resp = requests.get(geo_url, timeout=10)
            geo_resp.raise_for_status()
            results = geo_resp.json().get("results", [])
            if results:
                break

        if not results:
            return f"Could not find coordinates for '{location}'."

        lat = results[0]["latitude"]
        lng = results[0]["longitude"]
        resolved_name = results[0].get("name", location)

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lng}"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max"
            f"&temperature_unit=celsius&forecast_days=7"
        )
        resp = requests.get(weather_url, timeout=10)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})

        wmo_descriptions = {
            0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
            55: "Dense drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow",
            80: "Light showers", 81: "Showers", 82: "Heavy showers",
            95: "Thunderstorm", 96: "T-storm + hail", 99: "Severe T-storm",
        }

        dates = daily.get("time", [])
        lines = [f"7-day forecast for {resolved_name}:"]
        for i, date_str in enumerate(dates):
            d = datetime.strptime(date_str, "%Y-%m-%d")
            day_name = d.strftime("%A")
            code = daily.get("weather_code", [0])[i] if i < len(daily.get("weather_code", [])) else 0
            t_max = daily.get("temperature_2m_max", [0])[i] if i < len(daily.get("temperature_2m_max", [])) else "?"
            t_min = daily.get("temperature_2m_min", [0])[i] if i < len(daily.get("temperature_2m_min", [])) else "?"
            precip = daily.get("precipitation_probability_max", [0])[i] if i < len(daily.get("precipitation_probability_max", [])) else 0
            wind = daily.get("wind_speed_10m_max", [0])[i] if i < len(daily.get("wind_speed_10m_max", [])) else "?"
            desc = wmo_descriptions.get(code, f"Code {code}")
            lines.append(
                f"  {day_name} ({date_str}): {desc}, {t_min}°C-{t_max}°C, "
                f"rain {precip}%, wind {wind} km/h"
            )

        result = "\n".join(lines)
        print(f"    [Tool] get_weekly_forecast({location}) -> {len(dates)} days")
        return result

    except Exception as e:
        error_msg = f"Weekly forecast failed for '{location}': {e}"
        print(f"    [Tool] get_weekly_forecast -> ERROR: {error_msg}")
        return error_msg


@tool
def estimate_commute(origin: str, destination: str) -> str:
    """Estimate travel/commute time between two locations using Google Maps Routes API.
    Returns the shortest route duration and distance."""
    try:
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_API_KEY,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.description",
        }
        body = {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": "DRIVE",
            "computeAlternativeRoutes": True,
            "routingPreference": "TRAFFIC_AWARE",
        }

        resp = requests.post(url, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        routes = data.get("routes", [])
        if not routes:
            result = f"Could not find a driving route from {origin} to {destination}."
            print(f"    [Tool] estimate_commute -> {result}")
            return result

        def parse_duration_seconds(route):
            dur_str = route.get("duration", "0s")
            return int(dur_str.rstrip("s"))

        best = min(routes, key=parse_duration_seconds)
        total_seconds = parse_duration_seconds(best)
        minutes = total_seconds // 60
        distance_m = best.get("distanceMeters", 0)
        distance_km = round(distance_m / 1000, 1)
        route_name = best.get("description", "main road")

        result = (
            f"Shortest route from {origin} to {destination}: "
            f"{minutes} min ({distance_km} km) via {route_name}"
        )
        print(f"    [Tool] estimate_commute -> {result}")
        return result

    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get("error", {}).get("message", str(e))
        except Exception:
            error_detail = str(e)
        error_msg = f"Google Routes API error: {error_detail}"
        print(f"    [Tool] estimate_commute -> ERROR: {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Commute estimation failed: {e}"
        print(f"    [Tool] estimate_commute -> ERROR: {error_msg}")
        return error_msg


# Bundle tools for binding and for the ToolNode
tools = [get_weather, get_weekly_forecast, estimate_commute]
llm_with_tools = llm.bind_tools(tools)

# Pre-built ToolNode executes whichever tools the LLM requested
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# Node 1: Task Ingester (weekly-aware)
# ---------------------------------------------------------------------------

def task_ingester(state: SchedulerState):
    """
    Parses the user's raw, messy input into structured tasks with weekly context.
    Assigns preferred days, detects recurring tasks, and extracts date deadlines.
    """
    print("\n--- INGESTING TASKS (WEEKLY) ---")
    raw_tasks = state.get("raw_tasks", "No tasks provided.")
    now = _now(state)
    week = _week_dates(state)
    user_location = state.get("user_location", "Home")

    week_display = "\n".join(f"  - {d['display']} ({d['date_str']})" for d in week)

    prompt = f"""You are an expert task analyzer for WEEKLY planning. The current date and time is: {now}
The user is currently located at: {user_location}

THE PLANNING WEEK (7 days starting today):
{week_display}

=== RAW INPUT (user tasks + uploaded documents) ===
{raw_tasks}
=== END RAW INPUT ===

CRITICAL RULES — VIOLATIONS WILL CAUSE ERRORS:
1. You MUST create one JSON entry for EVERY SINGLE task/event/class/appointment in the input above.
2. PRESERVE EXACT NAMES. If the input says "Calculus II lecture", the task field must be "Calculus II lecture" — NOT "University class" or "Math" or any summary.
3. NEVER merge multiple tasks into one. "Gym" and "Grocery shopping" = 2 separate entries.
4. NEVER skip tasks. If the input has 15 items, your output must have at least 15 entries (more if recurring tasks expand to multiple days).
5. Recurring tasks (e.g. "gym Mon/Wed/Fri", a class that meets Tue/Thu) must set is_recurring=true and list ALL day names in recurrence_days.
6. If a task from an uploaded document/photo specifies a day and time, preserve that EXACTLY in preferred_day and deadline.

For each task:
- Estimate a reasonable duration in minutes.
- Assign a priority from 1 to 10 (1 = most urgent/important, 10 = least).
- Note any implied deadlines relative to the current time.
- Note any locations mentioned (useful for travel/commute planning).
- Determine if the task specifies or implies a particular day of the week. If so, set "preferred_day" to that day name (e.g. "Monday"). If flexible, set to null.
- Determine if this is a recurring task (e.g. "gym every Mon/Wed/Fri", "daily standup"). If so, set "is_recurring" to true and list the day names in "recurrence_days". Otherwise false and [].
- If the task has a hard date deadline (e.g. "submit report by Thursday"), set "date_deadline" to the YYYY-MM-DD date. Otherwise null.

LOCATION DISAMBIGUATION:
If a task mentions a place name without a full address or city, assume it is
near the user's current location ({user_location}) and append the city/area.
For example, if the user is in Bucuresti and says "Mega Image at Alba Iulia",
output "Mega Image, Piata Alba Iulia, Bucuresti" — NOT the city of Alba Iulia
in Transylvania. Always resolve ambiguous locations to the nearest local match.

Return ONLY a valid JSON list of objects — no markdown, no code fences, no explanation:
[{{"task": "EXACT name from input", "duration": 60, "priority": 1, "deadline": "HH:MM or null", "location": "full disambiguated address or null", "preferred_day": "Monday or null", "is_recurring": false, "recurrence_days": [], "date_deadline": "YYYY-MM-DD or null", "goal_id": null}}]"""

    # Append macro-goal tagging instructions if the user set goals
    macro_goals = state.get("macro_goals", [])
    if macro_goals:
        slug_map = _build_goal_slug_map(macro_goals)
        goals_with_slugs = "\n".join(f"  \"{slug}\" → \"{gname}\"" for slug, gname in slug_map.items())
        prompt += f"""

MACRO-GOAL TAGGING (CRITICAL — goal progress tracking depends on this):
The user has set these weekly goals with string IDs:
{goals_with_slugs}

For EACH task in your output, determine if it contributes to one of these goals.

RULES:
- Set "goal_id" to the EXACT string slug (e.g. "goal_read_brothers_karamazov") if the task DIRECTLY serves that goal.
- If a task does NOT clearly serve any goal, set "goal_id" to null.
- Read the goal name carefully and match by THEME, not by surface words.
- A task about reading a specific book matches a goal about reading that book.
- A task about driving theory matches a goal about getting a driving license.
- University classes, meals, breaks, commutes → always null.
- Tasks generated from long-term goals (marked "for goal: ...") MUST get the
  matching goal_id slug. Match by the goal name, not by unrelated goals.

EXAMPLE:
  Goals: ["goal_read_brothers_karamazov" → "Read 100 pages from Brothers Karamazov",
          "goal_driving_license" → "Get Driving License"]
  "Read Brothers Karamazov — pages 1-25" → goal_id: "goal_read_brothers_karamazov"
  "Study DRPCIV driving theory Chapter 3"  → goal_id: "goal_driving_license"
  "Calculus II lecture"                    → goal_id: null (university class)
  "Go to DMV"                             → goal_id: "goal_driving_license"
  "Buy groceries"                         → goal_id: null (unrelated)"""

    response = _invoke_with_backoff(llm, [HumanMessage(content=prompt)])

    try:
        raw_text = extract_text(response.content)
        cleaned = raw_text.strip().strip("`").removeprefix("json").strip()
        parsed_tasks = json.loads(cleaned)
        if not isinstance(parsed_tasks, list):
            raise ValueError("Expected a JSON list")
    except (json.JSONDecodeError, AttributeError, ValueError):
        # Fallback: scan for JSON arrays in mixed output
        try:
            start = raw_text.index("[")
            end = raw_text.rindex("]") + 1
            parsed_tasks = json.loads(raw_text[start:end])
        except (ValueError, json.JSONDecodeError):
            # Last resort: look for JSON objects and wrap in list
            objs = _find_all_json_objects(raw_text)
            task_objs = [o for o in objs if "task" in o]
            if task_objs:
                parsed_tasks = task_objs
            else:
                print(f"  Warning: could not parse task JSON. Using fallback.")
                parsed_tasks = [
                    {"task": raw_tasks, "duration": 60, "priority": 5, "deadline": None,
                     "location": None, "preferred_day": None, "is_recurring": False,
                     "recurrence_days": [], "date_deadline": None}
                ]

    # --- Post-process: force-assign goal_id slugs ---
    # The LLM often fails to tag tasks with the correct goal_id even when told.
    # This programmatic step matches parsed tasks back to:
    #   1. goal_chunks (from goal_decomposer) using goal_name keywords
    #   2. macro_goals (weekly goals) using fuzzy keyword overlap
    goal_chunks = state.get("goal_chunks", [])
    macro_goals = state.get("macro_goals", [])
    slug_map = _build_goal_slug_map(macro_goals) if macro_goals else {}

    # Build keyword index from goal_chunks (long-term goal tasks)
    chunk_keywords: list[tuple[str, set[str]]] = []
    for c in goal_chunks:
        gname = c.get("goal_name", "")
        if gname:
            slug = _goal_slug(gname)
            kw = _extract_keywords(gname)
            task_kw = _extract_keywords(c.get("task", ""))
            combined_kw = kw | task_kw
            if combined_kw:
                chunk_keywords.append((slug, combined_kw))

    # Build keyword index from macro_goals (weekly goals)
    macro_keywords: list[tuple[str, set[str]]] = []
    for slug, gname in slug_map.items():
        kw = _extract_keywords(gname)
        if kw:
            macro_keywords.append((slug, kw))

    tagged = 0
    for task in parsed_tasks:
        if task.get("goal_id") is not None:
            continue  # Already tagged by LLM, trust it (enforce_goal_ids will verify later)
        title_kw = _extract_keywords(task.get("task", ""))
        if not title_kw:
            continue

        best_slug = None
        best_score = 0

        # Match against goal_chunks first (more specific)
        for slug, kw in chunk_keywords:
            score = _fuzzy_keyword_overlap(title_kw, kw)
            if score >= 1 and score > best_score:
                best_score = score
                best_slug = slug

        # Then try macro_goals
        if best_slug is None:
            for slug, kw in macro_keywords:
                score = _fuzzy_keyword_overlap(title_kw, kw)
                if score >= 1 and score > best_score:
                    best_score = score
                    best_slug = slug

        if best_slug is not None:
            task["goal_id"] = best_slug
            tagged += 1

    if tagged:
        print(f"  [Goal Tagging] Post-tagged {tagged} task(s) with goal_id slugs.")

    summary_msg = HumanMessage(
        content=f"[Task Ingester] Parsed {len(parsed_tasks)} tasks at {now}:\n"
                f"{json.dumps(parsed_tasks, indent=2)}"
    )

    print(f"  Parsed {len(parsed_tasks)} task(s).")
    return {
        "parsed_tasks": parsed_tasks,
        "revision_count": 0,
        "messages": [summary_msg],
    }


# ---------------------------------------------------------------------------
# Node 2: Scheduler (weekly tool-calling agent)
# ---------------------------------------------------------------------------

def _build_scheduler_prompt(state: SchedulerState) -> tuple[str, list[dict], int]:
    """
    Build the scheduler system prompt from the current state.
    Returns (prompt_text, parsed_tasks, total_task_slots).
    Extracted so both the tool-calling and structured-output phases share
    the exact same instructions.
    """
    parsed_tasks = state.get("parsed_tasks", [])
    critique = state.get("critique", "")
    user_location = state.get("user_location", "Home")
    now = _now(state)
    week = _week_dates(state)

    critique_section = critique if critique else "None — this is the first draft."
    week_display = "\n".join(f"  - {d['display']} ({d['date_str']})" for d in week)

    # Build explicit task checklist so the LLM can't miss any
    task_checklist = []
    total_task_slots = 0
    for i, t in enumerate(parsed_tasks, 1):
        name = t.get("task", "Untitled")
        pday = t.get("preferred_day")
        recurring = t.get("is_recurring", False)
        rec_days = t.get("recurrence_days", [])
        deadline = t.get("date_deadline")
        dur = t.get("duration", 60)

        day_info = ""
        if recurring and rec_days:
            day_info = f" [RECURRING: {', '.join(rec_days)}]"
            total_task_slots += len(rec_days)
        elif pday:
            day_info = f" [preferred: {pday}]"
            total_task_slots += 1
        else:
            total_task_slots += 1

        dl_info = f" [DEADLINE: {deadline}]" if deadline else ""
        task_checklist.append(f"  {i}. {name} ({dur}min){day_info}{dl_info}")

    task_checklist_str = "\n".join(task_checklist)

    # Build explicit goal_id lookup from parsed_tasks so the LLM has
    # a clear, authoritative table mapping task names to their goal IDs.
    goal_id_table_lines = []
    macro_goals = state.get("macro_goals", [])

    # Available goals header for the LLM — using string slugs
    if macro_goals:
        slug_map = _build_goal_slug_map(macro_goals)
        goals_list = ", ".join(f'"{slug}": "{gname}"' for slug, gname in slug_map.items())
        available_goals = f"Available Goals: {{{goals_list}}}"
    else:
        available_goals = "Available Goals: {} (none set)"

    # Reuse slug_map from above (or empty dict) for goal label lookups
    if not macro_goals:
        slug_map = {}

    for t in parsed_tasks:
        gid = t.get("goal_id")
        if gid is not None:
            goal_label = ""
            if str(gid) in slug_map:
                goal_label = f" (goal: \"{slug_map[str(gid)]}\")"
            goal_id_table_lines.append(f"  - \"{t.get('task', '?')}\" -> goal_id: \"{gid}\"{goal_label}")
    goal_id_table = "\n".join(goal_id_table_lines) if goal_id_table_lines else "  (no tasks have goal_id set)"

    prompt = f"""You are an expert productivity scheduler. Your job is to create a DETAILED, COMPLETE 7-day weekly schedule.

Current time: {now}
User location: {user_location}

PLANNING WEEK:
{week_display}

######################################################################
#                                                                    #
#  ABSOLUTE COMPLETENESS — THE #1 RULE                               #
#                                                                    #
#  You are given {len(parsed_tasks)} tasks below. Your output MUST   #
#  contain a scheduled entry for EVERY SINGLE ONE of them.           #
#  No exceptions. No shortcuts. No summarizing. No grouping.         #
#                                                                    #
#  MINIMUM OUTPUT SIZE: Your output must contain AT LEAST            #
#  {total_task_slots} task entries (not counting breaks/travel).     #
#  If you produce fewer, the schedule is REJECTED.                   #
#                                                                    #
######################################################################

======================================================================
TASK CHECKLIST — {len(parsed_tasks)} tasks, {total_task_slots} slots
You MUST schedule EVERY SINGLE ONE. Check them off as you go.
======================================================================
{task_checklist_str}
======================================================================

THE 1:1 RULE (CRITICAL):
- You have {len(parsed_tasks)} unique tasks and {total_task_slots} required slots.
- Your final JSON MUST contain at least {total_task_slots} real task entries
  (type != "break" and type != "travel").
- Each task in the checklist MUST appear at least once. Recurring tasks
  MUST appear once per recurrence day (e.g. 3 recurrence_days = 3 entries).
- DO NOT summarize multiple tasks into one entry.
- DO NOT skip "obvious" or "implicit" tasks — EVERY task must be an
  explicit entry in your JSON output, including fixed university classes,
  template tasks, and routine errands.
- DO NOT group distinct tasks together (e.g. "Study + Read" is INVALID
  if they are separate items in the checklist).

FULL TASK DATA:
{json.dumps(parsed_tasks, indent=2)}

PREVIOUS CRITIQUE TO ADDRESS:
{critique_section}

CRITICAL RULES — FAILURE TO FOLLOW = INVALID SCHEDULE:
1. EVERY task in the checklist above MUST appear in your output. Count them.
   After generating, go through the checklist one by one and verify each
   task name appears. If ANY is missing, add it before outputting.
2. Use the EXACT task name from the checklist — do NOT rename, summarize, or merge tasks.
3. Recurring tasks MUST appear on EVERY one of their recurrence_days as separate entries.
4. Fixed/Template tasks (university classes, etc.) MUST be explicitly written
   out in the JSON at their exact day and time. They are NOT "implicit" —
   if they don't appear in your output, they are MISSING.
5. If a task has a date_deadline, schedule it on or before that date.
6. If a task has a preferred_day, schedule it on that day.

SCHEDULING HIERARCHY (strict precedence):
1. TEMPLATE/FIXED tasks (from recurring schedule like university classes) — these have EXACT days and times. NEVER move them. Schedule other tasks AROUND them. You MUST include them in the output.
2. Hard deadlines (date_deadline) and fixed appointments — ABSOLUTE, never violate.
3. User critique instructions (e.g. "Move gym to Wednesday") — override other flexible logic.
4. Goal-derived weekly tasks (from long-term goals) — high priority, schedule early in available gaps.
5. preferred_day assignments — respect when possible.
6. Recurring tasks — must appear on ALL their recurrence_days.
7. Priority scale (1=highest, 10=lowest) — only for flexible time ordering.
8. Do NOT arrive at locations more than 15 minutes early.

WEEKLY PLANNING RULES:
1. TODAY ({week[0]['display']}): Start from NOW ({now}). Do NOT use a generic morning start.
2. FUTURE DAYS: Plan from 08:00 to 23:00. No tasks before 07:00 or after 23:30.
3. Include 5-10 minute breaks between tasks.
4. ONLY schedule the tasks from the checklist — do NOT invent new tasks.
5. Do NOT create oversized buffer blocks.
6. Group location-based tasks on the same day to minimize commute.
7. Place high-priority tasks earlier in the week.

======================================================================
PHYSICAL CONSTRAINT RULES (YOU CANNOT BREAK PHYSICS)
======================================================================

THE COMMUTE RULE (BACK-TO-BACK CLASSES):
If two consecutive fixed tasks (e.g. university classes) end and start at the
same time but are in DIFFERENT locations, you CANNOT teleport the user.
You MUST do one of:
  a) Insert a "Walk/Transition" travel entry for the last 10 minutes of the
     first block. Example: Class A is 08:00-10:00 in Room 101, Class B is
     10:00-12:00 in Lab B2 → schedule Class A as 08:00-09:50, then
     "Walk to Lab B2" as 09:50-10:00, then Class B as 10:00-12:00.
  b) If the locations are far apart and need more than 10 min, use
     estimate_commute to get the real travel time and adjust accordingly.
Do NOT leave overlapping or zero-gap entries at different locations.

THE 90-MINUTE RULE:
You are FORBIDDEN from scheduling any continuous work or study block
longer than 90 minutes without a break. After every 90 minutes of focused
work, you MUST insert a minimum 10-minute "Break" entry.
  - If a fixed university lab is 4 hours (240 min), split it:
    Lab 08:00-09:30, Break 09:30-09:40, Lab 09:40-11:10, Break 11:10-11:20,
    Lab 11:20-12:00.
  - If a study session is 120 min, split into 90 + break + 30.
This rule applies to ALL work/study blocks, including template tasks.

WORKLOAD BALANCING:
You are FORBIDDEN from cramming all floating (non-fixed, non-deadline) tasks
into one or two days (e.g. Sunday). You must:
  - Scan ALL 7 days for empty afternoons, evenings, and gaps between classes.
  - Distribute Priority 2-3 floating tasks EVENLY across the week.
  - If Friday afternoon is empty and Sunday is overloaded, move tasks to Friday.
  - Aim for no more than ~6 hours of non-fixed work on any single day.
  - Only stack tasks on one day if there is literally no other available slot.
======================================================================

======================================================================
GOAL ID ASSIGNMENT — STRICT RULES (READ CAREFULLY)
======================================================================
{available_goals}

STRICT GOAL ID MAPPING:
You MUST read the EXACT NAME of each goal before attaching its string slug ID.
  - A task named "Study DRPCIV driving theory" is about DRIVING, not about
    "Read a book". If the Available Goals include "goal_read_brothers_karamazov"
    and "goal_driving_license", this task gets goal_id: "goal_driving_license",
    NOT "goal_read_brothers_karamazov".
  - A task named "Read Brothers Karamazov (25 pages)" is about READING.
    It gets goal_id: "goal_read_brothers_karamazov", NOT "goal_driving_license".
  - If a task does NOT perfectly match a goal's theme, goal_id MUST be null.
  - University classes, commutes, meals, breaks → ALWAYS goal_id: null.
  - goal_id values are ALWAYS strings (e.g. "goal_read_brothers_karamazov") or null.
    NEVER use integers for goal_id.

Ask yourself for EVERY task: "Does the task title describe work that directly
advances THIS SPECIFIC goal?" If no → null. If unsure → null.

The following is the AUTHORITATIVE mapping of task names to goal IDs.
This was determined during task ingestion. You MUST follow it exactly.

GOAL ID LOOKUP TABLE:
{goal_id_table}

MANDATORY RULES:
1. EXACT COPY: When you output a scheduled entry for a task, you MUST copy
   its goal_id EXACTLY from the lookup table above. Do NOT guess, infer, or
   reassign goal IDs based on your own interpretation.
2. NULL BY DEFAULT: If a task name does NOT appear in the lookup table above,
   its goal_id MUST be null. Period. No exceptions.
3. NO CROSS-CONTAMINATION: The following entry types MUST ALWAYS have
   goal_id: null, regardless of context:
   - type "break" (coffee breaks, rest breaks, etc.)
   - type "travel" (commute, driving, walking between locations)
   - type "meal" (lunch, dinner, snacks)
   - type "shower" (freshen up, getting ready)
   Any task that is NOT directly doing work toward the goal gets null.
4. CHUNKING: If you split a goal-tagged task into multiple sessions across
   different days (e.g. "Read 100 pages" becomes 4x "Read 25 pages"),
   EVERY chunk MUST carry the SAME goal_id as the original task.
   The frontend progress bar depends on this — missing a goal_id breaks tracking.
5. VERIFICATION: Before outputting, scan every entry in your JSON:
   - For each entry with a non-null goal_id: confirm the title matches a task
     from the lookup table that has that exact goal_id.
   - For each entry with goal_id: null: confirm it is NOT in the lookup table
     with a non-null goal_id.
======================================================================

FINAL SELF-CHECK (do this BEFORE outputting):
1. Count your real task entries (exclude breaks/travel). Is it >= {total_task_slots}? If NO, you dropped tasks — find and add them.
2. Go through the TASK CHECKLIST above one by one. For each task, find it in your output. If it's missing, add it.
3. Verify every goal_id matches the lookup table — read the goal NAME, not just the number.
4. Check for back-to-back entries at different locations — insert travel/transition blocks.
5. Check no work block exceeds 90 minutes without a break.
6. Check floating tasks are spread across the week, not crammed into one day.

Include ALL 7 days: {', '.join(d['day_name'] for d in week)}.
Empty days get an empty array [].
type: work, break, travel, fitness, call, errand, meal, shower.
priority: 1-10 for tasks, null for breaks/travel.
goal_id: MUST match the lookup table above. If not in table, MUST be null."""

    return prompt, parsed_tasks, total_task_slots


def _finalize_schedule(raw_structured: dict, parsed_tasks: list[dict],
                       macro_goals: list[str] | None = None) -> tuple[dict, str]:
    """
    Pipeline: Pydantic validation → programmatic goal_id enforcement → text.
    Always runs as the final step regardless of how the schedule was generated.
    """
    validated = _validate_with_pydantic(raw_structured)
    enforced = _enforce_goal_ids(validated, parsed_tasks, macro_goals)
    text = _schedule_to_text(enforced)
    return enforced, text


def _extract_tool_context(messages: list) -> str:
    """
    Extract a summary of tool call results from the conversation history
    so the structured-output LLM has the weather/commute data without
    needing bind_tools itself.
    """
    tool_results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results.append(f"[Tool result: {msg.name}] {msg.content}")
    if not tool_results:
        return "No tool data available."
    return "\n".join(tool_results)


def scheduler(state: SchedulerState):
    """
    Drafts a 7-day chronological schedule.

    Architecture — generate-then-validate:

    Phase 1 (tool-calling): Uses llm_with_tools to fetch weather/commute data.
        The graph loops scheduler → tools → scheduler until no more tool calls.

    Phase 2 (generation): Uses llm_with_tools (NOT llm_structured) to produce
        the full schedule as raw JSON. This avoids the output truncation that
        with_structured_output() causes on large schedules (70+ entries).

    Phase 3 (validation): Parses the raw JSON, validates every entry through
        the ScheduledTask Pydantic model, then programmatically enforces
        correct goal_id values from the authoritative parsed_tasks mapping.
        The LLM's goal_id output is NEVER trusted as-is.
    """
    print("\n--- DRAFTING WEEKLY SCHEDULE ---")
    messages = state.get("messages", [])
    revision_count = state.get("revision_count", 0)
    parsed_tasks = state.get("parsed_tasks", [])
    macro_goals = state.get("macro_goals", [])

    # ---- Continuation after tool execution ----
    if messages and isinstance(messages[-1], ToolMessage):
        # Check if the LLM wants to call more tools
        response = _invoke_with_backoff(llm_with_tools, messages)
        has_more_tool_calls = bool(getattr(response, "tool_calls", None))
        if has_more_tool_calls:
            return {"messages": [response]}

        # --- Phase 2 + 3: Tools done → parse, validate, enforce ---
        print("  Tools complete — parsing schedule response...")
        raw_structured, _ = _parse_schedule_response(response.content)
        structured, text = _finalize_schedule(raw_structured, parsed_tasks, macro_goals)
        return {
            "messages": [response],
            "draft_schedule": text,
            "structured_schedule": structured,
        }

    # ---- Fresh scheduling attempt ----
    base_prompt, parsed_tasks, total_task_slots = _build_scheduler_prompt(state)
    user_location = state.get("user_location", "Home")

    # Full prompt: base rules + tool instructions + output format
    tool_instructions = f"""

MANDATORY TOOL USAGE:
Before generating the schedule, call these tools:
- get_weekly_forecast(location="{user_location}") for 7-day weather
- estimate_commute(origin="{user_location}", destination=<location>) for EACH task with a location
- get_weather(location="{user_location}") for today's outdoor/physical tasks
Call tools FIRST, then build the schedule using their results.

OUTPUT FORMAT — Return ONLY valid JSON, no markdown fences, no text before/after:
{{"Monday": [{{"start":"HH:MM","end":"HH:MM","title":"EXACT Task Name","type":"work|break|travel|fitness|call|errand|meal|shower","priority":1,"duration_min":60,"location":"place or null","notes":"context or null","weather":"brief note or null","commute":"travel info or null","goal_id":null}}], "Tuesday": [...], ...}}"""

    full_prompt = base_prompt + tool_instructions

    human_msg = HumanMessage(content=full_prompt)
    response = _invoke_with_backoff(llm_with_tools, messages + [human_msg])
    has_tool_calls = bool(getattr(response, "tool_calls", None))

    if has_tool_calls:
        # Phase 1: LLM wants tools — route to ToolNode, will come back here
        return {
            "messages": [human_msg, response],
            "revision_count": revision_count + 1,
        }

    # Phase 2 + 3: No tools → parse, validate, enforce
    raw_structured, _ = _parse_schedule_response(response.content)
    structured, text = _finalize_schedule(raw_structured, parsed_tasks, macro_goals)
    return {
        "messages": [human_msg, response],
        "draft_schedule": text,
        "structured_schedule": structured,
        "revision_count": revision_count + 1,
    }


# ---------------------------------------------------------------------------
# Node 3: Critic (weekly-aware)
# ---------------------------------------------------------------------------

def critic(state: SchedulerState):
    """
    Evaluates the weekly draft schedule holistically — checks coverage,
    balance, recurring tasks, deadlines, and workload distribution.
    """
    print("\n--- CRITIQUING WEEKLY SCHEDULE ---")
    draft_schedule = state.get("draft_schedule", "")
    parsed_tasks = state.get("parsed_tasks", [])
    revision_count = state.get("revision_count", 0)
    now = _now(state)
    week = _week_dates(state)

    week_display = "\n".join(f"  - {d['display']} ({d['date_str']})" for d in week)

    # Build a task name checklist for the critic to verify against
    task_names = []
    for t in parsed_tasks:
        name = t.get("task", "Untitled")
        recurring = t.get("is_recurring", False)
        rec_days = t.get("recurrence_days", [])
        if recurring and rec_days:
            for day in rec_days:
                task_names.append(f"  - {name} (on {day})")
        else:
            task_names.append(f"  - {name}")
    task_names_str = "\n".join(task_names)

    prompt = f"""You are a strict productivity critic evaluating a WEEKLY schedule.
The current time is: {now}

THE PLANNING WEEK:
{week_display}

PROPOSED WEEKLY SCHEDULE:
{draft_schedule}

======================================================================
REQUIRED TASKS — Every one of these MUST appear in the schedule:
{task_names_str}
Total required entries: {len(task_names)}
======================================================================

ORIGINAL TASK DATA:
{json.dumps(parsed_tasks, indent=2)}

EVALUATION STEPS — Do these IN ORDER:
1. COMPLETENESS CHECK (MOST CRITICAL — if this fails, STOP and report):
   Go through the required tasks list above ONE BY ONE. For EACH task,
   scan the entire schedule and verify it appears with its EXACT name.
   - If a task is marked RECURRING, verify it appears on EVERY required day.
   - Template/fixed tasks (university classes, etc.) must be EXPLICITLY present.
     Do NOT assume they are "implicitly" included.
   - List ALL missing tasks by name. If even ONE task is missing, the schedule
     CANNOT be rated PERFECT.
2. TASK COUNT: Count the actual task entries (excluding breaks/travel) in the
   schedule. The minimum is {len(task_names)}. If the count is lower, list
   which tasks are missing. This is an AUTOMATIC FAILURE.
3. GOAL ID ACCURACY: For each scheduled entry, check the goal_id field against the original task data:
   a. If the original task has goal_id="some_slug", every scheduled entry for that task MUST have goal_id="some_slug".
   b. If the original task has goal_id=null, the scheduled entry MUST have goal_id=null.
   c. Breaks, travel, meals, and showers MUST ALWAYS have goal_id=null.
   d. If a task was split into chunks, ALL chunks must share the same goal_id.
   Flag ANY mismatches as critical errors — wrong goal_id breaks the user's progress tracking.
4. BACK-TO-BACK LOCATION CHECK: Find any two consecutive entries that end/start
   at the same time but have DIFFERENT locations. If there is no travel/transition
   entry between them, flag it as "IMPOSSIBLE: [Task A location] → [Task B location]
   with zero travel time". The scheduler must insert a walk/transition block.
5. 90-MINUTE RULE: Find any single work/study block longer than 90 minutes
   without a break inserted. Flag it: "VIOLATION: [Task] runs [X] minutes
   without a break — must split with a 10-min break every 90 min."
6. WORKLOAD BALANCE: Count non-fixed task hours per day. If any single day has
   more than ~6 hours of floating tasks while other days have empty afternoons,
   flag it: "IMBALANCE: [Day] has [X]h of floating work, but [Other Day]
   afternoon is empty — redistribute."
7. GOAL ID SEMANTIC CHECK: For each entry with a non-null goal_id (a string slug
   like "goal_read_brothers_karamazov"), read the task title AND the goal name.
   Does the title ACTUALLY describe work for that goal?
   Example: "Study driving theory" with goal_id "goal_read_brothers_karamazov" is WRONG.
   Flag any mismatch: "WRONG GOAL: [Task] has goal_id [slug] ([Goal Name])
   but does not serve that goal."
8. DEADLINES: Are all date_deadline constraints met?
9. DAY PREFERENCES: Are preferred_day assignments respected?
10. TODAY'S SCHEDULE: Does today start from current time ({now}), not a generic morning?
11. SLEEP BOUNDARIES: No tasks before 07:00 or after 23:30.
12. PRIORITY ORDER: High-priority tasks (1) scheduled earlier within each day?
13. TRAVEL LOGIC: Tasks with locations have commute time? Location-based tasks grouped?
14. BREAKS: Reasonable breaks between tasks (5-10 min)?

This is revision {revision_count} of 7.

If ALL criteria are met (especially completeness — ALL tasks present), reply with EXACTLY: PERFECT
Otherwise, provide a concise bulleted list of specific changes. If tasks are MISSING, list them explicitly."""

    response = _invoke_with_backoff(llm_thinking, [HumanMessage(content=prompt)])
    critique_text = extract_text(response.content).strip()

    critic_log = HumanMessage(
        content=f"[Critic — Revision {revision_count}] {critique_text}"
    )

    verdict = "APPROVED" if "PERFECT" in critique_text.upper() else "NEEDS REVISION"
    print(f"  Verdict: {verdict}")

    return {
        "critique": critique_text,
        "messages": [critic_log],
    }


# ---------------------------------------------------------------------------
# Node 4: Human-in-the-Loop Review
# ---------------------------------------------------------------------------

def human_review(state: SchedulerState):
    """
    Pauses the graph using LangGraph's interrupt() mechanism.
    The user is asked to approve the schedule or provide manual tweaks.
    """
    print("\n--- AWAITING HUMAN REVIEW ---")

    schedule = state.get("draft_schedule", "")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    feedback = interrupt({
        "schedule": schedule,
        "final_critique": critique,
        "revisions_used": revision_count,
        "prompt": "Do you approve this weekly schedule? Type 'approve' or describe your tweaks.",
    })

    feedback_str = str(feedback).strip()

    if feedback_str.lower() in ("approve", "approved", "yes", "y", "ok", "looks good", "lgtm"):
        print("  User approved the schedule.")
        return {
            "human_feedback": "approved",
            "messages": [HumanMessage(content="[Human] Weekly schedule approved.")],
        }
    else:
        print(f"  User requested changes: {feedback_str}")
        return {
            "human_feedback": feedback_str,
            "critique": f"USER REQUESTED CHANGES: {feedback_str}",
            "messages": [HumanMessage(content=f"[Human Feedback] {feedback_str}")],
        }


# ---------------------------------------------------------------------------
# Node 5: Reschedule Agent — dynamic "I'm Behind" rescheduler
# ---------------------------------------------------------------------------

def reschedule_agent(state: RescheduleState):
    """
    Triggered when the user clicks "I'm Behind". Takes remaining unchecked
    tasks for today, compresses the rest of the day, and pushes low-priority
    tasks to tomorrow if needed.
    """
    print("\n--- DYNAMIC RESCHEDULER ---")
    current_time = state.get("current_time", "12:00")
    user_location = state.get("user_location", "Home")
    today_name = state.get("today_name", "Today")
    tomorrow_name = state.get("tomorrow_name", "Tomorrow")
    remaining_tasks = state.get("remaining_tasks", [])
    tomorrow_schedule = state.get("tomorrow_schedule", [])

    if not remaining_tasks:
        print("  No remaining tasks to reschedule.")
        return {
            "rescheduled_today": [],
            "rescheduled_tomorrow": tomorrow_schedule,
            "messages": [HumanMessage(content="[Rescheduler] No tasks remaining.")],
        }

    prompt = f"""You are an emergency schedule optimizer. The user is BEHIND on their tasks.

CURRENT TIME: {current_time}
TODAY: {today_name}
TOMORROW: {tomorrow_name}
LOCATION: {user_location}

REMAINING UNCHECKED TASKS FOR TODAY (not yet started or completed):
{json.dumps(remaining_tasks, indent=2)}

TOMORROW'S CURRENT SCHEDULE:
{json.dumps(tomorrow_schedule, indent=2)}

YOUR MISSION — Rebuild the rest of today starting from {current_time}:

RULES (strict priority order):
1. Priority 1-3 tasks MUST stay today. These are non-negotiable.
2. Compress or remove ALL breaks to 5 minutes max.
3. Remove any buffer/padding time.
4. If tasks still don't fit before 23:30:
   a. Push priority 7-10 tasks to tomorrow.
   b. Push priority 4-6 tasks to tomorrow only as a last resort.
   c. NEVER push priority 1-3 tasks to tomorrow.
5. Pushed tasks go at the END of tomorrow's schedule.
6. Preserve exact task titles and goal_id values.
7. Respect sleep boundary: no tasks after 23:30.

OUTPUT FORMAT — Return ONLY valid JSON with two keys:
{{"rescheduled_today": [{{"start":"HH:MM","end":"HH:MM","title":"Task Name","type":"work|break|travel|fitness|call|errand|meal|shower","priority":1,"duration_min":60,"location":"place or null","notes":"context or null","goal_id":null}}], "rescheduled_tomorrow": [{{"start":"HH:MM","end":"HH:MM","title":"Task Name","type":"...","priority":1,"duration_min":60,"location":"place or null","notes":"context or null","goal_id":null}}]}}

No markdown fences, no explanation text."""

    response = _invoke_with_backoff(llm, [HumanMessage(content=prompt)])
    raw_text = extract_text(response.content)

    # Parse the response
    rescheduled_today = []
    rescheduled_tomorrow = tomorrow_schedule

    try:
        cleaned = raw_text.strip().strip("`").removeprefix("json").strip()
        data = json.loads(cleaned)
        if isinstance(data, dict):
            rescheduled_today = data.get("rescheduled_today", [])
            rescheduled_tomorrow = data.get("rescheduled_tomorrow", tomorrow_schedule)
    except (json.JSONDecodeError, TypeError):
        # Fallback: try brace-depth scanner
        objs = _find_all_json_objects(raw_text)
        for obj in objs:
            if "rescheduled_today" in obj:
                rescheduled_today = obj.get("rescheduled_today", [])
                rescheduled_tomorrow = obj.get("rescheduled_tomorrow", tomorrow_schedule)
                break

    pushed_count = len(rescheduled_tomorrow) - len(tomorrow_schedule)
    print(f"  Rescheduled: {len(rescheduled_today)} tasks today, "
          f"{pushed_count} pushed to tomorrow.")

    return {
        "rescheduled_today": rescheduled_today,
        "rescheduled_tomorrow": rescheduled_tomorrow,
        "messages": [HumanMessage(
            content=f"[Rescheduler] Rebuilt rest of {today_name}: "
                    f"{len(rescheduled_today)} tasks, {pushed_count} pushed to {tomorrow_name}."
        )],
    }
