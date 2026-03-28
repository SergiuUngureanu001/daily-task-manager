import base64
import io
import json
import os
import time
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pypdf import PdfReader

try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AnthropicRateLimitError = None

from state import SchedulerState

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------------
# LLM setup — Claude Haiku via Anthropic API
# ---------------------------------------------------------------------------
# max_retries=0 disables the SDK's own fast retries for rate limits — we handle
# backoff ourselves in _invoke_with_backoff with longer, smarter delays.
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_retries=0,
    timeout=120,
)


# ---------------------------------------------------------------------------
# Rate-limit-aware LLM invoke wrapper with exponential backoff
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
BASE_DELAY = 20  # seconds — rate limit window is 60s, so 20s is a safe first wait


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect whether an exception is a 429 rate-limit error."""
    # 1. Check by exception type (anthropic SDK raises RateLimitError)
    if AnthropicRateLimitError and isinstance(exc, AnthropicRateLimitError):
        return True
    # 2. Check the HTTP status code if available (langchain wrappers)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    # 3. Fallback: string matching on error message
    error_str = str(exc).lower()
    return "429" in error_str or "rate_limit" in error_str


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
# Document processing helpers — called from app.py before graph invocation
# ---------------------------------------------------------------------------

def process_uploaded_file(filename: str, file_bytes: bytes) -> str:
    """
    Process an uploaded file and extract actionable task text.
    Handles .txt, .pdf (via pypdf), and images (via Claude vision).
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
            "Extract all actionable tasks, deadlines, and to-do items "
            "from this document text. Return only the extracted tasks "
            "as a bullet-point list.\n\n" + text
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
                    "Extract all actionable tasks, deadlines, and to-do items "
                    "visible in this image. Return them as a bullet-point list."
                ),
            },
        ])])
        return extract_text(response.content)

    return f"(Unsupported file type: .{ext})"


# ---------------------------------------------------------------------------
# Node 0: Document Processor — merges uploaded file text with raw_tasks
# ---------------------------------------------------------------------------

def document_processor(state: SchedulerState):
    """
    Combines text extracted from uploaded documents with the user's
    typed tasks so the Task Ingester has a complete picture.
    """
    uploaded_text = state.get("uploaded_files_text", "")
    raw_tasks = state.get("raw_tasks", "")

    if uploaded_text.strip():
        print("\n--- DOCUMENT PROCESSOR ---")
        print(f"  Merging uploaded document text with user tasks.")
        combined = (
            f"{raw_tasks}\n\n"
            f"Additional tasks extracted from uploaded documents:\n"
            f"{uploaded_text}"
        )
        return {"raw_tasks": combined}

    return {}


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

Analyze these raw tasks: "{raw_tasks}"

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

Return ONLY a valid JSON list of objects — no markdown, no code fences:
[{{"task": "name", "duration": 60, "priority": 1, "deadline": "HH:MM or null", "location": "full disambiguated address or null", "preferred_day": "Monday or null", "is_recurring": false, "recurrence_days": [], "date_deadline": "YYYY-MM-DD or null"}}]"""

    response = _invoke_with_backoff(llm, [HumanMessage(content=prompt)])

    try:
        raw_text = extract_text(response.content)
        cleaned = raw_text.strip().strip("`").removeprefix("json").strip()
        parsed_tasks = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  Warning: could not parse task JSON ({e}). Using fallback.")
        parsed_tasks = [
            {"task": raw_tasks, "duration": 60, "priority": 5, "deadline": None,
             "location": None, "preferred_day": None, "is_recurring": False,
             "recurrence_days": [], "date_deadline": None}
        ]

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

def scheduler(state: SchedulerState):
    """
    Drafts a 7-day chronological schedule. Uses bound tools (weather, forecast,
    commute) when the plan involves travel or outdoor activities.
    """
    print("\n--- DRAFTING WEEKLY SCHEDULE ---")
    messages = state.get("messages", [])
    revision_count = state.get("revision_count", 0)

    # ---- Continuation after tool execution ----
    if messages and isinstance(messages[-1], ToolMessage):
        response = _invoke_with_backoff(llm_with_tools, messages)
        has_more_tool_calls = bool(getattr(response, "tool_calls", None))
        if has_more_tool_calls:
            return {"messages": [response]}
        structured, text = _parse_schedule_response(response.content)
        return {
            "messages": [response],
            "draft_schedule": text,
            "structured_schedule": structured,
        }

    # ---- Fresh scheduling attempt ----
    parsed_tasks = state.get("parsed_tasks", [])
    critique = state.get("critique", "")
    user_location = state.get("user_location", "Home")
    now = _now(state)
    week = _week_dates(state)

    critique_section = critique if critique else "None — this is the first draft."
    week_display = "\n".join(f"  - {d['display']} ({d['date_str']})" for d in week)

    prompt = f"""You are an expert productivity scheduler planning a FULL 7-DAY WEEK.
The current time is: {now}

USER'S CURRENT LOCATION: {user_location}

THE PLANNING WEEK:
{week_display}

TASKS TO SCHEDULE:
{json.dumps(parsed_tasks, indent=2)}

PREVIOUS CRITIQUE TO ADDRESS:
{critique_section}

SCHEDULING HIERARCHY (in strict order of precedence):
1. Hard deadlines (date_deadline) and fixed appointments are ABSOLUTE — never violate them.
2. Direct instructions from the user's critique (e.g. "Move gym to Wednesday") override all other logic.
3. Tasks with preferred_day should be placed on that day when possible.
4. Recurring tasks (is_recurring=true) must appear on ALL their recurrence_days.
5. Only when time is flexible should you sort by the 1-10 priority scale.
6. Do NOT arrive at locations more than 15 minutes early — time travel precisely.

WEEKLY PLANNING RULES:
1. TODAY ({week[0]['display']}): Start from NOW ({now}). Do NOT use a generic 9-to-5 template.
2. FUTURE DAYS: Plan from 08:00 to 23:00 (respect sleep boundaries — no tasks before 07:00 or after 23:30).
3. Include 5-10 minute breaks between tasks.
4. ONLY schedule the tasks listed above — do not invent new ones.
5. Do NOT create oversized buffer blocks. Once a task is done, move to the next one.
6. Balance workload across the week — avoid stacking everything on one day.
7. Group location-based tasks on the same day to minimize commute.
8. Place high-priority tasks earlier in the week when possible.
9. Recurring tasks (e.g. gym Mon/Wed/Fri) must appear on each specified day.
10. If a task has a date_deadline, it MUST be completed on or before that date.

MANDATORY TOOL USAGE:
Before generating the schedule, you MUST call the following tools:
- Call get_weekly_forecast(location="{user_location}") to get the 7-day weather outlook. Use this to plan outdoor activities on good-weather days.
- For ANY task with a non-null "location" field, call estimate_commute(origin="{user_location}", destination=<task location>) to get realistic travel time.
- For ANY outdoor or physical task (gym, sports, walking, park, etc.), call get_weather(location="{user_location}") for current conditions (today's tasks).
Do NOT skip tool calls. Call the tools FIRST, then build the schedule using their results.

OUTPUT FORMAT:
Return ONLY a valid JSON object — no markdown fences, no text before or after.
The object is keyed by day name, each value is an array of scheduled blocks:

{{"Monday": [{{"start":"HH:MM","end":"HH:MM","title":"Task Name","type":"work|break|travel|fitness|call|errand|meal|shower","priority":1,"duration_min":60,"location":"place or null","notes":"1-line context or null","weather":"brief weather note or null","commute":"travel info or null"}}], "Tuesday": [...], ...}}

Include ALL 7 days of the week ({', '.join(d['day_name'] for d in week)}).
Days with no tasks should still appear with an empty array.
type must be one of: work, break, travel, fitness, call, errand, meal, shower.
priority is 1-10 for real tasks, null for breaks/travel."""

    human_msg = HumanMessage(content=prompt)
    response = _invoke_with_backoff(llm_with_tools, messages + [human_msg])
    has_tool_calls = bool(getattr(response, "tool_calls", None))

    if has_tool_calls:
        return {
            "messages": [human_msg, response],
            "revision_count": revision_count + 1,
        }

    structured, text = _parse_schedule_response(response.content)
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

    prompt = f"""You are a strict productivity critic evaluating a WEEKLY schedule.
The current time is: {now}

THE PLANNING WEEK:
{week_display}

PROPOSED WEEKLY SCHEDULE:
{draft_schedule}

ORIGINAL TASKS:
{json.dumps(parsed_tasks, indent=2)}

Evaluate against these criteria:
1. COMPLETENESS: Are ALL tasks included? None should be missing. Recurring tasks must appear on every specified recurrence day.
2. DEADLINES: Are all date_deadline constraints met? Tasks must be scheduled on or before their deadline date.
3. DAY PREFERENCES: Are preferred_day assignments respected?
4. WORKLOAD BALANCE: Is the work spread reasonably across the week? No single day should be overloaded while others are empty.
5. TODAY'S SCHEDULE: Does today's plan start from the current time, not a generic morning?
6. SLEEP BOUNDARIES: No tasks scheduled before 07:00 or after 23:30.
7. PRIORITY ORDER: Within each day, are high-priority tasks (closest to 1) scheduled earlier?
8. TRAVEL LOGIC: Tasks with locations have realistic commute time? Location-based tasks grouped on same days?
9. BREAKS: Are there reasonable breaks between tasks?

This is revision {revision_count} of 7.

If the schedule meets ALL criteria, reply with EXACTLY the word: PERFECT
Otherwise, provide a concise bulleted list of specific changes needed."""

    response = _invoke_with_backoff(llm, [HumanMessage(content=prompt)])
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
