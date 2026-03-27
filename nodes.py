import base64
import io
import json
import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pypdf import PdfReader

from state import SchedulerState

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------------
# LLM setup — Claude Haiku via Anthropic API
# ---------------------------------------------------------------------------
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_retries=3,
    timeout=60,
)


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


# ---------------------------------------------------------------------------
# Schedule JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_schedule_response(response_content):
    """
    Parse the scheduler's LLM response into (structured_list, text_version).
    Falls back to ([], raw_text) if JSON extraction fails.
    """
    raw = extract_text(response_content)

    # Try direct JSON parse (clean response)
    for attempt in [raw, raw.strip().strip("`").removeprefix("json").strip()]:
        try:
            data = json.loads(attempt)
            if isinstance(data, list):
                return data, _schedule_to_text(data)
        except (json.JSONDecodeError, TypeError):
            continue

    # Try to extract a JSON array from mixed text
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        data = json.loads(raw[start:end])
        if isinstance(data, list):
            return data, _schedule_to_text(data)
    except (ValueError, json.JSONDecodeError):
        pass

    print("  Warning: could not parse schedule JSON, using raw text.")
    return [], raw


def _schedule_to_text(entries):
    """Convert structured schedule JSON to readable text for the critic."""
    lines = []
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
        response = llm.invoke([HumanMessage(content=(
            "Extract all actionable tasks, deadlines, and to-do items "
            "from this document text. Return only the extracted tasks "
            "as a bullet-point list.\n\n" + text
        ))])
        return extract_text(response.content)

    elif ext in ("png", "jpg", "jpeg"):
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        media_type = "image/png" if ext == "png" else "image/jpeg"
        response = llm.invoke([HumanMessage(content=[
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
        # Step 1: Geocode via Open-Meteo's free geocoder (no API key needed)
        # Try the full location first, then fall back to just the city name
        # (Open-Meteo geocoder works best with city/place names, not full addresses)
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

        # Step 2: Fetch current weather from Open-Meteo
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
def estimate_commute(origin: str, destination: str) -> str:
    """Estimate travel/commute time between two locations using Google Maps Routes API.
    Returns the shortest route duration and distance."""
    try:
        # Google Routes API (New) — computeRoutes endpoint
        url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": GOOGLE_API_KEY,
            # Request these fields in the response
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

        # Pick the shortest route by duration
        # Duration comes as e.g. "1523s" — parse the seconds
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
        # Surface the actual Google error message if available
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
tools = [get_weather, estimate_commute]
llm_with_tools = llm.bind_tools(tools)

# Pre-built ToolNode executes whichever tools the LLM requested
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# Node 1: Task Ingester
# ---------------------------------------------------------------------------

def task_ingester(state: SchedulerState):
    """
    Parses the user's raw, messy input into structured tasks.
    Injects the exact current time so the LLM can reason about deadlines.
    """
    print("\n--- INGESTING TASKS ---")
    raw_tasks = state.get("raw_tasks", "No tasks provided.")
    now = _now(state)

    prompt = f"""You are an expert task analyzer. The current date and time is: {now}

Analyze these raw tasks: "{raw_tasks}"

For each task:
- Estimate a reasonable duration in minutes.
- Assign a priority from 1 to 10 (1 = most urgent/important, 10 = least).
- Note any implied deadlines relative to the current time.
- Note any locations mentioned (useful for travel/commute planning).

Return ONLY a valid JSON list of objects — no markdown, no code fences:
[{{"task": "name", "duration": 60, "priority": 1, "deadline": "HH:MM or null", "location": "place or null"}}]"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Attempt to parse JSON from the LLM response
    try:
        raw_text = extract_text(response.content)
        cleaned = raw_text.strip().strip("`").removeprefix("json").strip()
        parsed_tasks = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  Warning: could not parse task JSON ({e}). Using fallback.")
        parsed_tasks = [
            {"task": raw_tasks, "duration": 60, "priority": 5, "deadline": None, "location": None}
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
# Node 2: Scheduler (tool-calling agent)
# ---------------------------------------------------------------------------

def scheduler(state: SchedulerState):
    """
    Drafts a chronological schedule. Uses bound tools (weather, commute)
    when the plan involves travel or outdoor activities.

    If the last message is a ToolMessage, this is a continuation after tool
    execution — re-invoke the LLM with the full message history so it can
    incorporate the tool results into its schedule.
    """
    print("\n--- DRAFTING SCHEDULE ---")
    messages = state.get("messages", [])
    revision_count = state.get("revision_count", 0)

    # ---- Continuation after tool execution ----
    if messages and isinstance(messages[-1], ToolMessage):
        response = llm_with_tools.invoke(messages)
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

    critique_section = critique if critique else "None — this is the first draft."

    prompt = f"""You are an expert productivity scheduler. The current time is: {now}

USER'S CURRENT LOCATION: {user_location}

TASKS TO SCHEDULE:
{json.dumps(parsed_tasks, indent=2)}

PREVIOUS CRITIQUE TO ADDRESS:
{critique_section}

RULES:
1. Start the schedule from NOW ({now}). Do NOT use a generic 9-to-5 template.
2. Schedule higher-priority tasks (priority closer to 1) earlier in the day.
3. Respect any deadlines relative to the current time.
4. Include 5-10 minute breaks between tasks.
5. ONLY schedule the tasks listed above — do not invent new ones.

MANDATORY TOOL USAGE:
Before generating the schedule, you MUST call the following tools:
- For ANY task that has a non-null "location" field, call estimate_commute(origin="{user_location}", destination=<task location>) to get realistic travel time. Add this travel time to the schedule.
- For ANY outdoor or physical task (gym, sports, walking, park, etc.), call get_weather(location="{user_location}") to check conditions and note them in the schedule.
Do NOT skip tool calls. Call the tools FIRST, then build the schedule using their results.

OUTPUT FORMAT:
Return ONLY a valid JSON array — no markdown fences, no text before or after.
Each element is one scheduled block:
[{{"start":"HH:MM","end":"HH:MM","title":"Task Name","type":"work|break|travel|fitness|call|errand|meal|shower","priority":1,"duration_min":60,"location":"place or null","notes":"1-line context or null","weather":"brief weather note or null","commute":"travel info or null"}}]

type must be one of: work, break, travel, fitness, call, errand, meal, shower.
priority is 1-10 for real tasks, null for breaks/travel."""

    human_msg = HumanMessage(content=prompt)
    response = llm_with_tools.invoke(messages + [human_msg])
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
# Node 3: Critic
# ---------------------------------------------------------------------------

def critic(state: SchedulerState):
    """
    Evaluates the draft schedule against the original tasks, priorities,
    deadlines, and the current time. Outputs specific fixes or 'PERFECT'.
    """
    print("\n--- CRITIQUING SCHEDULE ---")
    draft_schedule = state.get("draft_schedule", "")
    parsed_tasks = state.get("parsed_tasks", [])
    revision_count = state.get("revision_count", 0)
    now = _now(state)

    prompt = f"""You are a strict productivity critic. The current time is: {now}

PROPOSED SCHEDULE:
{draft_schedule}

ORIGINAL TASKS:
{json.dumps(parsed_tasks, indent=2)}

Evaluate against these criteria:
1. Are ALL tasks included? None should be missing.
2. Are high-priority tasks (priority closest to 1) scheduled earlier in the day?
3. Are deadlines respected given the current time?
4. Is there realistic buffer and travel time between tasks?
5. Does the schedule start from the current time, not a generic morning?

This is revision {revision_count} of 3.

If the schedule meets ALL criteria, reply with EXACTLY the word: PERFECT
Otherwise, provide a concise bulleted list of specific changes needed."""

    response = llm.invoke([HumanMessage(content=prompt)])
    critique_text = extract_text(response.content).strip()

    # Log to message history so future iterations have context
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

    - On approval: sets human_feedback to 'approved' and routes to END.
    - On tweak: injects the feedback as a new critique and routes back
      to the scheduler for another revision.
    """
    print("\n--- AWAITING HUMAN REVIEW ---")

    schedule = state.get("draft_schedule", "")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    # interrupt() pauses the graph and surfaces this payload to the caller.
    # When the caller resumes with Command(resume=<value>), that value is
    # returned here as `feedback`.
    feedback = interrupt({
        "schedule": schedule,
        "final_critique": critique,
        "revisions_used": revision_count,
        "prompt": "Do you approve this schedule? Type 'approve' or describe your tweaks.",
    })

    feedback_str = str(feedback).strip()

    if feedback_str.lower() in ("approve", "approved", "yes", "y", "ok", "looks good", "lgtm"):
        print("  User approved the schedule.")
        return {
            "human_feedback": "approved",
            "messages": [HumanMessage(content="[Human] Schedule approved.")],
        }
    else:
        print(f"  User requested changes: {feedback_str}")
        return {
            "human_feedback": feedback_str,
            "critique": f"USER REQUESTED CHANGES: {feedback_str}",
            "messages": [HumanMessage(content=f"[Human Feedback] {feedback_str}")],
        }
