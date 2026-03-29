# AI Weekly Schedule Optimizer

> A multi-agent AI system that builds your perfect week — balancing university classes, long-term goals, and dynamic tasks — then lets you review, tweak, and finalize the plan before a single minute is committed.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-00A67E.svg)](https://github.com/langchain-ai/langgraph)
[![Gemini 2.5 Pro](https://img.shields.io/badge/Gemini-2.5_Pro-4285F4.svg)](https://ai.google.dev/)


Built with **LangGraph** for orchestration and **Gemini 2.5 Pro** for reasoning, this is not a simple chatbot wrapper — it is a stateful, multi-node agent graph with tool-calling, a critic loop, and a human-in-the-loop approval gate. The scheduler reasons about physical constraints (commute times, 90-minute focus limits, workload balance across days) and uses real-time weather and route data to place tasks intelligently.

---

## Key Features

### Multi-Agent Architecture
Six specialized LangGraph nodes collaborate through a directed graph with conditional routing and revision loops. Each node has a single responsibility — decomposing goals, ingesting tasks, drafting the schedule, critiquing it, and gating human approval — ensuring separation of concerns and debuggability.

### Real-Time Tool Integration
The Scheduler node has access to three LangChain tools it can invoke autonomously:
- **`get_weather`** — Fetches current conditions from [Open-Meteo](https://open-meteo.com/) to annotate outdoor tasks.
- **`get_weekly_forecast`** — Pulls a 7-day forecast to plan weather-sensitive tasks on the best days.
- **`estimate_commute`** — Calculates driving time and distance via the [Google Routes API](https://developers.google.com/maps/documentation/routes) to insert realistic travel blocks between locations.

### Multimodal Document Processing
Upload PDFs, images (JPG/PNG), or text files of syllabi, handwritten to-do lists, or timetable screenshots. The AI extracts every task, class, and deadline using Gemini's vision capabilities — no manual transcription needed.

### Strategic Goal Tracking
Define multi-month goals (e.g., "Obtain Driving License" or "Read 500 pages of Brothers Karamazov") with deadlines and hour estimates. The **Goal Decomposer** node automatically calculates weekly chunks based on remaining time and injects them into the schedule. Progress bars track completion across sessions.

### Human-in-the-Loop (HITL) Review
The graph uses LangGraph's `interrupt()` mechanism to pause execution after the Critic approves a draft. You see the full schedule, critique it in plain English ("move boxing to the evening", "I can't do Tuesday afternoon"), and the Scheduler incorporates your feedback in the next revision — up to 7 iterations.

### Session Persistence & Templates
- **Templates**: Save recurring weekly schedules (e.g., a full university semester timetable) and load them with one click. Fixed classes are treated as non-negotiable anchors.
- **Session History**: Every finalized schedule is stored in SQLite with full state, so you can review past weeks.
- **Streak Tracking**: Consecutive days with all P1 tasks completed are tracked as a productivity streak.

### Focus Mode & Dynamic Rescheduling
- **Focus Mode**: A distraction-free view that highlights only the current task with a live countdown timer (powered by `streamlit-autorefresh`).
- **"I'm Behind" Button**: Triggers a separate `RescheduleState` graph that compresses remaining tasks into the rest of the day and pushes low-priority items to tomorrow.

### Export
One-click CSV and Excel export via Pandas, producing a clean spreadsheet with day, time, task, type, priority, location, goal, and notes columns.

---

## System Architecture

The core scheduling pipeline is a **LangGraph `StateGraph`** with conditional edges and a revision loop:

```
START
  │
  ▼
┌─────────────────┐
│ Goal Decomposer │  Queries active long-term goals from SQLite,
│                 │  calculates weeks remaining, and generates
│                 │  concrete weekly task chunks via LLM.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Document      │  Merges uploaded file text, template tasks,
│   Processor     │  and goal chunks into a single raw input
│                 │  string for the Task Ingester.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Task Ingester  │  LLM parses the raw text into structured JSON:
│                 │  task name, duration, priority, location, day
│                 │  preference, recurrence, deadlines.
│                 │  Post-processing assigns goal_id slugs.
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌───────────┐
│   Scheduler     │─────▶│   Tools   │  Weather / Forecast / Commute
│                 │◀─────│  (LangChain)
│  (2-phase:      │      └───────────┘
│   tool-calling  │
│   then JSON     │
│   generation)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌─────────────────┐
│     Critic      │──NO──▶│   Scheduler     │  (revision loop,
│  (thinking LLM) │       │   (retry)       │   up to 7 rounds)
│                 │       └─────────────────┘
└────────┬────────┘
         │ PERFECT
         ▼
┌─────────────────┐
│  Human Review   │  interrupt() pauses the graph.
│  (HITL gate)    │  User approves or requests tweaks.
│                 │──TWEAK──▶ back to Scheduler
└────────┬────────┘
         │ APPROVED
         ▼
        END
```

### Pass-Through Goal ID Architecture

To prevent the LLM from hallucinating or misassigning goal IDs, the system uses **semantic string slugs** (e.g., `"goal_read_brothers_karamazov"`) instead of fragile integer indices. A three-layer programmatic enforcement pipeline (`_enforce_goal_ids`) runs after every schedule generation:

1. **Layer 1 — Exact Match**: Title matches a parsed task with a known `goal_id` → copy it.
2. **Layer 2 — Fuzzy Keyword Match**: Stemmed keyword overlap against parsed tasks (handles renamed titles).
3. **Layer 3 — Macro-Goal Match**: Keyword overlap against goal names directly (catches chunked tasks like "Read Karamazov pages 1-25" → goal "Read 100 pages from Brothers Karamazov").

The LLM's `goal_id` output is **never trusted** — it is always overwritten by this deterministic pipeline.

---

## Tech Stack

| Layer            | Technology                                                       |
| ---------------- | ---------------------------------------------------------------- |
| **LLM**          | Google Gemini 2.5 Pro (via `langchain-google-genai`)             |
| **Orchestration**| LangGraph `StateGraph` with conditional edges and `interrupt()`  |
| **Framework**    | LangChain Core (messages, tools, tool nodes)                     |
| **Validation**   | Pydantic v2 (`BaseModel` for `ScheduledTask` / `WeeklySchedule`)|
| **Frontend**     | Streamlit with `streamlit-autorefresh` for live timers           |
| **Persistence**  | SQLite (sessions, streaks, templates, long-term goals)           |
| **Checkpointer** | `langgraph-checkpoint-sqlite` for graph state persistence        |
| **APIs**         | Open-Meteo (weather), Google Routes API (commute times)          |
| **Export**        | Pandas + openpyxl (CSV and Excel)                               |
| **PDF Parsing**  | pypdf                                                            |
| **Containerization** | Docker + Docker Compose                                     |

---

## Installation & Setup

### Prerequisites

- Python 3.12+
- A [Google AI Studio](https://aistudio.google.com/) API key (for Gemini 2.5 Pro)
- A [Google Cloud](https://console.cloud.google.com/) API key with the Routes API enabled (for commute estimation)
- Docker & Docker Compose (optional, for containerized deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-weekly-schedule-optimizer.git
cd ai-weekly-schedule-optimizer
```

### 2. Create a `.env` File

```bash
cp .env.example .env
```

Add your API keys:

```env
# Required — powers all LLM calls (Gemini 2.5 Pro)
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here

# Required — used for commute time estimation (Google Routes API)
GOOGLE_API_KEY=your_google_cloud_api_key_here
```

> **Note**: Weather data from Open-Meteo requires no API key.

### 3a. Run with Docker (Recommended)

```bash
docker compose up --build
```

The app will be available at **http://localhost:8501**. SQLite databases are persisted in a Docker volume.

### 3b. Run Locally

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### 4. Run in Terminal Mode (No UI)

The graph can also run headless for quick testing:

```bash
python graph.py
```

This starts an interactive CLI session with the same HITL interrupt flow.

---

## Project Structure

```
.
├── app.py              # Streamlit frontend — UI, checkboxes, progress bars, export
├── graph.py            # LangGraph definition — nodes, edges, routing, compilation
├── nodes.py            # All LLM nodes, tools, Pydantic models, goal enforcement
├── state.py            # TypedDict state schemas (SchedulerState, RescheduleState)
├── session_store.py    # SQLite CRUD for sessions, streaks, templates, goals
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image definition
├── docker-compose.yml  # Single-command deployment with persistent volume
└── .env                # API keys (not committed)
```

---

## How It Works — A Typical Session

1. **Set your location** in the sidebar (used for timezone detection and commute calculation).
2. **Load a template** (e.g., your university semester timetable) or start from scratch.
3. **Type your tasks** for the week — ad-hoc errands, project work, fitness goals.
4. **Upload documents** (optional) — photos of handwritten lists, PDF syllabi, etc.
5. **Set weekly macro-goals** (optional) — "Read 100 pages of Dostoyevsky", "3 boxing sessions".
6. **Click Generate** — the AI builds a full 7-day schedule with travel blocks, breaks, weather annotations, and goal-tagged entries.
7. **Review the draft** — the Critic has already checked for missing tasks, impossible transitions, and goal misattribution. You can approve or request changes in plain English.
8. **Track your week** — check off tasks as you complete them. Progress bars update live. Use Focus Mode for distraction-free execution.
9. **Hit "I'm Behind"** if your day goes off-track — the AI compresses remaining tasks and pushes low-priority items to tomorrow.
10. **Export** the final schedule as CSV or Excel.

