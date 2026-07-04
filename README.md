# Iris — Autonomous Document Agent

Iris is an AI agent that turns natural-language requests into polished Word (.docx) documents. Give it a prompt like *"Create a project plan for launching a customer support knowledge base"* and it autonomously plans the structure, writes each section, reviews its own output, and returns a downloadable DOCX — all without human intervention.

Built with [LangGraph](https://github.com/langchain-ai/langraph) and [Groq](https://groq.com) (llama-3.1-8b-instant), Iris runs as a FastAPI server with an optional Streamlit UI.

---

## How it works

Iris is a directed acyclic graph of seven nodes. Each node is a distinct agent responsibility:

```
User request
    │
    ▼
┌────────────────┐
│   planner      │  Decomposes the request into a document brief + TODO list
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ orchestrator   │  Routes pending tasks to the worker (loops until all done)
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  todo_worker   │  Writes one section at a time (heading, paragraphs, bullets)
└───────┬────────┘
        │  ◄──── loops back to orchestrator
        ▼
┌────────────────┐
│  reflection    │  Self-check: reviews and revises every section for quality
└───────┬────────┘
        │
        ▼
┌──────────────────────────┐
│ docx_content_generator   │  Assembles sections into structured markdown
└───────────┬──────────────┘
            │
            ▼
┌──────────────────┐
│ generate_report  │  Renders markdown → python-docx → base64 DOCX
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  chat_response   │  Writes the final user-facing summary message
└─────────┬────────┘
          │
          ▼
     Response (DOCX + message + task trace)
```

**Key design points:**

- The **orchestrator → worker loop** lets Iris handle any number of sections without modifying the graph.
- The **reflection node** acts as a quality gate — it reviews the full draft and rewrites sections that need improvement.
- Every node communicates through a shared `TypedDict` state, so nodes are decoupled and individually testable.

---



## Features


| Feature                     | Details                                                                                                                                                                                                    |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Autonomous planning**     | LLM decomposes any business-document request into 3–6 section tasks with a title, tone, and audience                                                                                                       |
| **Multi-step execution**    | LangGraph state machine routes tasks through the worker loop until all sections are complete                                                                                                               |
| **Reflection / self-check** | A dedicated review node rewrites each section for clarity, structure, and completeness — a content-richness check ensures no content is lost (the **one engineering improvement** from the original brief) |
| **DOCX generation**         | python-docx produces properly formatted Word files with headings, paragraphs, bullet lists, and styled margins                                                                                             |
| **Streaming progress**      | SSE endpoint pushes real-time per-node updates so the UI can show live progress                                                                                                                            |
| **Structured LLM output**   | All LLM calls use Pydantic-validated JSON schemas with retry and error recovery                                                                                                                            |
| **REST API**                | `POST /agent` for blocking calls, `POST /agent/stream` for SSE, `POST /documents/save` and `GET /documents/{filename}` for document persistence                                                            |
| **MockLLM fallback**        | Offline mode for development without API credits — swap `GroqLLM` for `MockLLM`                                                                                                                            |


---



## The reflection improvement

The assignment required **one real engineering improvement**. Iris implements **reflection / self-check**: after all worker nodes finish drafting sections, the reflection node reviews the full document and rewrites it.

**Why this choice:** The llama-3.1-8b-instant model is small and fast but can produce thin or slightly off-topic content. A review pass catches weak sections, strengthens prose, and ensures consistency. A content-richness safety check compares each revised section against its original — if the revision is thinner (fewer paragraphs + bullets), the original is kept instead.

**What it required:**

- A flat `ReflectionSpec` schema that the 8B model can reliably produce (no nested Pydantic models — the model struggles with `$defs`)
- A `_compact_schema()` helper that renders field names and types as human-readable text instead of JSON Schema keywords (prevents the model from echoing schema definitions back as data)
- A `_split_section_text()` heuristic to handle the case where the LLM returns all sections concatenated in one string
- Increased `max_retries` (2 → 3) for the reflection call, since it produces the longest output

---



## Project structure

```
iris/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI server — 4 routes
│   │   └── settings.py      # pydantic-settings (reads .env)
│   ├── agent/
│   │   ├── graph.py         # LangGraph StateGraph — nodes + edges
│   │   ├── factory.py       # IrisAgent class — run() and run_streamed()
│   │   ├── agent_state.py   # State models (TypedDict + Pydantic)
│   │   └── nodes/
│   │       ├── planner.py
│   │       ├── orchestrator.py
│   │       ├── todo_worker.py
│   │       ├── content_generator.py
│   │       ├── generate_docx.py
│   │       ├── reflection.py
│   │       └── response.py
│   ├── llms/
│   │   ├── llm.py           # LLM protocol + MockLLM + shared schemas
│   │   ├── groq.py          # GroqLLM implementation
│   │   └── ai_client.py     # AIClientContainer (AsyncGroq)
│   ├── schemas/
│   │   ├── request.py       # UserRequest
│   │   └── response.py      # AgentResponse, DocxSaveRequest/Response
│   ├── streamlit_app.py     # Standalone demo UI
│   ├── pyproject.toml       # uv-managed dependencies
│   ├── uv.lock
│   └── artifacts/           # Saved DOCX files (gitignored)
├── .env                     # GROQ_API_KEY (gitignored)
├── .gitignore
├── ASSIGNMENT.txt           # Original assignment brief
└── README.md
```

---



## Quick start



### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Groq API key](https://console.groq.com) (free tier works)



### Setup

```bash
# Clone the repo
git clone <repo-url>
cd iris

# Set your API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# Start the API server
cd backend
uv run uvicorn app.main:app --reload
```

The API is now running at `http://localhost:8000`.

### Start the Streamlit UI (optional)

```bash
# In a separate terminal
uv run streamlit run streamlit_app.py
```

---



## API reference



### `POST /agent`

Generate a document synchronously.

```json
{"request": "Create a project plan for launching a customer support knowledge base with a 2-week timeline."}
```

Returns an `AgentResponse` with the DOCX as a base64 string plus the full task trace.

### `POST /agent/stream`

Same as `/agent` but yields Server-Sent Events for live progress updates. Each event contains the current node name, task completion status, and active task.

### `POST /documents/save`

Persist a generated DOCX to the server's `artifacts/` directory.

### `GET /documents/{filename}`

Download a previously saved DOCX file.

---



## Example usage

```bash
curl -s -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"request": "Draft a one-page project plan for launching a knowledge base"}' \
  | python3 -c "
import sys, json, base64
resp = json.load(sys.stdin)
print('Title:', resp['title'])
print('Type:', resp['document_type'])
print('Tasks:', len(resp['tasks']))
print('DOCX length:', len(resp['docx_base64']), 'bytes')
print()
print('Message:', resp['message'])
print()
print('Assumptions:')
for a in resp['assumptions']:
    print(f'  - {a}')
print()
print('Task trace:')
for t in resp['tasks']:
    print(f'  [{t[\"status\"]}] Task {t[\"id\"]}: {t[\"task\"]}')
"
```

---



## Engineering decisions and tradeoffs


| Decision                                    | Rationale                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **LangGraph over raw LangChain / CrewAI**   | LangGraph's `StateGraph` gives explicit control over the node-level execution flow and state passing. Makes the agent's decision-making visible and debuggable — every state mutation is traceable to one node.                                                                                              |
| **Flat Pydantic schemas**                   | The 8B Groq model cannot reliably produce JSON with `$defs` (nested models). Solved by flattening all schemas — `PlanSpec.tasks` became `list[str]`, `ReflectionSpec.sections` became `list[str]`. Every field is a primitive or a list of primitives.                                                       |
| **Compact schema prompts over JSON Schema** | Feeding `model_json_schema()` into the prompt caused the model to echo schema keywords back as data (e.g. `"properties": {"title": ...}`). The fix: a `_compact_schema()` helper that renders each field as `"field_name": string (required)` — readable by the model and impossible to confuse with output. |
| **Orchestrator-worker loop**                | Instead of dynamic sub-graph spawning, a simple conditional edge loops between orchestrator and worker. This is less flexible but drastically simpler — no sub-graph state management, no parallel execution coordination.                                                                                   |
| **MockLLM for offline dev**                 | A `MockLLM` class implements the same `LLM` protocol with canned responses. Enables frontend and API development without any LLM API calls or API keys.                                                                                                                                                      |
| **Streaming via SSE**                       | Server-Sent Events over WebSocket: simpler infrastructure (no connection manager, no ping/pong), naturally maps to LangGraph's `astream(stream_mode="updates")`, and works with Streamlit's `st.write_stream()`.                                                                                             |


---



## Stack


| Layer               | Technology                  |
| ------------------- | --------------------------- |
| Language            | Python 3.10                 |
| Agent framework     | LangGraph 1.2+              |
| LLM provider        | Groq (llama-3.1-8b-instant) |
| API                 | FastAPI + Uvicorn           |
| Document generation | python-docx                 |
| Frontend (demo)     | Streamlit                   |
| Package manager     | uv                          |
| Validation          | Pydantic 2                  |
| Configuration       | pydantic-settings           |


---

