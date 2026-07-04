# Iris — Autonomous Document Agent

## Quick start

```bash
# Start API (from backend/)
uv run uvicorn app.main:app --reload

# Start demo UI (separate terminal)
uv run streamlit run streamlit_app.py
```

`POST /agent` with `{"request": "..."}` returns a DOCX as base64 + task trace.  
`POST /documents/save` persists a DOCX to `backend/artifacts/`.  
`GET /documents/{filename}` serves a saved DOCX.

## Project structure

```
backend/
├── app/
│   ├── main.py          # FastAPI entrypoint — 4 routes: /agent, /health, /documents/save, /documents/{filename}
│   └── settings.py      # pydantic-settings, reads .env from repo root
├── agent/
│   ├── graph.py         # LangGraph StateGraph — nodes + edges
│   ├── factory.py       # IrisAgent class, loads .env, injects GroqLLM
│   ├── agent_state.py   # TypedDict + Pydantic models for graph state
│   └── nodes/
│       ├── planner.py           # LLM decomposes request → PlanSpec + Todo list
│       ├── orchestrator.py      # Routes: pending task → worker, else → next_phase
│       ├── todo_worker.py       # LLM generates one DocumentSection per task
│       ├── content_generator.py # Assembles sections into markdown
│       ├── generate_docx.py     # python-docx → base64 DOCX
│       ├── reflection.py        # (available but NOT wired — see graph.py line 22)
│       └── response.py          # LLM writes final user message
├── llms/
│   ├── llm.py           # LLM Protocol + MockLLM (offline fallback) + shared helpers
│   ├── groq.py          # GroqLLM — uses llama-3.1-8b-instant
│   └── ai_client.py     # AIClientContainer (holds AsyncGroq)
├── schemas/             # Pydantic request/response models
├── streamlit_app.py     # Standalone Streamlit UI
├── pyproject.toml       # uv-managed dependencies
├── uv.lock
└── artifacts/           # Saved DOCX files (gitignored)
```

## Key facts

- **Python 3.10** (`backend/.python-version`), **uv** package manager.
- LLM is **Groq / llama-3.1-8b-instant**. Set `GROQ_API_KEY` in `.env` at repo root.
- **MockLLM** in `llms/llm.py` can be used for offline development (swap in tests).
- Settings (`app/settings.py`) expose `IRIS_LLM_PROVIDER`, `IRIS_LLM_MODEL`, `IRIS_MAX_PLAN_TASKS` but only Groq path is implemented.
- **Reflection node** (`agent/nodes/reflection.py`) is wired and runs after all workers complete. Graph: `planner → orchestrator ↔ todo_worker (loop) → reflection → docx_content_generator → generate_report_node → chat_response → END`.
- Agent always runs **LangGraph `StateGraph`** with `ainvoke()`.
- No test framework, no linter/formatter/typecheck config present.
- DOCX artifacts go to `backend/artifacts/` (gitignored).

## Conventions

- All imports use `from __future__ import annotations`.
- State models in `agent_state.py` use `TypedDict(total=False)`.
- LLM calls use structured output via `generate_structured(schema, ...)` with JSON schema enforcement.
- Schemas must be **flat** (no nested Pydantic models) — the small Groq model struggles with `$defs`. Lists of strings are preferred over lists of structured objects. See `PlanSpec.tasks` and `ReflectionSpec.sections` as templates.
- System prompts are inline strings, not separate files.
