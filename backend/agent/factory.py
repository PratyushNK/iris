from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncIterator, cast

from groq import AsyncGroq

from agent.agent_state import IrisAgentState as AgentState
from agent.graph import iris
from llms.groq import GroqLLM
from schemas.response import AgentResponse, TodoResponse


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


NODE_LABELS: dict[str, str] = {
    "planner": "Planning document structure...",
    "orchestrator": "Scheduling tasks...",
    "todo_worker": "Writing section...",
    "reflection": "Reviewing and refining document...",
    "docx_content_generator": "Assembling content...",
    "generate_report_node": "Generating DOCX file...",
    "chat_response": "Writing final response...",
}


def _todo_json(tasks: list) -> list[dict]:
    return [
        {
            "id": t.id,
            "task": f"{t.section_heading}: {t.task}",
            "status": t.status.value,
        }
        for t in tasks
    ]


class IrisAgent:
    def __init__(self) -> None:
        _load_env_file()
        self.graph = iris
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required to run the Groq-backed agent")
        self.llm = GroqLLM(client=AsyncGroq(api_key=groq_api_key))

    async def run(self, user_query: str) -> AgentResponse:
        initial_state: AgentState = cast(AgentState, {"query": user_query, "llm": self.llm})

        result = cast(AgentState, await self.graph.ainvoke(initial_state))

        tasks = result.get("tasks", [])
        response_tasks = [
            TodoResponse(
                id=task.id,
                task=f"{task.section_heading}: {task.task}",
                status=task.status.value,
                result=task.result,
            )
            for task in tasks
        ]

        return AgentResponse(
            message=result.get("final_response", "The autonomous agent completed the request."),
            title=result.get("title", "Autonomous Deliverable"),
            document_type=result.get("document_type", "business report"),
            assumptions=result.get("assumptions", []),
            tasks=response_tasks,
            docx_filename=result.get("docx_filename") or "iris_deliverable.docx",
            docx_base64=result.get("docx_file_b64") or "",
            execution_notes=result.get("execution_notes")
            or [
                f"Planned {len(tasks)} section task(s)",
                "Generated section content through worker agents",
                "Reviewed and revised the draft through reflection self-check",
                "Assembled markdown and exported a DOCX document",
            ],
            reflection_assessment=result.get("reflection_assessment", ""),
            reflection_notes=result.get("reflection_notes", []),
        )

    async def run_streamed(self, user_query: str) -> AsyncIterator[str]:
        initial_state: AgentState = cast(AgentState, {"query": user_query, "llm": self.llm})
        accumulated: dict = {}

        try:
            async for update in self.graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in update.items():
                    if node_name in ("__start__", "__end__"):
                        continue

                    # Remember which task was active before this node ran
                    prev_active_id = accumulated.get("current_todo_task_id")

                    accumulated.update(node_output)

                    tasks = accumulated.get("tasks", [])
                    active_id = accumulated.get("current_todo_task_id")

                    message = NODE_LABELS.get(node_name, f"Running {node_name}...")
                    if node_name == "todo_worker" and prev_active_id is not None:
                        task = next((t for t in tasks if t.id == prev_active_id), None)
                        if task:
                            message = f"Writing section: {task.section_heading}"
                    elif node_name == "orchestrator" and active_id is None and tasks:
                        message = f"All {len(tasks)} tasks completed"

                    progress = {
                        "type": "progress",
                        "node": node_name,
                        "message": message,
                        "active_task_id": active_id,
                        "tasks": _todo_json(tasks),
                        "assumptions": accumulated.get("assumptions", []),
                        "reflection_assessment": accumulated.get("reflection_assessment", ""),
                        "reflection_notes": accumulated.get("reflection_notes", []),
                        "title": accumulated.get("title", ""),
                        "document_type": accumulated.get("document_type", ""),
                    }

                    yield f"data: {json.dumps(progress)}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
            return

        tasks = accumulated.get("tasks", [])
        response = AgentResponse(
            message=accumulated.get("final_response", "The autonomous agent completed the request."),
            title=accumulated.get("title", "Autonomous Deliverable"),
            document_type=accumulated.get("document_type", "business report"),
            assumptions=accumulated.get("assumptions", []),
            tasks=[
                TodoResponse(
                    id=t.id,
                    task=f"{t.section_heading}: {t.task}",
                    status=t.status.value,
                    result=t.result,
                )
                for t in tasks
            ],
            docx_filename=accumulated.get("docx_filename", "iris_deliverable.docx"),
            docx_base64=accumulated.get("docx_file_b64", ""),
            execution_notes=accumulated.get("execution_notes")
            or [
                f"Planned {len(tasks)} section task(s)",
                "Generated section content through worker agents",
                "Reviewed and revised the draft through reflection self-check",
                "Assembled markdown and exported a DOCX document",
            ],
            reflection_assessment=accumulated.get("reflection_assessment", ""),
            reflection_notes=accumulated.get("reflection_notes", []),
        )
        yield f"data: {json.dumps({'type': 'completed', 'result': response.model_dump()})}\n\n"


def get_iris_agent() -> IrisAgent:
    return IrisAgent()