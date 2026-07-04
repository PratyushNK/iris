from __future__ import annotations

import os
from pathlib import Path
from typing import cast

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
def get_iris_agent() -> IrisAgent:
    return IrisAgent()