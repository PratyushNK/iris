from typing import cast

from agent.agent_state import DocumentSection, IrisAgentState as AgentState, TodoStatus
from llms.groq import GroqLLM
from llms.llm import LLMConfig, SectionContentSpec


def _build_worker_prompt(state: AgentState, section_heading: str, task_text: str) -> str:
    assumptions = state.get("assumptions", [])
    return (
        f"User request: {state.get('query', '')}\n"
        f"Document title: {state.get('title', 'Business Document')}\n"
        f"Document type: {state.get('document_type', 'business report')}\n"
        f"Audience: {state.get('audience', 'internal stakeholders')}\n"
        f"Tone: {state.get('tone', 'professional and action-oriented')}\n"
        f"Assumptions: {'; '.join(assumptions) if assumptions else 'None'}\n"
        f"Section heading: {section_heading}\n"
        f"Section task: {task_text}\n\n"
        "Generate substantive business content for this section only. "
        "Use clear paragraphs and bullet lists where appropriate. "
        "Do not mention the agent, TODO list, or document generation process."
    )


async def todo_worker(state: AgentState):
    tasks = state.get("tasks", [])
    active_id = state.get("current_todo_task_id")
    llm_obj = state.get("llm")
    if llm_obj is None:
        raise ValueError("Todo worker requires an injected GroqLLM instance")
    llm = cast(GroqLLM, llm_obj)

    active_task = next((task for task in tasks if task.id == active_id), None)
    if active_task is None:
        return {"current_todo_task_id": None}

    section_content = await llm.generate_structured(
        SectionContentSpec,
        prompt=_build_worker_prompt(state, active_task.section_heading, active_task.task),
        system_prompt=(
            "You are a worker agent that drafts one section of a structured business document. "
            "Return only the content for the requested section."
        ),
        llm_config=LLMConfig(
            model="llama-3.1-8b-instant",
            temperature=0.45,
            max_tokens=1200,
            max_retries=2,
        ),
    )

    generated_section = DocumentSection(
        heading=section_content.heading or active_task.section_heading,
        paragraphs=section_content.paragraphs,
        bullets=section_content.bullets,
    )

    updated_tasks = []
    for task in tasks:
        if task.id == active_id:
            updated_tasks.append(
                task.model_copy(
                    update={
                        "status": TodoStatus.COMPLETED,
                        "result": f"Generated section '{generated_section.heading}'.",
                    }
                )
            )
        else:
            updated_tasks.append(task)

    existing_sections = list(state.get("docx_sections", []))
    existing_sections.append(generated_section)

    return {
        "tasks": updated_tasks,
        "docx_sections": existing_sections,
        "current_todo_task_id": None,
    }
