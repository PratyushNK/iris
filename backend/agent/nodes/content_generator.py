from agent.agent_state import (
    DocumentSection,
    IrisAgentState as AgentState,
    TodoStatus,
)


def _completed_task_lines(state: AgentState) -> list[str]:
    tasks = state.get("tasks", [])
    return [
        f"Task {task.id}: {task.result or task.task}"
        for task in tasks
        if task.status == TodoStatus.COMPLETED
    ]


# --- Docx Content Generator Node ---
def docx_content_generator(state: AgentState):
    title = state.get("title", "Autonomous Business Document")
    document_type = state.get("document_type", "business report")
    audience = state.get("audience", "internal stakeholders")
    tone = state.get("tone", "professional and action-oriented")
    assumptions = state.get("assumptions", [])
    tasks = state.get("tasks", [])

    completed_lines = _completed_task_lines(state)
    planned_lines = [f"{task.id}. {task.task}" for task in tasks]

    sections = [
        DocumentSection(
            heading="Executive Summary",
            paragraphs=[
                (
                    f"This {document_type} was produced autonomously from the user request and tailored for {audience}. "
                    f"The document uses a {tone} style and resolves missing details with explicit assumptions."
                )
            ],
        ),
        DocumentSection(heading="Autonomous Plan", bullets=planned_lines),
        DocumentSection(heading="Completed Work", bullets=completed_lines),
    ]

    if assumptions:
        sections.append(DocumentSection(heading="Assumptions", bullets=assumptions))

    docx_content = (
        f"# {title}\n\n"
        f"Document type: {document_type}\n"
        f"Audience: {audience}\n"
        f"Tone: {tone}\n\n"
        f"## Planned Tasks\n"
        + "\n".join(f"- {line}" for line in planned_lines)
        + (
            "\n\n## Completed Work\n" + "\n".join(f"- {line}" for line in completed_lines)
            if completed_lines
            else ""
        )
    )

    return {
        "docx_content": docx_content,
        "docx_sections": sections,
    }
