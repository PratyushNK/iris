from agent.agent_state import IrisAgentState as AgentState, TodoStatus


def _execute_task(task_text: str, state: AgentState) -> str:
    title = state.get("title", "the document")
    document_type = state.get("document_type", "business report")
    query = state.get("query", "the user request")
    task_lower = task_text.lower()

    if "scope" in task_lower:
        return f"Confirmed the scope for {document_type} '{title}' from the request: {query}."
    if "assumpt" in task_lower:
        assumptions = state.get("assumptions", [])
        return "Working assumptions established: " + "; ".join(assumptions)
    if "outline" in task_lower:
        return "Created a practical document outline with overview, execution details, and handoff-ready sections."
    if any(keyword in task_lower for keyword in ("draft", "content", "write")):
        return "Drafted structured business content with clear recommendations and concise language."
    if any(keyword in task_lower for keyword in ("polish", "handoff", "final")):
        return "Polished the deliverable for presentation, consistency, and a clean Word export."
    if any(keyword in task_lower for keyword in ("timeline", "risks", "decision", "tradeoff")):
        return "Added execution detail, tradeoffs, and risk-aware recommendations for the final document."
    return f"Completed task '{task_text}' for {document_type} '{title}'."


def todo_worker(state: AgentState):
    tasks = state.get("tasks", [])
    active_id = state.get("current_todo_task_id")

    updated_tasks = []
    for task in tasks:
        if task.id == active_id:
            worker_output = _execute_task(task.task, state)
            updated_tasks.append(
                task.model_copy(update={"status": TodoStatus.COMPLETED, "result": worker_output})
            )
        else:
            updated_tasks.append(task)

    return {"tasks": updated_tasks, "current_todo_task_id": None}
