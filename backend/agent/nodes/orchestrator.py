from agent.agent_state import IrisAgentState as AgentState, TodoStatus


# --- Orchestrator Node ---
def orchestrator(state: AgentState):
    tasks = state.get("tasks", [])
    next_task = next((task for task in tasks if task.status == TodoStatus.PENDING), None)

    if next_task:
        return {"current_todo_task_id": next_task.id}

    return {"current_todo_task_id": None}


# --- Conditional Router ---
def orchestration_router(state: AgentState):
    if state.get("current_todo_task_id") is not None:
        return "worker"
    return "next_phase"