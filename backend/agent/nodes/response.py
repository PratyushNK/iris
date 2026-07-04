from typing import cast

from agent.agent_state import IrisAgentState as AgentState, TodoStatus
from llms.groq import GroqLLM
from llms.llm import LLMConfig, ResponseSpec


# --- Chat Response Node ---
async def chat_response(state: AgentState):
    title = state.get("title", "Autonomous Deliverable")
    document_type = state.get("document_type", "business report")
    tasks = state.get("tasks", [])
    completed_tasks = [task for task in tasks if task.status == TodoStatus.COMPLETED]
    assumptions = state.get("assumptions", [])
    docx_filename = state.get("docx_filename") or f"{title.lower().replace(' ', '_')}.docx"
    llm_obj = state.get("llm")
    if llm_obj is None:
        raise ValueError("Response node requires an injected GroqLLM instance")
    llm = cast(GroqLLM, llm_obj)

    task_lines = [f"- Task {task.id}: {task.task} -> {task.result or 'completed'}" for task in completed_tasks]
    prompt = (
        f"User request: {state.get('query', '')}\n"
        f"Document title: {title}\n"
        f"Document type: {document_type}\n"
        f"Assumptions: {'; '.join(assumptions) if assumptions else 'None'}\n"
        f"Completed tasks:\n" + "\n".join(task_lines) + "\n\n"
        f"Write a polished 2-4 sentence final response for the user that confirms the deliverable, "
        f"mentions the generated document filename '{docx_filename}', and summarizes the execution at a high level."
    )

    response = await llm.generate_structured(
        ResponseSpec,
        prompt=prompt,
        system_prompt="You are the final response writer for an autonomous business document agent.",
        llm_config=LLMConfig(model="llama-3.1-8b-instant", temperature=0.35, max_tokens=500, max_retries=2),
    )

    return {
        "final_response": response.message
    }
