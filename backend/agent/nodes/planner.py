from typing import cast

from agent.agent_state import IrisAgentState as AgentState, Todo, TodoStatus
from llms.groq import GroqLLM
from llms.llm import LLMConfig, PlanSpec


# --- Planner Node ---
async def planner(state: AgentState):
    user_query = state.get("query", "")
    llm_obj = state.get("llm")
    if llm_obj is None:
        raise ValueError("Planner requires an injected GroqLLM instance")
    llm = cast(GroqLLM, llm_obj)
    plan = await llm.generate_structured(
        PlanSpec,
        prompt=user_query,
        system_prompt=(
            "You are a planning agent that decomposes natural-language business requests into a small, "
            "actionable TODO list and a document brief."
        ),
        llm_config=LLMConfig(model="llama-3.1-8b-instant", temperature=0.2, max_tokens=1200, max_retries=2),
    )

    tasks = [
        Todo(id=index + 1, task=item.task, status=TodoStatus.PENDING, result=None)
        for index, item in enumerate(plan.tasks)
    ]

    return {
        "title": plan.title,
        "document_type": plan.document_type,
        "audience": plan.audience,
        "tone": plan.tone,
        "assumptions": plan.assumptions,
        "tasks": tasks,
    }
