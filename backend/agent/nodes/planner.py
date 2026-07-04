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
            "You are a planning agent that decomposes natural-language business document requests "
            "into a document brief and a TODO list.\n\n"
            "Rules:\n"
            "- Each TODO item must produce ONE section of the final document.\n"
            "- Every task needs a clear section_heading (the heading that will appear in the DOCX).\n"
            "- The task field must describe the specific content to generate for that section.\n"
            "- Choose 3-6 sections if unspecified. Else follow from the user's request, appropriate to the requested document type.\n"
            "- Do NOT create meta tasks about planning, outlining, polishing, or exporting.\n"
            "- Resolve missing details with explicit assumptions in the assumptions list."
        ),
        llm_config=LLMConfig(model="llama-3.1-8b-instant", temperature=0.2, max_tokens=1200, max_retries=2),
    )

    tasks = [
        Todo(
            id=index + 1,
            task=item.task,
            section_heading=item.section_heading,
            status=TodoStatus.PENDING,
            result=None,
        )
        for index, item in enumerate(plan.tasks)
    ]

    return {
        "title": plan.title,
        "document_type": plan.document_type,
        "audience": plan.audience,
        "tone": plan.tone,
        "assumptions": plan.assumptions,
        "tasks": tasks,
        "docx_sections": [],
    }
