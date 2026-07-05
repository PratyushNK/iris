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
            "- The tasks array contains strings. Each string is formatted as: 'Section Heading: task description'\n"
            "- Each task must produce ONE section of the final document.\n"
            "- Choose 3-6 sections if unspecified, appropriate to the requested document type.\n"
            "- Make sure each section is unique and doesn't have redundant goals.\n"
            "- Do NOT create meta tasks about planning, outlining, polishing, or exporting.\n"
            "- Resolve missing details with explicit assumptions in the assumptions list.\n"
            "- If you find the user query could be ambiguous or even conflicting, try to accomodate both ideas first if possible. If not, then pick the rationally safe option while being explicit in the assumption with a small reasoning.\n"
            "- CRITICAL: Embed a tight word budget in each task description (e.g. 'Write 3-4 sentences on…' or 'List 4-5 bullet points for…'). The final document must fit 1.5–2 pages total (~350–500 words), so each section must be brief."
        ),
        llm_config=LLMConfig(model="llama-3.1-8b-instant", temperature=0.2, max_tokens=800, max_retries=2),
    )

    tasks = []
    for index, item in enumerate(plan.tasks):
        if ":" in item:
            section_heading, task_text = item.split(":", 1)
        else:
            section_heading = "Untitled Section"
            task_text = item
        tasks.append(Todo(
            id=index + 1,
            task=task_text.strip(),
            section_heading=section_heading.strip(),
            status=TodoStatus.PENDING,
            result=None,
        ))

    return {
        "title": plan.title,
        "document_type": plan.document_type,
        "audience": plan.audience,
        "tone": plan.tone,
        "assumptions": plan.assumptions,
        "tasks": tasks,
        "docx_sections": [],
    }
