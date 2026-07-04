from typing import cast

from agent.agent_state import IrisAgentState as AgentState
from llms.groq import GroqLLM
from llms.llm import LLMConfig, ResponseSpec


def _section_preview(state: AgentState) -> str:
    sections = state.get("docx_sections", [])
    if not sections:
        return "No section content was generated."

    preview_lines = []
    for section in sections:
        preview_lines.append(f"- {section.heading}")
        for paragraph in section.paragraphs[:2]:
            preview_lines.append(f"  {paragraph}")
        for bullet in section.bullets[:3]:
            preview_lines.append(f"  - {bullet}")
    return "\n".join(preview_lines)


# --- Chat Response Node ---
async def chat_response(state: AgentState):
    title = state.get("title", "Autonomous Deliverable")
    document_type = state.get("document_type", "business report")
    assumptions = state.get("assumptions", [])
    docx_filename = state.get("docx_filename") or f"{title.lower().replace(' ', '_')}.docx"
    llm_obj = state.get("llm")
    if llm_obj is None:
        raise ValueError("Response node requires an injected GroqLLM instance")
    llm = cast(GroqLLM, llm_obj)

    prompt = (
        f"User request: {state.get('query', '')}\n"
        f"Document title: {title}\n"
        f"Document type: {document_type}\n"
        f"Assumptions: {'; '.join(assumptions) if assumptions else 'None'}\n"
        f"Generated filename: {docx_filename}\n\n"
        f"Document section preview:\n{_section_preview(state)}\n\n"
        "Write ONE consolidated paragraph for the user. "
        "Summarize what was delivered, the most important points covered, "
        "and confirm the DOCX is ready for download. "
        "Do not use bullet points or headings."
    )

    response = await llm.generate_structured(
        ResponseSpec,
        prompt=prompt,
        system_prompt=(
            "You are the final response writer for an autonomous business document agent. "
            "Return a single polished paragraph in message."
        ),
        llm_config=LLMConfig(model="llama-3.1-8b-instant", temperature=0.35, max_tokens=500, max_retries=2),
    )

    return {
        "final_response": response.message,
        "execution_notes": response.execution_notes,
    }
