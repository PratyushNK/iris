from typing import cast

from agent.agent_state import DocumentSection, IrisAgentState as AgentState
from llms.groq import GroqLLM
from llms.llm import LLMConfig, ReflectionSpec


def _format_sections_for_review(sections: list[DocumentSection]) -> str:
    blocks: list[str] = []
    for section in sections:
        blocks.append(f"Section: {section.heading}")
        for paragraph in section.paragraphs:
            blocks.append(f"Paragraph: {paragraph}")
        for bullet in section.bullets:
            blocks.append(f"Bullet: {bullet}")
        blocks.append("")
    return "\n".join(blocks).strip()


def _build_reflection_prompt(state: AgentState) -> str:
    sections = state.get("docx_sections", [])
    assumptions = state.get("assumptions", [])
    return (
        f"User request: {state.get('query', '')}\n"
        f"Document title: {state.get('title', 'Business Document')}\n"
        f"Document type: {state.get('document_type', 'business report')}\n"
        f"Audience: {state.get('audience', 'internal stakeholders')}\n"
        f"Tone: {state.get('tone', 'professional and action-oriented')}\n"
        f"Assumptions: {'; '.join(assumptions) if assumptions else 'None'}\n\n"
        "Draft sections to review:\n"
        f"{_format_sections_for_review(sections)}\n\n"
        "Review the draft against the user request and document brief. "
        "Return reflection notes describing issues found and a fully revised sections list. "
        "Each revised section must keep the same heading unless a merge or split is necessary. "
        "Do not mention the agent, TODO list, or review process inside section content."
    )


def _to_document_sections(reflection: ReflectionSpec) -> list[DocumentSection]:
    return [
        DocumentSection(
            heading=section.heading,
            paragraphs=section.paragraphs,
            bullets=section.bullets,
        )
        for section in reflection.sections
        if section.heading.strip()
    ]


async def reflection(state: AgentState):
    llm_obj = state.get("llm")
    if llm_obj is None:
        raise ValueError("Reflection node requires an injected GroqLLM instance")
    llm = cast(GroqLLM, llm_obj)

    draft_sections = list(state.get("docx_sections", []))
    if not draft_sections:
        return {
            "reflection_assessment": "No draft sections were available to review.",
            "reflection_notes": ["Worker agents did not produce any section content."],
        }

    review = await llm.generate_structured(
        ReflectionSpec,
        prompt=_build_reflection_prompt(state),
        system_prompt=(
            "You are a reviewer agent performing reflection and self-check on a business document draft. "
            "Identify gaps, weak sections, and misalignment with the request, then return improved sections."
        ),
        llm_config=LLMConfig(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=2500, max_retries=2),
    )

    revised_sections = _to_document_sections(review)
    if not revised_sections:
        revised_sections = draft_sections
        reflection_notes = list(review.reflection_notes) + [
            "Reviewer returned no revised sections; kept the worker draft unchanged.",
        ]
    else:
        reflection_notes = review.reflection_notes

    return {
        "docx_sections": revised_sections,
        "reflection_assessment": review.overall_assessment,
        "reflection_notes": reflection_notes,
    }
