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
        "Rewrite each section — tighten prose, remove redundancy, keep it brief. "
        "Keep every substantive point from the original; only add, rephrase, or reorganize. "
        "Do NOT summarize, truncate, or replace content with review notes. "
        "Each section string must contain the complete revised content, not a description of changes. "
        "Do not mention the agent, TODO list, or review process inside section content. "
        "Aim for 2-4 sentences or 2-4 bullet points per section."
    )


def _parse_section_string(text: str) -> DocumentSection:
    lines = text.strip().split("\n")
    heading = lines[0].strip() if lines else "Untitled Section"
    paragraphs: list[str] = []
    bullets: list[str] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            bullets.append(line[2:])
        else:
            paragraphs.append(line)
    return DocumentSection(heading=heading, paragraphs=paragraphs, bullets=bullets)


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
            "You are a reviewer agent performing reflection and self-check on a concise business document draft. "
            "Improve each section while preserving its brevity — keep all original content, tighten prose, "
            "remove redundancy. The final document must fit 1.5–2 pages total.\n\n"
            "CRITICAL: Each section string must contain the FULL rewritten content. "
            "Do NOT return review notes, summaries, or placeholders. "
            "Format: heading on the first line, then paragraphs and bullet points "
            "(starting with '- ') on following lines."
        ),
        llm_config=LLMConfig(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=2048, max_retries=3),
    )

    revised_sections = [_parse_section_string(s) for s in review.sections if s.strip()]

    if not revised_sections:
        revised_sections = draft_sections
        reflection_notes = list(review.reflection_notes) + [
            "Reviewer returned no revised sections; kept the worker draft unchanged.",
        ]
    else:
        covered: set[str] = set()
        merged: list[DocumentSection] = []
        for revised in revised_sections:
            merged.append(revised)
            covered.add(revised.heading.lower())
        for draft in draft_sections:
            if draft.heading.lower() not in covered:
                merged.append(draft)
        revised_sections = merged
        reflection_notes = review.reflection_notes

    return {
        "docx_sections": revised_sections,
        "reflection_assessment": review.overall_assessment,
        "reflection_notes": reflection_notes,
    }
