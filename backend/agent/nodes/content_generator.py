from agent.agent_state import DocumentSection, IrisAgentState as AgentState


def _section_to_markdown(section: DocumentSection) -> str:
    lines = [f"## {section.heading}", ""]
    for paragraph in section.paragraphs:
        lines.append(paragraph)
        lines.append("")
    for bullet in section.bullets:
        lines.append(f"- {bullet}")
    if section.bullets:
        lines.append("")
    return "\n".join(lines).rstrip()


# --- Docx Content Generator Node ---
def docx_content_generator(state: AgentState):
    title = state.get("title", "Autonomous Business Document")
    document_type = state.get("document_type", "business report")
    audience = state.get("audience", "internal stakeholders")
    tone = state.get("tone", "professional and action-oriented")
    assumptions = state.get("assumptions", [])
    sections = list(state.get("docx_sections", []))

    if assumptions and not any(section.heading.lower() == "assumptions" for section in sections):
        sections.append(
            DocumentSection(
                heading="Assumptions",
                bullets=assumptions,
            )
        )

    markdown_parts = [
        f"# {title}",
        "",
        f"Document type: {document_type}",
        f"Audience: {audience}",
        f"Tone: {tone}",
        "",
    ]
    for section in sections:
        markdown_parts.append(_section_to_markdown(section))
        markdown_parts.append("")

    docx_content = "\n".join(markdown_parts).strip()

    return {
        "docx_content": docx_content,
        "docx_sections": sections,
    }
