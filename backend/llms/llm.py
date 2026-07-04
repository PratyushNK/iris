from __future__ import annotations

from typing import Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel, Field


class SectionContentSpec(BaseModel):
    heading: str
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)


class PlanSpec(BaseModel):
    title: str
    document_type: str
    audience: str
    tone: str
    assumptions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    # Each task string: "Section Heading: task description"


class ResponseSpec(BaseModel):
    message: str
    execution_notes: list[str] = Field(default_factory=list)


class ReflectionSpec(BaseModel):
    overall_assessment: str
    reflection_notes: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    # Each section string: heading on first line, then paragraphs and bullets (starting with "- ") on following lines


class LLMConfig(BaseModel):
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    max_retries: int = 0


T = TypeVar("T", bound=BaseModel)

@runtime_checkable
class LLM(Protocol):

    async def generate(
        self,
        prompt: str, 
        system_prompt: str | None,
        llm_config: LLMConfig,
    ) -> str:
        
        """Generate response from prompt"""
        ... 


    async def generate_structured(
        self,
        schema: Type[T], 
        prompt: str, 
        system_prompt: str | None,
        llm_config: LLMConfig,
    ) -> T : 
        
        """Generate response from prompt"""
        ... 



def _normalize_request(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def _infer_document_type(prompt: str) -> str:
    lowered = prompt.lower()
    if any(keyword in lowered for keyword in ("meeting", "minutes")):
        return "meeting minutes"
    if any(keyword in lowered for keyword in ("sop", "standard operating procedure", "procedure")):
        return "standard operating procedure"
    if any(keyword in lowered for keyword in ("design", "architecture", "technical")):
        return "technical design"
    if any(keyword in lowered for keyword in ("proposal", "pitch")):
        return "business proposal"
    if any(keyword in lowered for keyword in ("spec", "specification", "product")):
        return "product specification"
    if any(keyword in lowered for keyword in ("plan", "roadmap", "strategy")):
        return "project plan"
    if any(keyword in lowered for keyword in ("report", "analysis", "review")):
        return "business report"
    return "business report"


def _infer_audience(prompt: str) -> str:
    lowered = prompt.lower()
    if any(keyword in lowered for keyword in ("executive", "board", "leadership")):
        return "executive stakeholders"
    if any(keyword in lowered for keyword in ("customer", "client")):
        return "external stakeholders"
    if any(keyword in lowered for keyword in ("engineering", "developer", "technical")):
        return "cross-functional delivery team"
    return "internal stakeholders"


def _infer_tone(prompt: str) -> str:
    lowered = prompt.lower()
    if any(keyword in lowered for keyword in ("formal", "executive", "board")):
        return "formal and concise"
    if any(keyword in lowered for keyword in ("technical", "design")):
        return "clear and implementation-focused"
    return "professional and action-oriented"


def _derive_title(prompt: str) -> str:
    cleaned_prompt = _normalize_request(prompt)
    words = cleaned_prompt.split()
    if len(words) <= 12:
        headline = cleaned_prompt
    else:
        headline = " ".join(words[:12])

    return headline[:1].upper() + headline[1:]


def _section_tasks(prompt: str) -> list[str]:
    lowered = prompt.lower()
    tasks = [
        "Executive Summary: Write an executive summary that states the purpose, key outcomes, and recommended next steps.",
        "Background and Objectives: Describe the background, objectives, and scope of the deliverable based on the user request.",
        "Main Content: Provide the main body content with structured recommendations, requirements, or plan details.",
        "Conclusion and Next Steps: Summarize conclusions, ownership, and immediate follow-up actions.",
    ]

    if any(keyword in lowered for keyword in ("meeting", "minutes")):
        tasks = [
            "Meeting Details: Capture meeting purpose, date context, attendees, and facilitator.",
            "Discussion Summary: Summarize the key discussion points and decisions made.",
            "Action Items: List action items with owners and due dates.",
        ]
    elif any(keyword in lowered for keyword in ("timeline", "schedule", "launch", "implementation", "plan", "roadmap")):
        tasks.insert(
            2,
            "Timeline and Milestones: Lay out phases, milestones, and a realistic timeline with dependencies.",
        )
    elif any(keyword in lowered for keyword in ("sop", "standard operating procedure", "procedure")):
        tasks = [
            "Purpose and Scope: State the procedure purpose, scope, and responsible roles.",
            "Prerequisites: Document prerequisites, inputs, and required tools or systems.",
            "Procedure Steps: Provide numbered step-by-step instructions for execution.",
            "Quality and Escalation: List quality checks, exceptions, and escalation paths.",
        ]

    if any(keyword in lowered for keyword in ("compare", "evaluate", "recommend", "choose", "tradeoff")):
        tasks.append(
            "Analysis and Recommendation: Compare options, explain tradeoffs, and provide a clear recommendation.",
        )

    if any(keyword in lowered for keyword in ("timeline", "schedule", "launch", "implementation", "risk")):
        if not any(task.startswith("Timeline and Milestones:") for task in tasks):
            tasks.append(
                "Risks and Mitigations: Identify execution risks, mitigations, and open dependencies.",
            )

    return tasks


def _mock_section_content(prompt: str, section_heading: str, task: str) -> SectionContentSpec:
    request = _normalize_request(prompt)
    return SectionContentSpec(
        heading=section_heading,
        paragraphs=[
            (
                f"This section covers {section_heading.lower()} for the requested { _infer_document_type(prompt) } "
                f"derived from: {request}."
            ),
            f"Task focus: {task}",
        ],
        bullets=[
            "Key point aligned to the user request",
            "Practical recommendation or requirement",
            "Clear next step for stakeholders",
        ],
    )


def _build_plan(prompt: str) -> PlanSpec:
    document_type = _infer_document_type(prompt)
    title = _derive_title(prompt)
    audience = _infer_audience(prompt)
    tone = _infer_tone(prompt)
    assumptions = [
        "The request is interpreted as an internal business deliverable unless stated otherwise.",
        "Where the prompt is underspecified, the agent makes explicit assumptions to keep execution moving.",
    ]

    if any(keyword in prompt.lower() for keyword in ("ambiguous", "missing", "conflicting")):
        assumptions.append("Conflicting or missing constraints are resolved using practical business defaults.")

    tasks = _section_tasks(prompt)

    return PlanSpec(
        title=title,
        document_type=document_type,
        audience=audience,
        tone=tone,
        assumptions=assumptions,
        tasks=tasks,
    )


def _mock_reflection(prompt: str) -> ReflectionSpec:
    sections: list[str] = []
    notes: list[str] = []
    for line in prompt.splitlines():
        if line.startswith("Section: "):
            heading = line.split(":", 1)[1].strip()
            sections.append(
                f"{heading}\nRevised content for {heading} after self-check review.\n- Clarified scope and actionable next steps"
            )
    if not sections:
        sections.append(
            "Document Content\nRevised content after self-check review.\n- Improved clarity and alignment to the user request"
        )
    notes = [
        "Verified each section addresses the original user request.",
        "Expanded thin sections with concrete recommendations.",
        "Aligned tone and terminology across sections.",
    ]
    return ReflectionSpec(
        overall_assessment="Document reviewed and revised for completeness, clarity, and request alignment.",
        reflection_notes=notes,
        sections=sections,
    )


def _build_response(prompt: str, title: str = "", document_type: str = "business report") -> ResponseSpec:
    request = _normalize_request(prompt)
    headline = title or _derive_title(prompt)
    return ResponseSpec(
        message=(
            f"I prepared '{headline}', a {document_type} based on your request. "
            f"The Word document includes structured sections covering the requested scope, "
            f"with explicit assumptions where details were missing. "
            f"You can download the DOCX for review and handoff."
        ),
        execution_notes=[
            "Decomposed the request into document section tasks",
            "Generated section content through worker agents",
            "Reviewed and revised the draft through reflection self-check",
            "Assembled markdown and exported a DOCX deliverable",
        ],
    )


class MockLLM:
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None,
        llm_config: LLMConfig,
    ) -> str:
        response = _build_response(prompt)
        return response.message

    async def generate_structured(
        self,
        schema: Type[T],
        prompt: str,
        system_prompt: str | None,
        llm_config: LLMConfig,
    ) -> T:
        if issubclass(schema, PlanSpec):
            return schema.model_validate(_build_plan(prompt).model_dump())

        if issubclass(schema, SectionContentSpec):
            section_heading = "Section"
            task_text = "Generate section content"
            for line in prompt.splitlines():
                if line.startswith("Section heading:"):
                    section_heading = line.split(":", 1)[1].strip()
                if line.startswith("Section task:"):
                    task_text = line.split(":", 1)[1].strip()
            return schema.model_validate(
                _mock_section_content(prompt, section_heading, task_text).model_dump()
            )

        if issubclass(schema, ReflectionSpec):
            return schema.model_validate(_mock_reflection(prompt).model_dump())

        if issubclass(schema, ResponseSpec):
            return schema.model_validate(_build_response(prompt).model_dump())

        return schema.model_validate({})
