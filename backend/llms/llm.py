from __future__ import annotations

from typing import Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel, Field


class PlannedTask(BaseModel):
    task: str
    rationale: str | None = None


class PlanSpec(BaseModel):
    title: str
    document_type: str
    audience: str
    tone: str
    assumptions: list[str] = Field(default_factory=list)
    tasks: list[PlannedTask] = Field(default_factory=list)


class ResponseSpec(BaseModel):
    message: str
    execution_notes: list[str] = Field(default_factory=list)


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


def _derive_title(prompt: str, document_type: str) -> str:
    cleaned_prompt = _normalize_request(prompt)
    words = cleaned_prompt.split()
    if len(words) <= 12:
        headline = cleaned_prompt
    else:
        headline = " ".join(words[:12])

    return headline[:1].upper() + headline[1:]


def _base_tasks(prompt: str) -> list[str]:
    lowered = prompt.lower()
    tasks = [
        "Interpret the request and lock the deliverable scope",
        "Build a structured outline and set reasonable assumptions",
        "Draft the business content with clear sections and recommendations",
        "Polish the final document and prepare the handoff summary",
    ]

    if any(keyword in lowered for keyword in ("ambiguous", "missing", "conflicting", "decide", "multi-step")):
        tasks.insert(1, "Resolve missing details by making explicit working assumptions")

    if any(keyword in lowered for keyword in ("compare", "evaluate", "recommend", "choose")):
        tasks.append("Add decision criteria, tradeoffs, and a recommendation")

    if any(keyword in lowered for keyword in ("timeline", "schedule", "launch", "implementation")):
        tasks.append("Add a realistic timeline and execution risks")

    return tasks


def _build_plan(prompt: str) -> PlanSpec:
    document_type = _infer_document_type(prompt)
    title = _derive_title(prompt, document_type)
    audience = _infer_audience(prompt)
    tone = _infer_tone(prompt)
    assumptions = [
        "The request is interpreted as an internal business deliverable unless stated otherwise.",
        "Where the prompt is underspecified, the agent makes explicit assumptions to keep execution moving.",
    ]

    if any(keyword in prompt.lower() for keyword in ("ambiguous", "missing", "conflicting")):
        assumptions.append("Conflicting or missing constraints are resolved using practical business defaults.")

    tasks = [PlannedTask(task=task) for task in _base_tasks(prompt)]

    return PlanSpec(
        title=title,
        document_type=document_type,
        audience=audience,
        tone=tone,
        assumptions=assumptions,
        tasks=tasks,
    )


def _build_response(prompt: str) -> ResponseSpec:
    request = _normalize_request(prompt)
    return ResponseSpec(
        message=f"The agent completed the requested deliverable for: {request}",
        execution_notes=[
            "Autonomous planning completed",
            "Task execution trace assembled",
            "DOCX document generated in memory",
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

        if issubclass(schema, ResponseSpec):
            return schema.model_validate(_build_response(prompt).model_dump())

        return schema.model_validate({})
