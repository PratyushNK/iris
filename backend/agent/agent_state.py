from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field

from llms.llm import LLM


class DocumentSection(BaseModel):
    heading: str
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)


class TodoStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PENDING = "pending"
    BLOCKED = "blocked"


class Todo(BaseModel):
    id: int
    task: str
    status: TodoStatus
    result: str | None = None


class IrisAgentState(TypedDict, total=False):
    query: str
    llm: LLM
    title: str
    document_type: str
    audience: str
    tone: str
    assumptions: list[str]
    tasks: list[Todo]
    current_todo_task_id: int | None
    docx_sections: list[DocumentSection]
    docx_content: str
    docx_file_b64: str | None
    docx_filename: str | None
    final_response: str

