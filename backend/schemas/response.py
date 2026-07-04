from pydantic import BaseModel, Field


class TodoResponse(BaseModel):
    id: int
    task: str
    status: str
    result: str | None = None


class AgentResponse(BaseModel):
    message: str
    title: str
    document_type: str
    assumptions: list[str] = Field(default_factory=list)
    tasks: list[TodoResponse] = Field(default_factory=list)
    docx_filename: str
    docx_base64: str
    execution_notes: list[str] = Field(default_factory=list)


class DocxSaveRequest(BaseModel):
    filename: str
    docx_base64: str


class DocxSaveResponse(BaseModel):
    filename: str
    saved_path: str
    download_url: str
    size_bytes: int