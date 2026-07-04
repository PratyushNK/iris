from __future__ import annotations

import base64
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse

from agent.factory import IrisAgent, get_iris_agent
from schemas.request import UserRequest
from schemas.response import AgentResponse, DocxSaveRequest, DocxSaveResponse


app = FastAPI(title="Iris Autonomous Agent API", version="0.1.0")
router = APIRouter()
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)



@router.post(
    "/agent",
    status_code=status.HTTP_200_OK,
    response_model=AgentResponse
)
async def user_request(
    user_request: UserRequest,
    agent: IrisAgent = Depends(get_iris_agent)
) -> AgentResponse:
    return await agent.run(user_request.request)


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/agent/stream")
async def user_request_stream(
    user_request: UserRequest,
    agent: IrisAgent = Depends(get_iris_agent),
):
    return StreamingResponse(
        agent.run_streamed(user_request.request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/documents/save", response_model=DocxSaveResponse)
async def save_document(payload: DocxSaveRequest) -> DocxSaveResponse:
    try:
        docx_bytes = base64.b64decode(payload.docx_base64)
    except Exception as exc:  # pragma: no cover - guardrail for malformed payloads
        raise HTTPException(status_code=400, detail="Invalid DOCX payload") from exc

    filename = Path(payload.filename).name
    if not filename.lower().endswith(".docx"):
        filename = f"{filename}.docx"

    saved_path = ARTIFACTS_DIR / filename
    saved_path.write_bytes(docx_bytes)

    return DocxSaveResponse(
        filename=filename,
        saved_path=str(saved_path),
        download_url=f"/documents/{filename}",
        size_bytes=len(docx_bytes),
    )


@router.get("/documents/{filename}")
async def download_document(filename: str) -> FileResponse:
    resolved_path = (ARTIFACTS_DIR / Path(filename).name).resolve()
    if ARTIFACTS_DIR not in resolved_path.parents and resolved_path != ARTIFACTS_DIR:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(
        path=resolved_path,
        filename=resolved_path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )



app.include_router(router)