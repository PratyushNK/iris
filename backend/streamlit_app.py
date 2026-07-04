from __future__ import annotations

import base64
import json
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import streamlit as st


st.set_page_config(page_title="Iris Autonomous Agent Demo", layout="wide")
st.title("Iris Autonomous Agent Demo")
st.caption("Run a request, inspect the autonomous TODO list, and download the generated DOCX.")

API_BASE_URL = st.sidebar.text_input("Backend base URL", value="http://localhost:8000")


def call_api(path: str, payload: dict | None = None, method: str = "GET") -> dict:
    url = f"{API_BASE_URL.rstrip('/')}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib_request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib_request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach backend: {exc.reason}") from exc


NODE_ICONS: dict[str, str] = {
    "planner": "\U0001f4cb",
    "orchestrator": "\U0001f504",
    "todo_worker": "\u270d\ufe0f",
    "reflection": "\U0001f50d",
    "docx_content_generator": "\U0001f4dd",
    "generate_report_node": "\U0001f4c4",
    "chat_response": "\U0001f4ac",
}


def _format_progress(event: dict) -> str:
    node = event.get("node", "")
    message = event.get("message", "")
    tasks = event.get("tasks", [])
    active_id = event.get("active_task_id")

    icon = NODE_ICONS.get(node, "\u2699\ufe0f")
    parts: list[str] = [f"{icon} **{message}**"]

    if tasks:
        completed = sum(1 for t in tasks if t["status"] == "completed")
        parts.append(f"Tasks: {completed}/{len(tasks)} completed")

    if active_id is not None:
        active = next((t for t in tasks if t["id"] == active_id), None)
        if active:
            parts.append(f"Current: {active['task']}")

    # Two trailing spaces + newline = visible line break in markdown
    return "  \n".join(parts) + "\n\n"


def read_sse_events(request_text: str):
    """Generate SSE event dicts from the /agent/stream endpoint."""
    url = f"{API_BASE_URL.rstrip('/')}/agent/stream"
    body = json.dumps({"request": request_text}).encode("utf-8")
    req = urllib_request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib_request.urlopen(req, timeout=300) as response:
        data_line: str | None = None
        for raw_line in response:
            text = raw_line.decode("utf-8").rstrip("\r\n")
            if text.startswith("data: "):
                data_line = text[6:]
            elif not text and data_line is not None:
                yield json.loads(data_line)
                data_line = None


def agent_streamer(request_text: str):
    """Generator for st.write_stream — yields progress markdown strings."""
    try:
        for event in read_sse_events(request_text):
            event_type = event.get("type")

            if event_type == "progress":
                yield _format_progress(event)

            elif event_type == "completed":
                st.session_state["agent_response"] = event["result"]
                yield "\U00002705 **Agent complete!**\n\n"
                return

            elif event_type == "error":
                yield f"\u274c **Agent error: {event['error']}**\n\n"
                return

    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        yield f"\u274c **API error {exc.code}: {detail}**\n\n"
    except URLError as exc:
        yield f"\u274c **Connection failed: {exc.reason}**\n\n"
    except Exception as exc:
        yield f"\u274c **Error: {exc}**\n\n"


default_request = (
    "Create a project plan for launching a customer support knowledge base "
    "with a 2-week timeline and limited details."
)

request_text = st.text_area("User request", value=default_request, height=140)
run_demo = st.button("Run agent", type="primary")

if run_demo and request_text.strip():
    st.session_state.pop("agent_response", None)
    st.session_state.pop("save_response", None)
    st.session_state.pop("stream_error", None)

    st.subheader("Live progress")
    st.write_stream(agent_streamer(request_text))

    agent_response = st.session_state.get("agent_response")
    if agent_response and agent_response.get("docx_base64"):
        try:
            save_response = call_api(
                "/documents/save",
                {
                    "filename": agent_response["docx_filename"],
                    "docx_base64": agent_response["docx_base64"],
                },
                method="POST",
            )
            st.session_state["save_response"] = save_response
        except Exception as exc:
            st.error(f"Failed to save document: {exc}")


agent_response = st.session_state.get("agent_response")
save_response = st.session_state.get("save_response")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("Agent output")
    if agent_response:
        st.markdown(f"### {agent_response['title']}")
        st.write(agent_response["message"])

        st.markdown("#### Assumptions")
        for assumption in agent_response.get("assumptions", []):
            st.write(f"- {assumption}")

        st.markdown("#### Execution notes")
        for note in agent_response.get("execution_notes", []):
            st.write(f"- {note}")

        reflection_notes = agent_response.get("reflection_notes", [])
        if reflection_notes or agent_response.get("reflection_assessment"):
            st.markdown("#### Reviewer reflection")
            if agent_response.get("reflection_assessment"):
                st.write(agent_response["reflection_assessment"])
            for note in reflection_notes:
                st.write(f"- {note}")
    else:
        st.info("Run the agent to see the generated response summary.")

with right:
    st.subheader("Task trace")
    if agent_response:
        task_rows = agent_response.get("tasks", [])
        for task in task_rows:
            status = task["status"].replace("_", " ").title()
            st.markdown(f"**Task {task['id']} \u2014 {status}**")
            st.write(task["task"])
            if task.get("result"):
                st.caption(task["result"])
            st.divider()
    else:
        st.info("The TODO list will appear here once the agent runs.")


st.subheader("Generated document")
if agent_response and save_response:
    st.success(f"Saved: {save_response['filename']} ({save_response['size_bytes']} bytes)")
    st.download_button(
        label="Download DOCX",
        data=base64.b64decode(agent_response["docx_base64"]),
        file_name=save_response["filename"],
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    st.markdown(
        f"[Open backend file endpoint]"
        f"({API_BASE_URL.rstrip('/')}{save_response['download_url']})"
    )
elif agent_response:
    st.download_button(
        label="Download DOCX",
        data=base64.b64decode(agent_response["docx_base64"]),
        file_name=agent_response["docx_filename"],
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
else:
    st.info("The DOCX download will appear after the agent completes.")
