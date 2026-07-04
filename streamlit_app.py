from __future__ import annotations

import json
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import requests
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
        with urllib_request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach backend: {exc.reason}") from exc


def fetch_document_bytes(download_url: str) -> bytes:
    url = f"{API_BASE_URL.rstrip('/')}{download_url}"
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.content


default_request = (
    "Create a project plan for launching a customer support knowledge base with a 2-week timeline and limited details."
)

request_text = st.text_area("User request", value=default_request, height=140)
run_demo = st.button("Run agent", type="primary")

if run_demo and request_text.strip():
    try:
        with st.spinner("Running autonomous agent..."):
            agent_response = call_api("/agent", {"request": request_text}, method="POST")
            save_response = call_api(
                "/documents/save",
                {
                    "filename": agent_response["docx_filename"],
                    "docx_base64": agent_response["docx_base64"],
                },
                method="POST",
            )
            doc_bytes = fetch_document_bytes(save_response["download_url"])

        st.session_state["agent_response"] = agent_response
        st.session_state["save_response"] = save_response
        st.session_state["doc_bytes"] = doc_bytes
    except Exception as exc:
        st.error(str(exc))


agent_response = st.session_state.get("agent_response")
save_response = st.session_state.get("save_response")
doc_bytes = st.session_state.get("doc_bytes")

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
    else:
        st.info("Run the agent to see the generated response summary.")

with right:
    st.subheader("Task trace")
    if agent_response:
        task_rows = agent_response.get("tasks", [])
        for task in task_rows:
            status = task["status"].replace("_", " ").title()
            st.markdown(f"**Task {task['id']} — {status}**")
            st.write(task["task"])
            if task.get("result"):
                st.caption(task["result"])
            st.divider()
    else:
        st.info("The TODO list will appear here once the agent runs.")


st.subheader("Generated document")
if agent_response and save_response and doc_bytes:
    st.success(f"Saved: {save_response['filename']} ({save_response['size_bytes']} bytes)")
    st.download_button(
        label="Download DOCX",
        data=doc_bytes,
        file_name=save_response["filename"],
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    st.markdown(f"[Open backend file endpoint]({API_BASE_URL.rstrip('/')}{save_response['download_url']})")
else:
    st.info("The DOCX download will appear after the agent completes.")