from langgraph.graph import StateGraph, START, END
from agent.agent_state import IrisAgentState as AgentState
from agent.nodes.content_generator import docx_content_generator
from agent.nodes.generate_docx import generate_report_node
from agent.nodes.orchestrator import orchestrator
from agent.nodes.orchestrator import orchestration_router
from agent.nodes.planner import planner
from agent.nodes.reflection import reflection
from agent.nodes.response import chat_response
from agent.nodes.todo_worker import todo_worker

# CREATE GRAPH
graph = StateGraph(AgentState)


# REGISTER NODES
graph.add_node("planner", planner)
graph.add_node("orchestrator", orchestrator)
graph.add_node("todo_worker", todo_worker)
graph.add_node("generate_report_node", generate_report_node)
graph.add_node("docx_content_generator", docx_content_generator)
graph.add_node("reflection", reflection)
graph.add_node("chat_response", chat_response)


# ENTRY POINT
graph.add_edge(START, "planner")

graph.add_edge("planner", "orchestrator")

graph.add_conditional_edges(
    "orchestrator",
    orchestration_router,
    {
        "worker": "todo_worker",
        "next_phase": "reflection"
    }
)

graph.add_edge("todo_worker", "orchestrator")

graph.add_edge("reflection", "docx_content_generator")

graph.add_edge("docx_content_generator", "generate_report_node")

graph.add_edge("generate_report_node", "chat_response")

# EXITS
graph.add_edge("chat_response", END)


# COMPILE
iris = graph.compile()
