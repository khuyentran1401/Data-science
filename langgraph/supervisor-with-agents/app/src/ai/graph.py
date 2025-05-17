from typing import TypedDict

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.ai.agents.supervisor import supervisor

# Define the state structure
class GraphState(TypedDict):
    messages: list
    next: str
    status: str

def error_handler(state: GraphState) -> GraphState:
    return {
        "messages": state["messages"] + ["⚠️ Something went wrong."],
        "next": "supervisor"
    }

# Crie o fluxo de trabalho com StateGraph
workflow = StateGraph(GraphState)

# FLow with error
workflow.add_node("supervisor", supervisor)

# Defina o ponto de entrada
workflow.set_entry_point("supervisor")

# Compile o fluxo de trabalho
corp_workflow = workflow.compile(checkpointer=MemorySaver()) 