from typing import Annotated, Sequence, TypedDict, List, Dict, Tuple

from langgraph.config import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

def collect_graph_states(graph, inputs, config=None):
    """
    Collects the full state of the graph after each super-step into a list.

    Args:
        graph: The compiled LangGraph instance.
        inputs: The initial input to the graph.
        config: Optional configuration dictionary.

    Returns:
        A list of state dictionaries representing the graph's state after each step.
    """
    states = []
    for state in graph.stream(inputs, config=config, stream_mode="values"):
        states.append(state)
    return states

def prepare_model_inputs(
    initial_prompt: str,
    state: AgentState, 
    config: RunnableConfig, 
    store: BaseStore
):
    # Retrieve user memories and add them to the system message
    # This function is called **every time** the model is prompted. It converts the state to a prompt
    user_id = config.get("configurable", {}).get("user_id")
    namespace = ("memories", user_id)
    memories = [m.value["data"] for m in store.search(namespace)]
    system_msg = f"{initial_prompt}. User memories: {', '.join(memories)}"
    return [{"role": "system", "content": system_msg}] + state["messages"]

def generate_system_prompt(
    role_description: str,
    agents: List[Dict[str, str]],
    usage_rules: List[str],
    communication_principles: List[str],
    format: str = "full"
) -> Tuple[str, ...]:
    """
    Generates a tuple for use as a system prompt, optionally compacted to save tokens.

    Args:
        role_description (str): A sentence describing the assistant's overarching role.
        agents (List[Dict[str, str]]): Each dict should contain 'name' and 'description'.
        usage_rules (List[str]): Rules about how to interact with agents.
        communication_principles (List[str]): Guidelines for formatting and tone.
        format (str): If 'compact', reduces verbosity. Default is 'full'.

    Returns:
        Tuple[str, ...]: A structured prompt ready for use in language model system instructions.
    """

    if format not in {"full", "compact"}:
        raise ValueError("format must be either 'full' or 'compact'")

    lines = []

    # Role description
    lines.append(role_description.strip())

    # Agents
    lines.append("Agents available:")
    for agent in agents:
        name = agent['name']
        description = agent['description']
        lines.append(f"- {name}: {description}" if format == "compact" else f"- **{name}**: {description}")

    # Usage rules
    lines.append("Usage rules:")
    for rule in usage_rules:
        lines.append(f"- {rule}")

    # Communication principles
    lines.append("Communication principles:")
    for principle in communication_principles:
        lines.append(f"- {principle}")

    return "\n".join(tuple(lines))