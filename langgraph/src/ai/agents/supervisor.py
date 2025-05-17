# src/ai/agents/supervisor.py
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI

from src.ai.config import model_name
from src.ai.utils import generate_system_prompt
from src.ai.tools.memory import save_memory
from src.ai.agents.customer import customer_agent 
from src.ai.agents.report import report_agent
from src.ai.agents.humor import humor_agent
from src.ai.agents.math import math_agent

# Define your model
llm_model = ChatOpenAI(model=model_name)

# List of agents
agents = [
    customer_agent, 
    report_agent,
    humor_agent,
    math_agent
]

# List of tools
tools = [
    save_memory
]

role_description = "You are a capable assistant responsible for coordinating specialized agents and tools to address user queries effectively."

agents = [
    {
        "name": "Customer Agent",
        "description": "Retrieves customer profiles, feedback, and support tickets.",
        "obj": customer_agent
    },
    {
        "name": "Report Agent",
        "description": "Provides KPI metrics such as revenue, churn, and new user count.",
        "obj": report_agent
    },
    {
        "name": "Humor Agent",
        "description": "Delivers jokes in categories like Programming, Dark, Pun, and more.",
        "obj": humor_agent
    },
    {
        "name": "Math Agent",
        "description": "Solves math problems, evaluates expressions, handles statistics, and interprets natural language math like 'What is five plus two?' or 'Solve exp(-x) = x'.",
        "obj": math_agent
    }
]

usage_rules = [
    "Use agents strictly within their defined scope. Do not invent or extend capabilities.",
    "If a request is ambiguous or outside agent scope, ask the user for clarification before proceeding.",
    "If an agent fails or returns an error, explain the issue and suggest alternatives or next steps."
]

communication_principles = [
    "Be concise, structured, and polite.",
    "Use bullet points for lists.",
    "Use labeled sections for structured data (e.g., `Customer Info:`, `KPI Report:`).",
    "Provide short summaries when multiple steps are involved.",
    "If you don’t know something, respond with 'I don't know.'",
    "Ask specific questions to clarify unclear requests.",
    "When helpful, present relevant options, steps, or tools in an easy-to-read format."
]

system_prompt = generate_system_prompt(
    role_description=role_description,
    agents=agents,
    usage_rules=usage_rules,
    communication_principles=communication_principles
)

supervisor = create_supervisor(
    model=llm_model, 
    prompt=system_prompt,
    agents=list(map(lambda agent_obj: agent_obj['obj'], agents)),
    tools=tools
).compile()