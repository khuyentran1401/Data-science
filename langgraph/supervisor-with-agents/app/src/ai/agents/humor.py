# src/ai/agents/humor.py
from langgraph.prebuilt import create_react_agent
from src.ai.config import provider_name, model_name
from src.ai.tools.jokes import get_random_joke

prompt=(
    "You are a humor expert. Use get_random_joke(category=...) to fetch jokes.\n"
    "Valid categories: Programming, Pun, Spooky, Misc.\n"
    "Default to 'Misc' if none is given. Don't invent categories.\n"
    "If the category is invalid, list the supported ones."
)

humor_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="humor_agent",
    prompt=prompt,
    tools=[get_random_joke]
)

