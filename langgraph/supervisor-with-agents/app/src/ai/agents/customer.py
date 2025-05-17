# src/ai/agents/customer.py
from langgraph.prebuilt import create_react_agent

from src.ai.config import provider_name, model_name
from src.ai.tools.customer import (
    get_customer_profile,
    get_customer_feedback,
    get_customer_support_tickets
)

prompt=(
    "Retrieves and summarizes customer profiles, feedback, and support tickets."
)

# Create the agent with the model
customer_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="customer_agent",
    prompt=prompt,
    tools=[
        get_customer_profile,
        get_customer_feedback,
        get_customer_support_tickets
    ]
)
