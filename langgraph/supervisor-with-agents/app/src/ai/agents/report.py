# src/ai/agents/report.py
from langgraph.prebuilt import create_react_agent

from src.ai.tools.kpi import (
    fetch_kpis, get_kpi_by_metric, list_available_kpis,
)
from src.ai.config import provider_name, model_name

# Create the agent with the model
prompt=(
    "You are a business analyst assistant for weekly KPI reports.\n"
    "Use tools:\n"
    "- `fetch_kpis`: fetch all.\n"
    "- `get_kpi_by_metric`: fetch specific metric.\n"
    "- `list_available_kpis`: list all known KPI names.\n"
    "Ask which metrics to report if unclear.\n"
    "For each metric, report value, target, unit, and trend.\n"
    "If a metric is unknown, inform the user and suggest 'list_available_kpis'.\n"
    "Do not invent KPI names or logic."
)


report_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="report_agent",
    prompt=prompt,
    tools=[ list_available_kpis, fetch_kpis, get_kpi_by_metric ],
)
