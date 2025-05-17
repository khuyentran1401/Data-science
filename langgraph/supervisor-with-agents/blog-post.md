## Introducing LangGraph

LangGraph is a lightweight Python library that makes it easy to build LLM‑driven agents by wiring together prompts, tools, and data sources. Instead of hand‑crafting prompt templates and I/O logic, you define a few simple functions (tools) and register them with a React‑style agent. LangGraph handles the rest: chaining LLM calls, routing tool invocations, and returning structured outputs.

This “MVP” playground repo shows how to blend LangGraph agents with fake in‑memory repositories, letting you explore customer analytics, KPI reporting, and even jokes. Everything’s built around Python, Pydantic models, and a handful of dependencies—no heavy orchestration frameworks required.

---

## Quick Setup

```bash
git clone https://github.com/khuyentran1401/Data-science
cd langgraph
uv sync --all-extras --dev
export OPENAI_API_KEY="your_key_here"
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

* Python 3.10+
* Dependencies managed via `uv sync`
* Requires an OpenAI API key for LLM calls

---

## Project Structure

```
src/
├── ai/
│   ├── agents/        # Agent definitions (customer, kpi, joke)
│   ├── config.py      # provider_name & model_name
│   ├── schemas/       # Pydantic models for inputs/outputs
│   └── tools/         # Simple functions wrapping repos
├── repositories/      # In‑memory fake data
└── main.py            # FastAPI app wiring agents to endpoints
tests/                 # Pytest suites for tools & agents
```

* **agents**: create LLM agents via `create_react_agent()`
* **tools**: thin wrappers over `Fake*Repository` classes
* **schemas**: strongly typed request/response shapes

---

## Core Agent Definition

Here’s the customer agent. It summarizes profile, feedback, and tickets by invoking three tools:

```python
# src/ai/agents/customer_agent.py
from langgraph.prebuilt import create_react_agent
from src.ai.config import provider_name, model_name
from src.ai.tools.customer import (
    get_customer_profile,
    get_customer_feedback,
    get_customer_support_tickets,
)

prompt = "Retrieves and summarizes customer profiles, feedback, and support tickets."

customer_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="customer_agent",
    prompt=prompt,
    tools=[get_customer_profile, get_customer_feedback, get_customer_support_tickets],
)
```

* **model**: e.g. `"openai:gpt-4"`
* **tools**: any callables matching `fn(args…) -> T`

---

## Tool Implementation

Each tool hits a fake in‑memory repo and returns Pydantic models or primitives:

```python
# src/ai/tools/customer.py
from src.repositories.customer import FakeCustomerRepository

repo = FakeCustomerRepository()

def get_customer_profile(customer_id: str):
    return repo.get_profile(customer_id)

def get_customer_feedback(customer_id: str):
    return repo.get_feedback(customer_id)

def get_customer_support_tickets(customer_id: str):
    return repo.get_support_tickets(customer_id)
```

The simplicity lets LangGraph route arguments, validate types, and load outputs into the LLM prompt seamlessly.

---

## Other Agents

### KPI Agent

```python
# src/ai/agents/kpi_agent.py
from langgraph.prebuilt import create_react_agent
from src.ai.tools.kpi import list_available_kpis, get_kpi_by_metric

prompt = "Fetches and explains KPIs like Revenue, Churn, or New Users."

kpi_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="kpi_agent",
    prompt=prompt,
    tools=[list_available_kpis, get_kpi_by_metric],
)
```

* `list_available_kpis()` returns a comma‑separated list
* `get_kpi_by_metric("Revenue")` yields formatted KPI details

### Joke Agent

```python
# src/ai/agents/joke_agent.py
from langgraph.prebuilt import create_react_agent
from src.ai.tools.joke import fetch_random_joke

prompt = "Tells a random joke from the JokeAPI."

joke_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="joke_agent",
    prompt=prompt,
    tools=[fetch_random_joke],
)
```

A fun way to test how agents handle external HTTP calls.

---

## Running an Example

You can invoke agents in Python:

```python
from src.ai.agents.customer_agent import customer_agent

response = customer_agent.invoke("What can you tell me about customer 123?")
print(response)
```

Or hit the FastAPI endpoints:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"message":"Show me customer 456 details"}'
```

You can perform multiple requests on a single 

---

## Next Steps

* **Guard rails**: Subject moderation, fact-check, prompt injection, Role scope, schema validation (with Pydantic-AI)
* **State persistence**: Currently, the implementation persists states locally. Hook agents to Redis, SQLite or Postgres
* **Multi-node flows**: build LangGraph graphs with conditionals
* **Real data sources**: swap fake repos for HubSpot or Mixpanel

This playground gives you a hands‑on feel for LangGraph’s core ideas. Fork it, tweak prompts, add tools, and see how easily you can orchestrate powerful, autonomous LLM workflows.
