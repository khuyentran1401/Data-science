# 🧠 LangGraph AI Agents Playground

This project demonstrates how to create LLM-based agents using [LangGraph](https://github.com/langchain-ai/langgraph), with integration to simulated tools via fake repositories. The focus is on scenarios involving customer data analysis, business metrics (KPIs), and even some humor with random jokes — all orchestrated by autonomous and composable agents.

## 📦 Project Structure

```
src/
├── ai/
│   ├── agents/         # Agent definitions (e.g., customer_agent, joke_agent)
│   ├── config.py       # Provider and model name (e.g., openai:gpt-4)
│   ├── schemas/        # Pydantic models used by agents and tools
│   └── tools/          # Tools used by the agents
├── repositories/       # Fake data sources (e.g., customers, KPIs)
tests/                  # Automated tests
```

## 🧠 Available Agents

### `customer_agent`
Provides insights about a customer based on:
- Profile (name, email, purchase history)
- Feedback received
- Support tickets

### `kpi_agent`
Queries KPIs such as Revenue, Churn, or New Users, including target and trend.

### `joke_agent` (experimental)
Fetches a random joke from [JokeAPI](https://jokeapi.dev), supporting categories (e.g., `Programming`, `Pun`, `Dark`, etc.).

## 🔧 How to Run

You can manually test the agents via script, interactive terminal, or API. Basic example with LangGraph:

```python
from src.ai.agents.customer import customer_agent

response = customer_agent.invoke("What can you tell me about customer 123?")
print(response)
```

To run locally:

```bash
uv sync --all-extras --dev
export OPENAI_API_KEY="..."
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 Requirements

- Python 3.10+
- `langgraph`, `pydantic`, `requests`
- LLM provider: OpenAI API key

## 🚧 Limitations

- The data repositories are fake and stored in-memory.
- Agents don't have state management or conversational history tracking.
- Using invalid joke categories or unhandled KPIs may return generic responses.

## 💡 Future Ideas

- Persistence with SQLite or Redis for agent states
- Creating LangGraph flows with multiple nodes and conditions
- Integration with real data sources (HubSpot, Mixpanel, etc.)