# Prompting Isn’t Enough: Why We Need Graphs, Not Just Chains

Large Language Models (LLMs) are remarkable at generating responses from natural language prompts. But when it comes to building real-world systems—like customer support assistants, research agents, or multi-step planners—prompting alone often falls short.

Even with clever prompt engineering, you quickly run into challenges:

* Outputs are unpredictable and hard to validate
* Context is lost between turns
* There's no built-in memory or state management
* You can’t easily coordinate multiple tools or decision points

Frameworks like LangChain tried to address this by chaining tools together. You build a sequence of steps: get input → call model → parse output → call another tool. It works well—for a while. But real-world use cases aren’t linear. They involve branching logic, stateful agents, and multiple interactions between tools and users.

Chains break down when:

* You need to revisit a previous step based on a new condition
* You want to interrupt or inspect a process mid-way
* You want agents to “think” over multiple turns before acting
* You need reusable modular logic—not just rigid pipelines

What you need is structure. Not just a sequence—but a graph.

LangGraph offers a new way to build with LLMs: agents connected through nodes in a state machine. Each node can represent a tool, a reasoning step, or even a conditional branch. You model flows explicitly, track state persistently, and coordinate agents as reusable building blocks.

This blog post explores LangGraph through a real-world example: a customer insight assistant that queries profiles, summaries, and tickets. We’ll walk through how it works, why it matters, and what it unlocks that chains can’t.

If you’ve ever felt your LLM workflows were getting too messy, too brittle, or just too linear—this is for you.

# Introducing LangGraph: Structured Workflows for Language Agents

LangGraph is a Python library for building multi-agent systems and tool-augmented LLM workflows using a graph-based execution model. It builds on ideas from LangChain and LangChain Expression Language (LCEL), but with one major upgrade: it lets you structure your logic as a graph, not just a chain.

In LangGraph, each node in the graph represents a tool, an agent, or a decision step. Edges define how control moves between them—based on the agent’s output, state variables, or external conditions. This makes LangGraph ideal for workflows where steps aren’t strictly linear, such as:

* calling different tools depending on the input,
* re-entering a reasoning loop until a stopping condition is met,
* coordinating multiple agents across a shared context or memory.

LangGraph’s model is essentially a stateful, condition-aware graph where LLM calls are just one type of node. This gives you full control over:

* State: What information flows through the graph (e.g., customer context, agent memory)
* Flow: How steps connect or diverge (e.g., retry, route, branch)
* Behavior: What each node does (e.g., call a model, use a tool, generate a plan)

It’s especially well-suited for use cases that require:

* Long-term memory or persistent context across turns
* Reasoning with tool use (a.k.a. ReAct or toolformer patterns)
* Agentic behavior with clear boundaries and decision points

To show LangGraph in action, we’ll walk through a minimal but functional MVP: a customer assistant that reasons through customer data—like profiles, summaries, and support tickets—using LangGraph’s graph-based orchestration.

Let’s look at the example repository and how it ties together LLM agents, tools, and structure.

## Example Walkthrough

This section dives into the **LangGraph-MVP** repository example, illustrating how to define autonomous agents that orchestrate prompts and tools against in‑memory data sources([GitHub][1]). The project leverages Pydantic schemas for structured I/O and LangGraph’s React‑style agent API to bind model calls with custom functions([Real Python][2]). Three primary agents—`customer_agent`, `kpi_agent`, and `joke_agent`—are registered with corresponding tool wrappers, enabling multi‑step workflows with minimal boilerplate([GitHub][1], [MarkTech Post][3]). FastAPI endpoints in the main application file expose each agent over HTTP, allowing easy integration into downstream services or frontends([GitHub][1]). Despite its simplicity, this example highlights fundamental LangGraph features like tool chaining, prompt templating, and the potential to extend to stateful graphs([langchain.com][4]).

---

## Project Structure

The repository’s directory layout shows separation between agent definitions, tool wrappers, and fake data repositories([GitHub][1]):

```
src/
├── ai/
│   ├── agents/        # Agent definitions
│   ├── config.py      # Provider & model settings
│   ├── schemas/       # Pydantic models
│   └── tools/         # Tool functions
├── repositories/      # In-memory fake data
└── main.py            # FastAPI wiring
tests/
```

* The **`agents/`** folder contains Python modules that instantiate LangGraph agents using `create_react_agent()`([GitHub][1]).
* **`tools/`** houses thin wrappers that call repository methods and return Pydantic models for type safety.
* **`repositories/`** holds fake in‑memory data sources, simplifying experimentation.

---

## Defining a Customer Agent

In **`src/ai/agents/customer_agent.py`**, a prompt outlines the workflow, and three tools are passed to the agent factory:

```python
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

* **`create_react_agent()`** wires model calls, prompt templates, and tool invocations into a single agent object.
* Each tool function wraps calls to **`FakeCustomerRepository`**, returning validated Pydantic schemas([GitHub][1]).

---

## Tool Wrappers

Tools live in **`src/ai/tools/`** and follow a simple pattern: import a repository instance and expose a callable:

```python
from src.repositories.customer import FakeCustomerRepository

repo = FakeCustomerRepository()

def get_customer_profile(customer_id: str):
    return repo.get_profile(customer_id)
```

* LangGraph inspects the function signature to route arguments automatically.
* Returned objects are serialized and validated against Pydantic schemas([Zep][5]).

---

## Exposing Agents via FastAPI

In **`src/main.py`**, each agent is mounted to a REST endpoint:

```python
from fastapi import FastAPI
from src.ai.agents.customer_agent import customer_agent
from src.ai.agents.kpi_agent import kpi_agent
from src.ai.agents.joke_agent import joke_agent

app = FastAPI()
app.post("/agent/customer")(customer_agent.invoke)
app.post("/agent/kpi")(kpi_agent.invoke)
app.post("/agent/joke")(joke_agent.invoke)
```

* This pattern provides a microservice interface for LLM‑driven workflows.
* Each endpoint simply proxies to the agent’s `.invoke()` method([GitHub][1]).

---

## Key Takeaways

* **Minimal Boilerplate**: Registering an agent requires only a prompt, model spec, and list of tools([Real Python][2]).
* **Structured I/O**: Pydantic schemas enforce type safety and predictable serialization([langchain.com][4]).
* **Composable**: New agents and tools slot in easily by following the same conventions([GitHub][1]).
* **Extensible**: While this MVP uses in‑memory repos, swapping in database‑backed sinks or stateful graphs is straightforward([medium.com][6]).

This example sets the stage for exploring best practices and considerations in building robust, production‑ready LangGraph applications.

[1]: https://github.com/brunolnetto/langgraph-mvp "GitHub - brunolnetto/langgraph-mvp"
[2]: https://realpython.com/langgraph-python/?utm_source=chatgpt.com "LangGraph: Build Stateful AI Agents in Python - Real Python"
[3]: https://www.marktechpost.com/2025/05/15/meet-langgraph-multi-agent-swarm-a-python-library-for-creating-swarm-style-multi-agent-systems-using-langgraph/?utm_source=chatgpt.com "Meet LangGraph Multi-Agent Swarm: A Python Library for Creating ..."
[4]: https://www.langchain.com/langgraph?utm_source=chatgpt.com "LangGraph - LangChain"
[5]: https://www.getzep.com/ai-agents/langgraph-tutorial?utm_source=chatgpt.com "LangGraph Tutorial: Building LLM Agents with LangChain's ... - Zep"
[6]: https://medium.com/ai-advances/creating-multi-agent-systems-with-langgraph-9a2e1da17c15?utm_source=chatgpt.com "Creating Multi-Agent Systems with LangGraph - Medium"

# Summary of Best Practices and Considerations

Effective LLM orchestration hinges on modular design, clear data contracts, robust state management, and comprehensive observability to ensure reliability and scalability in production environments([labelyourdata.com][1]). Enforcing structured inputs and outputs with Pydantic schemas reduces unpredictability and simplifies error handling, while dedicated monitoring and security controls safeguard against drift, misuse, and compliance risks([Medium][2], [WhyLabs][3]). Incremental development—starting with simple chains, evolving to stateful graphs—coupled with thorough testing and logging, yields maintainable, extensible LangGraph applications ready for real‑world workloads([mirascope.com][4], [zenml.io][5]).

---

## Design Agents and Tools Modularly

Define each tool as a small, focused function with a single responsibility, then register it with your agent to encourage reuse and simplify testing([labelyourdata.com][1]). Group related tools into modules (e.g., `customer_tools`, `kpi_tools`) so that agents remain lightweight and their capabilities clear at a glance([mirascope.com][4]). Drive separation of concerns by letting agents orchestrate tool calls rather than embedding business logic directly into prompts or LLM chains([zenml.io][5]).

---

## Define Clear Schemas and Contracts

Use Pydantic models to enforce input/output contracts for every tool invocation—this catches mismatches early and provides automatic validation and serialization([Medium][2], [pydantic.dev][6]). Keep schemas concise, leveraging nested models for complex structures, and document each field’s purpose to help future contributors understand data flow([Leonidas Constantinou][7]). When a model evolves, version schemas to avoid breaking existing agents mid‑flight([Comunidade OpenAI][8]).

---

## Manage State and Memory

For workflows that span multiple interactions or require historical context, implement stateful agents with explicit memory components rather than passing everything through prompts([letta.com][9], [aisc.substack.com][10]). Persist agent state in external stores (Redis, Databases) when replicability and audit trails are important, or use in‑process caches for ephemeral contexts([letta.com][9]). Define clear boundaries for memory size and retention policies to prevent runaway prompt lengths and escalating compute costs([Prompting Guide][11]).

---

## Implement Robust Error Handling

Anticipate transient failures—network timeouts, model rate limits, or JSON parsing errors—and wrap tool calls with retry logic and exponential backoff([labelyourdata.com][1]). Validate LLM responses against schemas; if validation fails, capture the raw output, log context, and decide whether to retry or surface an error to the user([mirascope.com][4]). Classify errors (user vs. system) to allow graceful degradation of functionality when downstream services are unavailable([spotintelligence.com][12]).

---

## Observability and Monitoring

Instrument agents to emit structured logs for each prompt, tool call, and decision branch, capturing inputs, outputs, and latencies for root‑cause analysis([Galileo AI][13], [Coralogix][14]). Track key metrics—error rates, response times, token usage—and build dashboards sourcing logs, metrics, and traces to spot drift or performance regressions([Edge Delta][15]). Establish alerting thresholds and automated health checks to detect silent failures or anomalous behavior quickly([Galileo AI][13]).

---

## Security and Compliance

Encrypt sensitive data both in transit and at rest, and apply role‑based access controls around agent configuration and tool endpoints to prevent unauthorized usage([labelyourdata.com][1]). Guard against prompt injection by sanitizing user inputs, using allow‑lists for tool names, and validating outputs against strict schemas([Datadog][16]). Maintain audit logs of all LLM interactions to support compliance requirements and forensic investigations if data breaches occur([WhyLabs][3]).

---

## Scalability and Performance

Design agents to batch requests where possible—for example, fetching multiple customer profiles in a single DB call—to reduce overhead and improve throughput([labelyourdata.com][1]). Leverage asynchronous execution or thread pools for I/O‑bound tool calls (databases, HTTP APIs) to maximize concurrency without blocking the event loop([Orq][17]). Profile token usage per step and adjust truncation or summarization strategies to keep cost and latency predictable at scale([Orq][17]).

---

## Incremental Development and Testing

Start small by orchestrating a simple chain of two or three tools, then gradually introduce branching and stateful logic as requirements grow([mirascope.com][4]). Write unit tests for each tool function using mock repositories to ensure deterministic behavior, and integration tests for agent workflows simulating common scenarios and edge cases([zenml.io][5]). Automate test runs in your CI pipeline, gating merges on coverage and lint checks to maintain code quality over time([Reddit][18]).

---

By following these practices—modular design, strict schemas, thoughtful state management, robust error handling, observability, security, and incremental testing—you’ll be well‑equipped to build and scale reliable LangGraph applications in production.

[1]: https://labelyourdata.com/articles/llm-orchestration?utm_source=chatgpt.com "LLM Orchestration: Strategies, Frameworks, and Best Practices"
[2]: https://medium.com/%40speaktoharisudhan/structured-outputs-from-llm-using-pydantic-1a36e6c3aa07?utm_source=chatgpt.com "Structured Outputs from LLM using Pydantic | by Harisudhan.S"
[3]: https://whylabs.ai/blog/posts/best-practices-monitoring-large-language-models-in-nlp?utm_source=chatgpt.com "Best Practices for Monitoring Large Language Models - WhyLabs AI"
[4]: https://mirascope.com/blog/llm-orchestration/?utm_source=chatgpt.com "A Guide to LLM Orchestration - Mirascope"
[5]: https://www.zenml.io/blog/llm-agents-in-production-architectures-challenges-and-best-practices?utm_source=chatgpt.com "LLM Agents in Production: Architectures, Challenges, and Best ..."
[6]: https://pydantic.dev/articles/llm-intro?utm_source=chatgpt.com "Steering Large Language Models with Pydantic"
[7]: https://www.leocon.dev/blog/2024/11/from-chaos-to-control-mastering-llm-outputs-with-langchain-and-pydantic/?utm_source=chatgpt.com "From Chaos to Control: Mastering LLM Outputs with LangChain ..."
[8]: https://community.openai.com/t/structured-output-precision-accuracy-pydantic-vs-a-schema/1054410?utm_source=chatgpt.com "Structured output Precision / Accuracy: Pydantic vs a Schema - API"
[9]: https://www.letta.com/blog/stateful-agents?utm_source=chatgpt.com "Stateful Agents: The Missing Link in LLM Intelligence | Letta"
[10]: https://aisc.substack.com/p/llm-agents-part-6-state-management?utm_source=chatgpt.com "LLM Agents, Part 6 - State Management - Deep Random Thoughts"
[11]: https://www.promptingguide.ai/research/llm-agents?utm_source=chatgpt.com "LLM Agents - Prompt Engineering Guide"
[12]: https://spotintelligence.com/2023/11/17/llm-orchestration-frameworks/?utm_source=chatgpt.com "How to manage LLM - Orchestration Made Simple [5 Frameworks]"
[13]: https://www.galileo.ai/blog/effective-llm-monitoring?utm_source=chatgpt.com "Effective LLM Monitoring: A Step-By-Step Process for AI Reliability ..."
[14]: https://coralogix.com/guides/aiops/llm-observability/?utm_source=chatgpt.com "LLM Observability: Challenges, Key Components & Best Practices"
[15]: https://edgedelta.com/company/blog/how-to-deal-with-llms-observability?utm_source=chatgpt.com "How To Deal With LLMs Observability: 10 Key Practices to Solve ..."
[16]: https://www.datadoghq.com/blog/monitor-llm-prompt-injection-attacks/?utm_source=chatgpt.com "Best practices for monitoring LLM prompt injection attacks to protect ..."
[17]: https://orq.ai/blog/llm-orchestration?utm_source=chatgpt.com "LLM Orchestration in 2025: Frameworks + Best Practices - Orq.ai"
[18]: https://www.reddit.com/r/LocalLLaMA/comments/1fx10hr/llm_ops_best_practices_and_workflow_integration/?utm_source=chatgpt.com "LLM Ops: Best Practices and Workflow Integration : r/LocalLLaMA"

## Conclusion: Embracing Structured LLM Workflows

Graph‑based orchestration transforms LLM applications by providing persistent context retention and stateful memory, enabling agents to reference past interactions seamlessly rather than relying solely on transient prompts([TigerGraph][1]). This structure dramatically improves response accuracy and helps minimize hallucinations through iterative retrieval loops that revisit relevant graph nodes before finalizing an answer([Medium][2], [arXiv][3]). Compared to linear chains, graph workflows offer greater explainability and modularity, making it easier to visualize, debug, and extend complex multi‑step processes([Graph Database & Analytics][4], [Reddit][5]). By fusing knowledge graphs with LLMs, you gain a dynamic reasoning engine that aligns AI outputs with domain constraints, organizational memory, and policy awareness([TigerGraph][1]). The graph paradigm also lays the foundation for multi‑agent collaboration, where different LLMs or tools can specialize in tasks like factual accuracy, empathetic tone, or domain‑specific computations([arXiv][6]). Finally, adopting graph‑centric workflows aligns with best practices in AI system design—promoting modularity, observability, and scalability—so you can iterate quickly while maintaining robust, auditable pipelines([LinkedIn][7], [Deepchecks][8]).

Ready to move beyond brittle chains? Explore the [LangGraph‑MVP repository](https://github.com/brunolnetto/langgraph-mvp) to see how graph‑oriented agents, Pydantic schemas, and tool wrappers come together in a minimal example. Start small by defining a few simple tools and prompts, then evolve your workflows into full‑blown graphs with conditional branches, memory components, and multi‑agent coordination. As you build, you’ll find that structured LLM orchestration not only reduces unexpected behavior but also unlocks new levels of flexibility, maintainability, and user trust.

[1]: https://www.tigergraph.com/glossary/knowledge-graph-llm/?utm_source=chatgpt.com "Knowledge Graph LLM - TigerGraph"
[2]: https://medium.com/%40nebulagraph/graph-rag-the-new-llm-stack-with-knowledge-graphs-e1e902c504ed?utm_source=chatgpt.com "Graph RAG: Unleashing the Power of Knowledge Graphs with LLM"
[3]: https://arxiv.org/html/2410.10039v1?utm_source=chatgpt.com "A Multi-LLM Orchestration Engine for Personalized, Context-Rich ..."
[4]: https://neo4j.com/blog/developer/genai-graph-gathering/?utm_source=chatgpt.com "A Tale of LLMs and Graphs: The GenAI Graph Gathering - Neo4j"
[5]: https://www.reddit.com/r/LocalLLaMA/comments/1hfrg2f/graphbased_editor_for_llm_workflows/?utm_source=chatgpt.com "Graph-Based Editor for LLM Workflows : r/LocalLLaMA - Reddit"
[6]: https://arxiv.org/abs/2410.10039?utm_source=chatgpt.com "A Multi-LLM Orchestration Engine for Personalized, Context-Rich Assistance"
[7]: https://www.linkedin.com/posts/anthony-alcaraz-b80763155_why-graph-based-memory-is-essential-for-next-generation-activity-7254805047060881408-FVor?utm_source=chatgpt.com "Why Graph-Based Memory is Essential for Next-Generation Artificial…"
[8]: https://www.deepchecks.com/glossary/llm-orchestration/?utm_source=chatgpt.com "What is LLM Orchestration? Orchestration Frameworks - Deepchecks"
