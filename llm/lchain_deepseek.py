# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-core==0.3.62",
#     "langchain-deepseek==0.1.3",
#     "langchain-ollama==0.3.3",
#     "langchain-openai==0.3.18",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Build Smarter Data Science Workflows with DeepSeek and LangChain""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Using DeepSeek Chat Models""")
    return


@app.cell
def _():
    import os

    DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
    return


@app.cell
def _():
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_deepseek import ChatDeepSeek

    # Initialize the chat model
    llm = ChatDeepSeek(
        model="deepseek-chat",  # Can also use "deepseek-reasoner"
        temperature=0,  # 0 for more deterministic responses
        max_tokens=None,  # None means model default
        timeout=None,  # API request timeout
        max_retries=2,  # Retry failed requests
    )

    # Create a conversation with system and user messages
    messages = [
        SystemMessage(
            content="You are a data scientist who writes efficient Python code"
        ),
        HumanMessage(
            content="Given a DataFrame with columns 'product' and 'sales', calculates the total sales for each product."
        ),
    ]

    # Generate a response
    response = llm.invoke(messages)
    print(response.content)
    return HumanMessage, SystemMessage, llm, messages


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can also use asynchronous operations for handling multiple requests without blocking:""")
    return


@app.cell
async def _(llm, messages):
    async def generate_async():
        response = await llm.ainvoke(messages)
        return response.content

    # In async context
    async_result = await generate_async()
    print(async_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Building Chains with DeepSeek""")
    return


@app.cell
def _(llm):
    from langchain_core.prompts import ChatPromptTemplate

    # Create a structured prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a data scientist who writes efficient {language} code",
            ),
            ("human", "{input}"),
        ]
    )

    # Build the chain
    chain = prompt | llm

    # Execute the chain
    result = chain.invoke(
        {
            "language": "SQL",
            "input": "Given a table with columns 'product' and 'sales', calculates the total sales for each product.",
        }
    )

    print(result.content)
    return ChatPromptTemplate, prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Streaming Responses""")
    return


@app.cell
def _(llm, prompt):
    from langchain_core.output_parsers import StrOutputParser

    streamed_chain = prompt | llm | StrOutputParser()

    for chunk in streamed_chain.stream(
        {
            "language": "SQL",
            "input": "Given a table with columns 'product' and 'sales', calculates the total sales for each product.",
        }
    ):
        print(chunk, end="", flush=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Structured Output""")
    return


@app.cell
def _(ChatPromptTemplate, llm):
    from typing import List

    from langchain_core.pydantic_v1 import BaseModel

    # Define the output schema
    class ApplicantProfile(BaseModel):
        first_name: str
        last_name: str
        experience_years: int
        primary_skill: List[str]

    # Bind the Pydantic model to the LLM for structured output
    structured_llm = llm.with_structured_output(ApplicantProfile)

    # Create a chain
    prompt_structured = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful asssitant. Provide your output in the requested structured format.",
            ),
            (
                "human",
                "Extract name, years of experience, and primary skill from {job_description}.",
            ),
        ]
    )

    chain_structured = prompt_structured | structured_llm

    # Get structured output
    job_description = "Khuyen Tran is a data scientist with 5 years of experience, skilled in Python and machine learning."
    profile = chain_structured.invoke({"job_description": job_description})
    print(f"First name: {profile.first_name}")
    print(f"Last name: {profile.last_name}")
    print(f"Years of experience: {profile.experience_years}")
    print(f"Primary skills: {profile.primary_skill}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Running DeepSeek Locally with Ollama""")
    return


@app.cell
def _(HumanMessage, SystemMessage):
    from langchain_ollama import ChatOllama

    local_deepseek_ollama = ChatOllama(
        model="deepseek-r1:1.5b", temperature=0.7, base_url="http://localhost:11434"
    )
    response_local = local_deepseek_ollama.invoke(
        [
            SystemMessage(
                content="You are a data scientist who writes efficient Python code"
            ),
            HumanMessage(
                content="Given a DataFrame with columns 'product' and 'sales', calculates the total sales for each product."
            ),
        ]
    )
    print("Response from local DeepSeek (via Ollama):")
    print(response_local.content)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
