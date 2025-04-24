# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nest-asyncio==1.6.0",
#     "openai==1.76.0",
#     "pandas==2.2.3",
#     "pydantic==2.11.3",
#     "pydantic-ai==0.1.4",
#     "pydantic-ai-slim[duckduckgo]==0.1.4",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Introduction""")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        instructions="Extract first name, last name, years of experience, and primary skill from the job applicant description.",
        input="Khuyen Tran is a data scientist with 5 years of experience, skilled in Python and machine learning.",
    )

    print(response.output_text)
    return


@app.cell(hide_code=True)
def _():
    # Core Workflow
    return


@app.cell(hide_code=True)
def _():
    import nest_asyncio

    nest_asyncio.apply()
    return


@app.cell
def _():
    from typing import List

    from pydantic import BaseModel
    from pydantic_ai import Agent

    return Agent, BaseModel, List


@app.cell
def _(BaseModel, List):
    class ApplicantProfile(BaseModel):
        first_name: str
        last_name: str
        experience_years: int
        primary_skill: List[str]

    return (ApplicantProfile,)


@app.cell
def _(Agent, ApplicantProfile):
    agent = Agent(
        "gpt-4o-mini-2024-07-18",
        system_prompt="Extract name, years of experience, and primary skill from the job applicant description.",
        output_type=ApplicantProfile,
    )

    result = agent.run_sync(
        "Khuyen Tran is a data scientist with 5 years of experience, skilled in Python and machine learning."
    )
    print(result.output)
    return (result,)


@app.cell
def _(result):
    result.output.model_dump()
    return


@app.cell
def _(result):
    import pandas as pd

    df = pd.DataFrame(result.output.model_dump())
    df
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Using the DuckDuckGo Search Tool""")
    return


@app.cell
def _(BaseModel, List):
    class UnemploymentDataSource(BaseModel):
        title: List[str]
        description: List[str]
        url: List[str]

    return (UnemploymentDataSource,)


@app.cell
def _(Agent, UnemploymentDataSource):
    from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

    # Define the agent with DuckDuckGo search tool
    search_agent = Agent(
        "gpt-4o-mini-2024-07-18",
        tools=[duckduckgo_search_tool()],
        system_prompt="Search DuckDuckGo and return links or resources that match the query.",
        output_type=UnemploymentDataSource,
    )

    # Run a search for unemployment rate dataset
    unemployment_result = search_agent.run_sync(
        "Monthly unemployment rate dataset for US from 2018 to 2024"
    )

    print(unemployment_result.output)
    return (unemployment_result,)


@app.cell
def _(pd, unemployment_result):
    unemployment_df = pd.DataFrame(unemployment_result.output.model_dump())
    unemployment_df
    return


if __name__ == "__main__":
    app.run()
