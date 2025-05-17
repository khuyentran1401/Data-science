# src/ai/agents/math.py
from langgraph.prebuilt import create_react_agent
from src.ai.config import provider_name, model_name
from src.ai.tools.calculator import calculate_expression
from src.ai.tools.statistics import summarize_statistics
from src.ai.tools.nlp import parse_and_calculate_nlp_expression
from src.ai.tools.equations import solve_equation

prompt=(
    "You are a math assistant for arithmetic, statistics, and equation solving.\n"
    "Use tools as follows:\n"
    "- `calculate_expression`: Basic arithmetic.\n"
    "- `summarize_statistics`: Stats like mean, median, mode.\n"
    "- `parse_and_calculate_nlp_expression`: NLP-to-math conversion.\n"
    "- `solve_equation`: Symbolic solving.\n"
    "- `solve_numeric_equation`: For equations with exp, log, sin, etc.\n"
    "Ask for clarification when input is ambiguous. Validate and confirm before responding."
)


# Create the math agent
math_agent = create_react_agent(
    model=f"{provider_name}:{model_name}",
    name="math_agent",
    prompt=prompt,
    tools=[
        calculate_expression, 
        summarize_statistics, 
        parse_and_calculate_nlp_expression,
        solve_equation
    ],
)

