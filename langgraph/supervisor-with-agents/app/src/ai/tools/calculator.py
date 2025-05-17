# src/ai/tools/calculator.py

def calculate_expression(expression: str) -> float:
    """
    Evaluates a simple mathematical expression and returns the result.
    Only supports basic arithmetic operations: +, -, *, /, **, and parentheses.

    Args:
        expression (str): A string mathematical expression (e.g., "2 * (3 + 4)").

    Returns:
        float: The result of the evaluation.
    """
    try:
        # Only allow safe built-ins
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        return float(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
