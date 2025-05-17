# src/ai/tools/symbolics.py
from sympy import symbols, symbols, lambdify, nsolve
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.sympify import SympifyError
from scipy.optimize import fsolve

def solve_equation(equation: str):
    """
    Numerically solves a given equation in terms of the variable 'x'.

    This function parses an equation string and attempts to solve it numerically 
    using SymPy's `nsolve`, which is useful for equations that can't be solved 
    symbolically (e.g., transcendental equations like exp(-x) = x).

    Args:
        equation (str): A string representing the equation to solve, in the format 
                        "lhs = rhs", e.g., "x**2 = 4" or "exp(-x) = x".

    Returns:
        float: A numerical approximation of the solution to the equation.

    Raises:
        ValueError: If the equation cannot be parsed or solved.

    Example:
        >>> solve_equation("x**2 = 4")
        2.0
    """
    x = symbols('x')
    try:
        lhs, rhs = map(str.strip, equation.split('='))
        lhs_expr = parse_expr(lhs)
        rhs_expr = parse_expr(rhs)
        sol = nsolve(lhs_expr - rhs_expr, x, 0.5)
        return float(sol)
    except (SympifyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid or unsolvable equation: {e}")

def solve_numeric_equation(expr_str: str) -> float:
    """
    Numerically solves an equation of the form 'lhs = rhs' for x using SciPy's fsolve.

    Args:
        expr_str (str): A string representation of the equation, e.g., 'exp(-x) = x'.
                        Must contain a single '=' sign and be evaluable with 'x' as a symbol.

    Returns:
        float: A numeric approximation of the solution to the equation.

    Notes:
        - This function uses eval(), so only use with trusted inputs.
        - Uses an initial guess of 0.5. For equations with multiple roots or sensitivity, consider exposing this as a parameter.
    """
    x = symbols('x')
    try:
        lhs, rhs = expr_str.split("=")
    except ValueError:
        raise ValueError("I was not able ")

    f = lambdify(x, eval(lhs.strip()) - eval(rhs.strip()), "numpy")
    sol = fsolve(f, 0.5)
    return float(sol[0])
