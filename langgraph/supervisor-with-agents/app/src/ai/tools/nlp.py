# src/ai/tools/nlp.py
from word2number import w2n

def parse_nlp_expression(expression: str) -> str:
    """
    Converts natural language math expressions into a Python-compatible math expression.
    """
    try:
        # First, convert number words to digits using word2number
        expression = ' '.join(
            str(w2n.word_to_num(word)) if word.isalpha() else word for word in expression.split()
        )
        
        # Next, convert operations (plus, minus, etc.)
        for word, symbol in operations.items():
            expression = expression.replace(word, symbol)

        # Handle "and" as plus (common in some NLP formats)
        expression = re.sub(r"\band\b", "+", expression)

        return expression
    except Exception as e:
        return str(e)

def parse_and_calculate_nlp_expression(expression: str) -> float:
    """
    Converts a natural language math expression into an executable Python expression and calculates the result.
    """
    try:
        parsed_expression = parse_nlp_expression(expression)
        result = eval(parsed_expression)
        return result
    except Exception as e:
        return f"Error: {str(e)}"