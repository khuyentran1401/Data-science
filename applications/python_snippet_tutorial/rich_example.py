from rich.console import Console
from rich.syntax import Syntax

console = Console()

code = """
import numpy as np
import pandas as pd

a = np.array([1, 2, 3, 4, 5])

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
"""
console.print(Syntax(code, 'python'))
