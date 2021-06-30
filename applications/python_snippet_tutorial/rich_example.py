from rich.console import Console
from rich.syntax import Syntax

console = Console()

code = """
import pandas as pd                                                              
                                                                                 
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})                              
                                                                                 
df = (df.assign(col3=lambda x: x.col1 * 100 + x.col2)                            
    .assign(col4=lambda x: x.col2 * x.col3)                                      
    )                                                                            
print(df)                                                                        
"""
console.print(Syntax(code, 'python'))
