from rich.console import Console
import pandas as pd

console = Console()

data = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
def edit_data(data):
    var_1 = 45
    var_2 = 30
    var_3 = var_1 + var_2
    data['a'] = [var_1,var_2,var_3]
    console.log(data, log_locals=True)

edit_data(data)