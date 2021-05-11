from rich_dataframe import DataFramePrettify 
from rich import print
# from rich.console import Console
import pandas as pd 

def test_create_table():
    df = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6]})
    # DataFramePrettify.create_table(df)
    table = DataFramePrettify(df).prettify()
    print(table)