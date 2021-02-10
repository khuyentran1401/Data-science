from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from sklearn.datasets import fetch_openml


console = Console()

def get_content(row):
    match = 'Match' if int(row['match']) else "Not match"
    d_age = f"[yellow]Age difference: {row['d_age']}"
    return f"[b]{match}[/b]\n{d_age}"
    
speed_dating = fetch_openml(name='SpeedDating', version=1)['frame'].sample(10)
data_renderables = [Panel(get_content(speed_dating.iloc[i])) for i in range(len(speed_dating))]

console.print(Columns(data_renderables))