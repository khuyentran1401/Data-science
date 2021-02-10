from rich.console import Console
from time import sleep

console = Console()

datas = [1,2,3,4,5]
with console.status("[bold green]Scraping data...") as status:
    while datas:
        data = datas.pop(0)
        sleep(1)
        console.log(f"[green]Finish scraping data[/green] {data}")
    
    console.log(f'[bold][red]Done!')
