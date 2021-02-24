from rich.console import Console
from rich.progress import track
from time import sleep

def scrape_data():
    sleep(0.01)

for _ in track(range(100), description='[green]Scraping data'):
    scrape_data()

console = Console()

datas = [1,2,3,4,5]
with console.status("[bold green]Scraping data...", spinner='aesthetic') as status:
    while datas:
        data = datas.pop(0)
        sleep(1)
        console.log(f"[green]Finish scraping data[/green] {data}")
    console.log(f'[bold][red]Done!')
