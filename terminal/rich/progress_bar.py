from rich.progress import track
from time import sleep

def scrape_data():
    sleep(0.1)

for _ in track(range(100), description='[green]Scraping data'):
    scrape_data()

