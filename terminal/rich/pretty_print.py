from rich import print, pretty
from rich.console import Console

pretty.install()

print({'num_list1': [1,2,3], 'num_list2': [3,4,5]})
print((1,2,3,4))
print(False, True)


# console = Console()
# console.print("Hello", "World!", style="bold red")
# console.print("This package [bold red]cannot[/bold red] be used for data with [u][bold cyan]categorical[/bold cyan] columns[/u].")
