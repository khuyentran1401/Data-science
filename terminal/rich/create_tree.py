from rich.tree import Tree
from rich import print

tree = Tree("Dog Data")
tree.add('data1')

data2 = tree.add('data2')
data2.add("[red]Winner")
data2.add("[green]Chihuahua")
data2.add("[blue]Greyhound")

print(tree)