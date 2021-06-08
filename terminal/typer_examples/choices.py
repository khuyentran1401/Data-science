import typer
from enum import Enum 

class Color(str, Enum):
    red = 'red'
    yellow = 'yellow'
    orange = 'orange'

def greeting(name: str, color: Color):
    """Say hello to users"""

    print(f"Hello {name}! Your character's color is {color}.")


if __name__ == '__main__':
    typer.run(greeting)
