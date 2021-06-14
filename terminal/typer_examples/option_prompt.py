import typer

def greeting(name: str = typer.Argument('Khuyen', help='Name of user')):
    """Say hello to users"""

    print(f'Hello {name}!')


if __name__ == '__main__':
    typer.run(greeting)
