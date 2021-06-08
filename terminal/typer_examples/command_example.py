import typer

app = typer.Typer()

@app.command()
def greeting(name: str):
    """Say hello to users"""

    print(f'Hello {name}!')

@app.command()
def say_bye(name: str):
    """Say bye to users"""

    print(f'Good bye {name}')

if __name__ == '__main__':
    app()
