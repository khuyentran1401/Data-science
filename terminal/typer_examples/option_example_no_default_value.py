import typer

def greeting(name: str = typer.Argument(..., help='Username'),
             is_user: bool = typer.Option(..., help='Whether user has signed up')):
    """Say hello to users"""
    if is_user:
        print(f'Hello {name}!')
    else:
        print(f"You haven't signed up yet. Please sign up to continue.")


if __name__ == '__main__':
    typer.run(greeting)
