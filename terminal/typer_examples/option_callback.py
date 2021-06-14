import typer

def check_if_user_exists(username: str):
    users = ['ben99', 'taylor', 'winnie76']
    if username not in users:
        raise typer.BadParameter(f"Username {username} doesn't exist. Please sign up to continue.")
    return username

def greeting(username: str = typer.Argument(..., help='Name of user', callback=check_if_user_exists)):
    """Say hello to users"""

    print(f'Hello {username}!')


if __name__ == '__main__':
    typer.run(greeting)
