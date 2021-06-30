import typer

app = typer.Typer()

@app.command()
def search_category():
    """Select post based on category"""
    pass 

@app.command()
def search_posts():
    """Search post based on the string pattern"""
    pass 

app()