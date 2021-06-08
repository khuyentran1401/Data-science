import typer 

def greeting(name: str):
    """Function to say hello to users"""
    
    print(f'Hello {name}!')

if __name__ == '__main__':
    typer.run(greeting)