import argparse

def greeting(name: str):
    print(f'Hello {name}!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Greet users')
    parser.add_argument('name', type=str)
    args = parser.parse_args()

    if args.name:   
        greeting(args.name)