import snoop 

@snoop
def factorial(x):
    if x == 1:
        return 1
    else:
        return (x * factorial(x-1))


if __name__ == "__main__":
    num = 5
    print(f"The factorial of {num} is {factorial(num)}")