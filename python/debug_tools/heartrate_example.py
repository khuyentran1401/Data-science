import heartrate 
from time import sleep

heartrate.trace(browser=True)

def factorial(x):
    if x == 1:
        return 1
    else:
        sleep(0.5)
        return (x * factorial(x-1))


if __name__ == "__main__":
    num = 20
    print(f"The factorial of {num} is {factorial(num)}")
