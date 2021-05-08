from reloading import reloading
from time import sleep

def get_evens(nums: list):
    for num in reloading(nums):
        if num % 2 == 0:
            print(f'{num} is even')
            sleep(2)
        else:
            print(f'{num} is odd')

get_evens([1, 3, 4, 6, 8, 9, 11, 12, 14, 16, 17])
