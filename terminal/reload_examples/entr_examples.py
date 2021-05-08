def get_evens(nums: list):
    return [num for num in nums if num % 2 == 0]

def test_get_evens():
    assert get_evens([1, 3, 4, 6, 8, 9]) == [4, 6, 8]

"""On your terminal
ls entr_examples.py | entr python entr_examples.py 
"""