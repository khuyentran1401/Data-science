from strip_interactive import run_interactive, get_clean_code
import numpy as np 

code = """
>>> import numpy as np
>>> print(np.array([1,2,3]))
[1 2 3]
>>> print(np.array([4,5,6]))
[4 5 6]
"""

def say_hello():
    print("hello")

print(get_clean_code(code))
run_interactive(code)
