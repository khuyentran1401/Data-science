from strip_interactive import run_interactive, get_clean_code
import numpy as np 

code = """
>>> import numpy as np
>>> # Create array
>>> print(np.array([1,2,3]))
[1 2 3]
>>> print(np.array([4,5,6]))
[4 5 6]
"""

print(get_clean_code(code))
run_interactive(code)
