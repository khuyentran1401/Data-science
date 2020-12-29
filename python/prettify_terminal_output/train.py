from termcolor import colored
from pyfiglet import Figlet
import time 
import pandas as pd 

df = pd.DataFrame({'a': [1,2,3], 
                    'b': [3,4,5]})
f = Figlet(font='banner3-D')
colors = ['yellow', 'red', 'green', 'blue']
print('Your original data is')
print(df)
for i, color in enumerate(colors):
    print(colored(f.renderText(f'Model {i+1}'), color))
    print('****************Training****************')
    time.sleep(2)
    print('Output is')
    print(df.multiply(i))
    print('****************Complete****************')
    
