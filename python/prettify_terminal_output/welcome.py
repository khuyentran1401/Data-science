from termcolor import colored
from pyfiglet import Figlet


f = Figlet(font='standard')
print(colored(f.renderText('Welcome to My Library!'), 'green'))