from termcolor import colored, cprint
from pyfiglet import Figlet


print_red = lambda x: cprint(x, 'red')
print_green = lambda x: cprint(x, 'green')

print_red("I'm red")
print_green("Yay! I'm green")
print_red("Oops! I'm red again")

f = Figlet(font='broadway')
print(colored(f.renderText('Hello'), 'green'))

f = Figlet(font='isometric2')
print(colored(f.renderText('Hello'), 'green'))


