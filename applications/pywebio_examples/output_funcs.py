from pywebio.output import *
from pywebio.input import * 

put_markdown('## Hello there')
put_text("I hope you are having a great day! Here is our menu")

put_table([
    ['Food', 'Price'],
    ['Noodle', 10], 
    ['Chicken and rice', 11]
    ])

with popup("Subscribe to the page"):
    put_text("Join other foodies!")

food = select("Choose your favorite food", ['noodle', 'chicken and rice'])
put_text(f"You chose {food}. Please wait until it is served!")
import time
put_processbar('bar')
for i in range(1, 11):
    set_processbar('bar', i / 10)
    time.sleep(0.1)
put_markdown("Here is your food! Enjoy!")

if food == 'noodle':
    put_image(open('noodle.jpeg', 'rb').read())
else:
    put_image(open('chicken_and_rice.jpeg', 'rb').read())

put_file("You can download the food here", b"Hello")
