from pywebio.input import * 
from pywebio.output import * 
from pywebio import start_server
from cutecharts.charts import Bar
from cutecharts.faker import Faker
import numpy as np 

def create_chart(labels: list, values: list):
    chart = Bar("Food comparison")
    chart.set_options(labels=labels, x_label="Food", y_label="Delicious level")
    chart.add_series("Delicious level", values)
    return chart 


def app():

    food_items = ['Chicken and rice', 'Pho', 'Beef bulgogi', 'Taco salad', 'Boiled eggs']
    delicious_level = [7, 10, 8, 7, 5]

    chart = create_chart(food_items, delicious_level)
    
    put_tabs([
    {'title': 'Food comparison', 'content': put_html(chart.render_notebook())},
    {'title': 'Menu', 'content': put_table([
            ['Food', 'Price'],
            ['Chicken and rice', '8'],
            ['Pho', '10'],
            ['Beef bulgogi', '8']
        ])},
    {'title': "I'm leaving", 'content': put_image(open('bye.jpeg', 'rb').read())}
    ])

if __name__ == '__main__':
    start_server(app, debug=True, port='44315')
