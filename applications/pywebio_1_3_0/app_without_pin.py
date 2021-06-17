from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
from utils import IMAGE_TO_LINK


def create_text():
    put_markdown("# Can Dinosaur and Circle Have the Same Statistics?")
    put_text("Two datasets with the same mean, standard deviation, and correlation "
             "could be very different.\n\n"
             "To test this idea, choose a starting shape and a target shape. The starting shape"
             " will transform into the target shape while keeping the same statistics.")

def app():
    create_text()
    start_shapes = ['dino', 'big_slant']
    target_shapes = ['x', 'h_lines', 'v_lines', 'wide_lines', 'high_lines', 'slant_up', 'slant_down',
                     'center', 'star', 'down_parab', 'circle', 'bullseye', 'dots']

    start_shape = select(label='start_shape', options=start_shapes)
    target = select(label='target', options=target_shapes)
    put_image(IMAGE_TO_LINK[f'{start_shape}_{target}.gif'], width='100%')

if __name__ == '__main__':
    start_server(app, debug=True, port='44315')
