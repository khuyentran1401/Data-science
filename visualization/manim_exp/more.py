from manim import *

class MovingFrame(Scene):
     def construct(self):
        # Write equations
        equation = MathTex("2x^2-5x+2", "=", "(x-2)(2x-1)")

        # Create animation
        self.play(Write(equation))

        # Add moving frames
        framebox1 = SurroundingRectangle(equation[0], buff=.1)
        framebox2 = SurroundingRectangle(equation[2], buff=.1)

        # Create animations
        self.play(Create(framebox1))  # creating the frame

        self.wait()
        # replace frame 1 with frame 2
        self.play(ReplacementTransform(framebox1, framebox2))
    
        self.wait()


class MathematicalEquation(Scene):
    def construct(self):
        # Write equations
        equation1 = MathTex("2x^2-5x+2")
        eq_sign_1 = MathTex("=")
        equation2 = MathTex("2x^2-4x-x+2")
        eq_sign_2 = MathTex("=")
        equation3 = MathTex("(x-2)(2x-1)")

        # Put each equation or sign in the appropriate positions
        equation1.next_to(eq_sign_1, LEFT)
        equation2.next_to(eq_sign_1, RIGHT)

        eq_sign_2.shift(DOWN)
        equation3.shift(DOWN)

        # Align bottom equations with the top equations
        eq_sign_2.align_to(eq_sign_1, LEFT)
        equation3.align_to(equation2, LEFT)

        # Group equations and sign
        eq_group = VGroup(equation1, eq_sign_1, equation2, eq_sign_2, equation3)

        # Create animation
        self.play(Write(eq_group))
        self.wait()

class MovingAndZoomingCamera(MovingCameraScene):
    def construct(self):
        # Write equations
        equation = MathTex("2x^2-5x+2", "=", "(x-2)(2x-1)")

        self.add(equation)
        self.play(self.camera.frame.animate.move_to(equation[0]).set(width=equation[0].width*2))
        self.wait(0.3)
        self.play(self.camera.frame.animate.move_to(equation[2]).set(width=equation[2].width*2))

class Graph(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            x_min=-3.5,
            x_max=3.5,
            y_min=-5,
            y_max=5,
            graph_origin=ORIGIN,
            axes_color=BLUE,
            x_labeled_nums=range(-4, 4, 2), # x tickers
            y_labeled_nums=range(-5, 5, 2), # y tickers
            **kwargs
        )

    def construct(self):
        self.setup_axes(animate=True)

        # Draw graph
        func_graph_cube = self.get_graph(lambda x: x**3, RED)
        func_graph_ncube = self.get_graph(lambda x: -x**3, GREEN)

        # Create labels
        graph_lab = self.get_graph_label(func_graph_cube, label="x^3")
        graph_lab2 = self.get_graph_label(func_graph_ncube, label="-x^3", x_val=-3)

        # Create a vertical line
        vert_line = self.get_vertical_line_to_graph(1.5, func_graph_cube, color=YELLOW)
        label_coord = self.input_to_graph_point(1.5, func_graph_cube)
        text = MathTex(r"x=1.5")
        text.next_to(label_coord)
       
        self.add(func_graph_cube, func_graph_ncube, graph_lab, graph_lab2, vert_line, text)
        self.wait(3)

class GroupCircles(Scene):
    def construct(self):

        # Create circles
        circle_green = Circle(color=GREEN)
        circle_blue = Circle(color=BLUE)
        circle_red = Circle(color=RED)
        
        # Set initial positions
        circle_green.shift(LEFT)
        circle_blue.shift(RIGHT)
        
        # Create 2 different groups
        gr = VGroup(circle_green, circle_red)
        gr2 = VGroup(circle_blue)
        self.add(gr, gr2) # add two groups to the scene
        self.wait()

        self.play((gr + gr2).animate.shift(DOWN)) # shift 2 groups down
        
        self.play(gr.animate.shift(RIGHT)) # move only 1 group
        self.play(gr.animate.shift(UP))

        self.play((gr + gr2).animate.shift(RIGHT)) # shift 2 groups to the right
        self.play(circle_red.animate.shift(RIGHT))
        self.wait()

class TracedPathExample(Scene):
    def construct(self):
        # Create circle and dot
        circ = Circle(color=BLUE).shift(4*LEFT)
        dot = Dot(color=BLUE).move_to(circ.get_start())

        # Group dot and circle
        rolling_circle = VGroup(circ, dot)
        trace = TracedPath(circ.get_start)

        rolling_circle.add_updater(lambda m: m.rotate(-0.3))  # Rotate the circle

        self.add(trace, rolling_circle) # add trace and rolling circle to the scene

        # Shift the circle to 8*RIGHT
        self.play(rolling_circle.animate.shift(8*RIGHT), run_time=4, rate_func=linear)

