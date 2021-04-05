from manim import * 

config.background_color = DARK_GRAY

class PointMovingOnShapes(Scene):
    def construct(self):
        
        # Create a square
        square = Square(color=BLUE)
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        
        # Create a circle
        circle = Circle()
        circle.set_fill(PINK, opacity=0.5)
        
        # Create a dot
        dot = Dot()
        dot2 = dot.copy().shift(RIGHT) # create a copy of the dot and shift
                                       # it to the right
        self.add(dot) # Add the dot to the scene
        
         # Create a line next to the dot
        line = Line([3, 0, 0], [5, 0, 0])
        self.add(line)

        
        # Create animations
        self.play(GrowFromCenter(square))
        self.play(Transform(square, circle)) 
        
        # shift the dot to the right
        self.play(Transform(dot, dot2))
        
         # rotates the dot abount point (2,0,0) for 1.5 s
        self.play(Rotating(dot, about_point=[2, 0, 0]), run_time=1.5)
       
        self.wait()



