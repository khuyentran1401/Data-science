"""Example for interrogate"""


class MathOperation:
    """Perform math operation"""

    def __init__(self, num) -> None:
        self.num = num

    def plus_two(self):
        """Add 2"""
        return self.num + 2

    def multiply_three(self):
        """Multiply by 3"""
        return self.num * 3
