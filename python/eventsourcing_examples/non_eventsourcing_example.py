class World():
    def __init__(self):
        self.history = []

    def add_to_history(self, something):
        self.history.append(something)


world = World()
assert world.history == []

world.add_to_history('something')
assert world.history == ['something']