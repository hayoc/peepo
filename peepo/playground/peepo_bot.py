class Peepo():

    def __init__(self):
        self.hunger = 0
        self.bladder = 0

    def update(self):
        if self.hunger < 100:
            self.hunger += 0.1
        if self.bladder < 100:
            self.bladder += 0.1
