import numpy as np
import random


class Peepo():

    def __init__(self):
        self.hunger = 0
        self.bladder = 0

    def update(self, model):
        network = model.models['main'].model

        network.get_cpds('wandering_left').values = np.array([0.1, 0.9]) if random.randint(0, 100) >= 90 else np.array([0.9, 0.1])
        network.get_cpds('wandering_right').values = np.array([0.1, 0.9]) if random.randint(0, 100) >= 90 else np.array([0.9, 0.1])

        if self.hunger < 100:
            self.hunger += 0.1
        if self.bladder < 100:
            self.bladder += 0.1
