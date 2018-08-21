import numpy as np
import random


class Peepo():

    def __init__(self):
        self.hunger = 0
        self.bladder = 0

        self.start_wander_left_chance = 0
        self.start_wander_right_chance = 0
        self.stop_wander_left_chance = 0
        self.stop_wander_right_chance = 0
        self.wandering_left = False
        self.wandering_right = False

    def update(self, model):
        network = model.models['main'].model

        if self.wandering_left:
            self.stop_wander_left_chance += 1
            if random.randint(0, 100) <= self.stop_wander_left_chance:
                network.get_cpds('wandering_left').values = np.array([0.1, 0.9])
                self.stop_wander_left_chance = 0
                self.wandering_left = False
                print('Stopped wandering left')
        else:
            self.start_wander_left_chance += 0.1
            if random.randint(0, 100) <= self.start_wander_left_chance:
                network.get_cpds('wandering_left').values = np.array([0.9, 0.1])
                self.start_wander_left_chance = 0
                self.wandering_left = True
                print('Started wandering left')

        if self.wandering_right:
            self.stop_wander_right_chance += 1
            if random.randint(0, 100) <= self.stop_wander_right_chance:
                network.get_cpds('wandering_right').values = np.array([0.1, 0.9])
                self.stop_wander_right_chance = 0
                self.wandering_right = True
                # print('Stopped wandering right')
        else:
            self.start_wander_right_chance += 0.1
            if random.randint(0, 100) <= self.start_wander_right_chance:
                network.get_cpds('wandering_right').values = np.array([0.9, 0.1])
                self.start_wander_right_chance = 0
                self.wandering_right = True
                # print('Started wandering right')

        if self.hunger < 100:
            self.hunger += 0.1
        if self.bladder < 100:
            self.bladder += 0.1
