#"15/11"
import random

import numpy as np


class Peepo():

    def __init__(self):
        self.hunger = 0
        self.bladder = 0

        self.wander_left_chance = 0
        self.wander_right_chance = 0
        self.wandering_left = False
        self.wandering_right = False

    def update(self, model):
        network = model.models['main'].model
        if self.wandering_left:
            #network.get_cpds('Azimuth_Belief').values = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
            self.wander_left_chance = 0
            self.wandering_left = False
        else:
            self.wander_left_chance += 0.1
            if random.randint(0, 100) <= self.wander_left_chance:
                #network.get_cpds('Azimuth_Predicted').values = network.get_cpds('Azimuth_next_cycle').values
                self.wandering_left = True

        if self.wandering_right:
            #network.get_cpds('Reward_Belief').values = np.array([0.1,0.1, 0.8])
            self.wander_right_chance = 0
            self.wandering_right = False
        else:
            self.wander_right_chance += 0.1
            if random.randint(0, 100) <= self.wander_right_chance:
                #network.get_cpds('Reward_Predicted').values =  network.get_cpds('Reward_next_cycle').values
                self.wandering_right = True

        if self.hunger < 100:
            self.hunger += 0.1
        if self.bladder < 100:
            self.bladder += 0.1
