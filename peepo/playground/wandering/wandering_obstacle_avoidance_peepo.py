import random

import numpy as np
import pygame as pg
from peepo.pp.v3.sensory_input import SensoryInput
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution


def change_distribution(old, new_values):
    result = []
    for x, old_value in enumerate(old):
        result.append((old_value[0], new_values[x]))
    return dict(result)


class Peepo:

    def __init__(self):
        self.wander_left_chance = 0
        self.wander_right_chance = 0
        self.wandering_left = False
        self.wandering_right = False

    def update(self, model):
        genmodel = model.genmodel
        network = genmodel.bayesian_network

        # [0.9, 0.1] = OFF
        # [0.1, 0.9] = ON
        if self.wandering_left:
            state = network.states[genmodel.get_node_index(model.WAN_LEFT)]
            state.distribution = DiscreteDistribution(change_distribution(state.distribution.items(), [0.9, 0.1]))
            self.wander_left_chance = 0
            self.wandering_left = False
        else:
            self.wander_left_chance += 0.1
            if random.randint(0, 100) <= self.wander_left_chance:
                state = network.states[genmodel.get_node_index(model.WAN_LEFT)]
                state.distribution = DiscreteDistribution(change_distribution(state.distribution.items(), [0.1, 0.9]))
                self.wandering_left = True

        if self.wandering_right:
            state = network.states[genmodel.get_node_index(model.WAN_RIGHT)]
            state.distribution = DiscreteDistribution(change_distribution(state.distribution.items(), [0.9, 0.1]))
            self.wander_right_chance = 0
            self.wandering_right = False
        else:
            self.wander_right_chance += 0.1
            if random.randint(0, 100) <= self.wander_right_chance:
                state = network.states[genmodel.get_node_index(model.WAN_RIGHT)]
                state.distribution = DiscreteDistribution(change_distribution(state.distribution.items(), [0.1, 0.9]))
                self.wandering_right = True


class SensoryInputVirtualPeepo(SensoryInput):
    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def action(self, node, prediction):
        # prediction [0.9, 0.1] = STOP
        # prediction [0.1, 0.9] = START
        if np.argmax(prediction) == 0:
            if 'left' in node:
                self.peepo.motor_output[pg.K_LEFT] = False
            if 'right' in node:
                self.peepo.motor_output[pg.K_RIGHT] = False
        else:
            if 'left' in node:
                self.peepo.motor_output[pg.K_LEFT] = True
            if 'right' in node:
                self.peepo.motor_output[pg.K_RIGHT] = True

    def value(self, name):
        if 'vision' in name:
            # prediction [0.9, 0.1] = NO OBSTACLE
            # prediction [0.1, 0.9] = OBSTACLE
            if '1' in name:
                return np.array([0.1, 0.9]) if self.peepo.obstacle_input['1'] else np.array([0.9, 0.1])
            if '2' in name:
                return np.array([0.1, 0.9]) if self.peepo.obstacle_input['2'] else np.array([0.9, 0.1])
            if '3' in name:
                return np.array([0.1, 0.9]) if self.peepo.obstacle_input['3'] else np.array([0.9, 0.1])
            if '4' in name:
                return np.array([0.1, 0.9]) if self.peepo.obstacle_input['4'] else np.array([0.9, 0.1])
            if '5' in name:
                return np.array([0.1, 0.9]) if self.peepo.obstacle_input['5'] else np.array([0.9, 0.1])
            if '6' in name:
                return np.array([0.1, 0.9]) if self.peepo.obstacle_input['6'] else np.array([0.9, 0.1])
        elif 'motor' in name:
            # prediction [0.9, 0.1] = STOPPED
            # prediction [0.1, 0.9] = MOVING
            if 'left' in name:
                return np.array([0.1, 0.9]) if self.peepo.motor_output[pg.K_LEFT] else np.array([0.9, 0.1])
            if 'right' in name:
                return np.array([0.1, 0.9]) if self.peepo.motor_output[pg.K_RIGHT] else np.array([0.9, 0.1])
