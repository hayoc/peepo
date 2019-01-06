import math

import numpy as np

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput

LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'

VISION = 'vision'
MOTOR = 'motor'


class Peepo:
    """
    This organism represents peepo. Each peepo takes as parameters a name, an initial position and the list of
    obstacles present in its environment.
    """

    SIZE = (2, 2)
    VIEW_DIST = 100

    def __init__(self, name, network, pos=(0, 0), obstacles=None):
        self.name = name
        self.network = network
        self.x = pos[0]
        self.y = pos[1]
        self.obstacles = obstacles or []
        self.food = 0
        self.motor = {
            LEFT: False,
            RIGHT: False,
            UP: False,
            DOWN: False
        }
        self.view = {
            LEFT: False,
            RIGHT: False,
            UP: False,
            DOWN: False
        }
        self.path = []
        self.loop = 0
        self.generative_model = GenerativeModel(network, SensoryInputPeepo(self))

    def update(self):
        if self.loop % 10 == 0:
            self.path.append((self.x, self.y))

        if self.motor[LEFT]:
            self.x -= 1
        if self.motor[RIGHT]:
            self.x += 1
        if self.motor[UP]:
            self.y += 1
        if self.motor[DOWN]:
            self.y -= 1

        self.view = {x: False for x in self.view}

        for obstacle in self.obstacles.copy():
            if self.x <= obstacle.x <= self.x + self.SIZE[0] and self.y <= obstacle.y <= self.y + self.SIZE[1]:
                self.food += 1
                self.obstacles.remove(obstacle)
            else:
                distance = math.hypot(obstacle.x - self.x, obstacle.y - self.y)
                if distance < self.VIEW_DIST:
                    v1_up = (self.x - self.x + 1, self.y - self.y + 1)
                    v2_up = (obstacle.x - self.x + 1, obstacle.y - self.y + 1)
                    cross_product_up = v1_up[0] * v2_up[1] - v1_up[1] * v2_up[0]

                    v1_down = (self.x - 1 - self.x, self.y + 1 - self.y)
                    v2_down = (self.x - 1 - obstacle.x, self.y + 1 - obstacle.y)
                    cross_product_down = v1_down[0] * v2_down[1] - v1_down[1] * v2_down[0]

                    if cross_product_up <= 0 and cross_product_down < 0:
                        self.view[DOWN] = True
                    elif cross_product_up > 0 and cross_product_down <= 0:
                        self.view[LEFT] = True
                    elif cross_product_up >= 0 and cross_product_down > 0:
                        self.view[UP] = True
                    elif cross_product_up < 0 and cross_product_down >= 0:
                        self.view[RIGHT] = True

        self.loop += 1


class SensoryInputPeepo(SensoryInput):

    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def value(self, name):
        if VISION in name:
            return [0.9, 0.1] if self.peepo.view[self.get_direction(name)] else [0.1, 0.9]
        if MOTOR in name:
            return [0.9, 0.1] if self.peepo.motor[self.get_direction(name)] else [0.1, 0.9]

    def action(self, node, prediction):
        if np.argmax(prediction) == 0:
            self.peepo.motor[self.get_direction(node)] = False
        else:
            self.peepo.motor[self.get_direction(node)] = True

    @staticmethod
    def get_direction(name):
        for direction in [LEFT, RIGHT, UP, DOWN]:
            if direction in name:
                return direction
        raise ValueError('Unexpected node name %s, could not find LEFT, RIGHT, UP or DOWN', name)


class Food:
    """
    This organism represents food. It is an inanimate object and therefore only takes a position parameter. A position
    at which it stays until it is eaten.
    """

    def __init__(self, name, pos=(0, 0)):
        self.name = name
        self.x = pos[0]
        self.y = pos[1]
