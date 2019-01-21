import math

import numpy as np
import pygame as pg

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput

LEFT = 'LEFT'
RIGHT = 'RIGHT'
UP = 'UP'
DOWN = 'DOWN'

VISION = 'VISION'
MOTOR = 'MOTOR'

TRANSPARENT = (0, 0, 0, 0)


class Peepo:
    """
    This organism represents peepo. Each peepo takes as parameters a name, an initial position and the list of
    obstacles present in its environment.
    """

    SIZE = (10, 10)
    VIEW_DIST = 200

    def __init__(self, name, network, pos=(0, 0), obstacles=None):
        self.rect = pg.Rect((0, 0), Peepo.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.image_original = self.image.copy()

        self.name = name
        self.network = network
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
        self.generative_model = GenerativeModel(network, SensoryInputPeepo(self), n_jobs=1)

    def update(self):
        self.rect = self.image.get_rect(center=self.rect.center)

        self.generative_model.process()

        if self.motor[LEFT]:
            self.rect.x -= 5
        elif self.motor[RIGHT]:
            self.rect.x += 5
        elif self.motor[UP]:
            self.rect.y += 5
        elif self.motor[DOWN]:
            self.rect.y -= 5

        if self.rect.x < 0 or self.rect.y < 0 or self.rect.x > 800 or self.rect.y > 800:
            self.rect.x, self.rect.y = 400, 400

        self.view = {x: False for x in self.view}

        for obstacle in self.obstacles:
            if self.rect.x <= obstacle.x <= self.rect.x + self.SIZE[0] and self.rect.y <= obstacle.y <= self.rect.y + \
                    self.SIZE[1]:
                self.food += 1
                print(len(self.obstacles))
                self.obstacles.remove(obstacle)
                print(self.name + ' found food!')
                print(len(self.obstacles))
            else:
                distance = math.hypot(obstacle.x - self.rect.x, obstacle.y - self.rect.y)
                if distance < self.VIEW_DIST:
                    v1_up = (self.rect.x - self.rect.x + 1, self.rect.y - self.rect.y + 1)
                    v2_up = (obstacle.x - self.rect.x + 1, obstacle.y - self.rect.y + 1)
                    cross_product_up = v1_up[0] * v2_up[1] - v1_up[1] * v2_up[0]

                    v1_down = (self.rect.x - 1 - self.rect.x, self.rect.y + 1 - self.rect.y)
                    v2_down = (self.rect.x - 1 - obstacle.x, self.rect.y + 1 - obstacle.y)
                    cross_product_down = v1_down[0] * v2_down[1] - v1_down[1] * v2_down[0]

                    if cross_product_up <= 0 and cross_product_down < 0:
                        self.view[DOWN] = True
                    elif cross_product_up > 0 and cross_product_down <= 0:
                        self.view[LEFT] = True
                    elif cross_product_up >= 0 and cross_product_down > 0:
                        self.view[UP] = True
                    elif cross_product_up < 0 and cross_product_down >= 0:
                        self.view[RIGHT] = True

        # print(self.name + ' : ' + str(self.rect.x) + ' - ' + str(self.rect.y))

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image


class Food:
    """
    This organism represents food. It is an inanimate object and therefore only takes a position parameter. A position
    at which it stays until it is eaten.
    """
    SIZE = (10, 10)

    def __init__(self, name, pos=(0, 0)):
        self.rect = pg.Rect((0, 0), Food.SIZE)
        self.x = pos[0]
        self.y = pos[1]
        self.rect.center = pos
        self.image = self.make_image()
        self.id = name

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("pink"), image_rect.inflate(-2, -2))
        return image

    def update(self):
        pass

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def __str__(self):
        return '[' + str(self.rect.x) + ', ' + str(self.rect.y) + ']'

    def __repr__(self):
        return '[' + str(self.rect.x) + ', ' + str(self.rect.y) + ']'


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
