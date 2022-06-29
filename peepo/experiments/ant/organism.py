import math
import random
import numpy as np
import pygame as pg

from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution

from peepo.pp.generative_model import GenerativeModel
from peepo.pp.peepo import Peepo

LEFT = 'left'
RIGHT = 'right'

VISION = 'vision'
MOTOR = 'pro'
TRANSPARENT = (0, 0, 0, 0)


class AntPeepo(Peepo):
    """
    This organism represents peepo.
    """

    SIZE = (4, 4)
    RADIUS = 50
    SPEED = 2

    def __init__(self, name, network, graphical, pos=(0, 0)):
        super().__init__(network)

        self.graphical = graphical
        self.name = name
        self.rect = pg.Rect(pos, AntPeepo.SIZE)
        self.rect.center = pos
        self.rotation = 0

        if self.graphical:
            self.image = self.make_image()
            self.image_original = self.image.copy()

        self.motor = {
            LEFT: False,
            RIGHT: False
        }
        self.generative_model = GenerativeModel(self, n_jobs=1)

        self.path = []

    def observation(self, name):
        if MOTOR.lower() in name.lower():
            return [0.1, 0.9] if self.motor[self.get_direction(name)] else [0.9, 0.1]
        return [0.5, 0.5]

    def action(self, node, prediction):
        if np.argmax(prediction) == 0:
            self.motor[self.get_direction(node)] = False
        else:
            self.motor[self.get_direction(node)] = True

    @staticmethod
    def get_direction(name):
        for direction in [LEFT, RIGHT]:
            if direction.lower() in name.lower():
                return direction
        raise ValueError('Unexpected node name %s, could not find LEFT, RIGHT', name)

    def update(self):
        self.generative_model.process()
        self.drift_hypotheses()

        self.rect.x += AntPeepo.SPEED * math.cos(math.radians(self.rotation))
        self.rect.y += AntPeepo.SPEED * math.sin(math.radians(self.rotation))

        if self.motor[LEFT]:
            self.rotation -= 10
            if self.rotation < 0:
                self.rotation = 360
        if self.motor[RIGHT]:
            self.rotation += 10
            if self.rotation > 360:
                self.rotation = 0

        if self.graphical:
            self.image = pg.transform.rotate(self.image_original, -self.rotation)
            self.rect = self.image.get_rect(center=self.rect.center)

        if self.rect.x < 0:
            self.rect.x = 799
        if self.rect.x > 800:
            self.rect.x = 1
        if self.rect.y < 0:
            self.rect.y = 799
        if self.rect.y > 800:
            self.rect.y = 1

        self.path.append((self.rect.x, self.rect.y))

    def drift_hypotheses(self):
        for root in self.generative_model.get_roots():
            root_index = self.generative_model.get_node_index(root.name)

            if random.choice([0, 1]):
                old_hypo = self.generative_model.bayesian_network.states[root_index].distribution.items()

                moving = [old_hypo[0][1], old_hypo[1][1]]
                if np.argmax(moving):  # ~ [0.1, 0.9] -> [0.9, 0.1]
                    change = 0.05
                    if moving[1] - change < 0.1:
                        moving = [0.9, 0.1]
                    else:
                        moving = [moving[0] + change, moving[1] - change]
                else:       # ~ [0.9, 0.1] -> [0.1, 0.9]
                    change = 0.1
                    if moving[0] - change < 0.1:
                        moving = [0.1, 0.9]
                    else:
                        moving = [moving[0] - change, moving[1] + change]

                new_hypo = {0: moving[0], 1: moving[1]}

                self.generative_model.bayesian_network.states[root_index].distribution = DiscreteDistribution(new_hypo)

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        for step in self.path:
            surface.set_at(step, pg.Color("red"))

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image
