from peepo.pp.peepo import Peepo
from peepo.pp.generative_model import GenerativeModel

import pygame as pg

import math


class Bacteria(Peepo):
    SIZE = (4, 4)
    RADIUS = 50
    SPEED = 2

    def __init__(self, name, network, pos=(0, 0)):
        super().__init__(network)

        self.name = name
        self.rect = pg.Rect(pos, Bacteria.SIZE)
        self.rect.center = pos
        self.rotation = 0

        self.image = self.make_image()
        self.image_original = self.image.copy()

        self.generative_model = GenerativeModel(self, n_jobs=1)

    def observation(self, name):
        return [0.0, 0.0]

    def action(self, node, prediction):
        pass

    def update(self):
        self.generative_model.process()

        self.rect.x += Bacteria.SPEED * math.cos(math.radians(self.rotation))
        self.rect.y += Bacteria.SPEED * math.sin(math.radians(self.rotation))

        self.image = pg.transform.rotate(self.image_original, -self.rotation)
        self.rect = self.image.get_rect(center=self.rect.center)

        if self.rect.x < 0 or self.rect.y < 0 or self.rect.x > 800 or self.rect.y > 800:
            self.rect.x, self.rect.y = 400, 400

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill((0, 0, 0, 0))
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image
