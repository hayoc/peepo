from peepo.pp.peepo import Peepo
from peepo.pp.generative_model import GenerativeModel

import pygame as pg
import numpy as np

import math
import random


class Bacteria(Peepo):
    SIZE = (3, 3)
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

        self.flagella_prb = 0.6  # likelihood that flagella will cause tumble or run
        self.surroundings = np.empty((4, 4))
        self.generative_model = GenerativeModel(self, n_jobs=1)

    def observation(self, name):
        center_pos = np.array([(Bacteria.SIZE[0]-1)/2, (Bacteria.SIZE[1]-1)/2])
        center_val = self.surroundings[center_pos[0]][center_pos[1]]

        angle_radians = np.radians(self.rotation)
        # Calculate the vector for the given angle
        vector = np.array([np.cos(angle_radians), np.sin(angle_radians)])
        # Calculate the position of the rotated vector
        rotated_vector = center_pos + vector
        # Round the rotated vector to the nearest integer to get the index of the matrix element
        index = np.round(rotated_vector).astype(int)

        target_val = self.surroundings[index[0], index[1]]

        if target_val > center_val:
            return [0.1, 0.9]
        else:
            return [0.9, 0.1]

    def action(self, node, prediction):
        if np.argmax(prediction) == 0:
            self.flagella_prb = 0.6
        else:
            self.flagella_prb = 0.05

    def update(self):
        self.generative_model.process()

        # flagella switch between tumble and run at random (initially 60% chance to tumble)
        if random.random() < self.flagella_prb:
            self.rect.x += Bacteria.SPEED * math.cos(math.radians(self.rotation))
            self.rect.y += Bacteria.SPEED * math.sin(math.radians(self.rotation))
        else:
            self.rotation += 5
            if self.rotation > 360:
                self.rotation = 0

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

    def get_pos(self):
        return self.rect.x, self.rect.y

    def set_surroundings(self, surroundings):
        self.surroundings = surroundings
