from peepo.pp.peepo import Peepo
from peepo.pp.generative_model import GenerativeModel

import pygame as pg
import numpy as np

import math
import random

import time

VISION = 'vision'
MOTOR = 'motor'

FLAGELLA_RUN_PRB = 0.9
FLAGELLA_TUMBLE_PRB = 0.3


class Bacteria(Peepo):
    SIZE = (3, 3)
    RADIUS = 50
    SPEED = 2

    def __init__(self, name, network, graphical=True, pos=(0, 0)):
        super().__init__(network)

        self.name = name
        self.graphical = graphical
        self.rect = pg.Rect(pos, Bacteria.SIZE)
        self.pos = pos
        self.rect.center = pos
        self.rotation = 313

        if self.graphical:
            self.image = self.make_image()
            self.image_original = self.image.copy()

        self.flagella_prb = FLAGELLA_RUN_PRB # likelihood that flagella will cause tumble or run
        self.surroundings = np.empty((3, 3))
        self.generative_model = GenerativeModel(self, n_jobs=1)

        self.health = 1000

        self.center = 0.
        self.target = 0.

        pg.font.init()
        self.fonty = pg.font.SysFont("monospace", 15)

    def observation(self, name):
        if VISION.lower() in name.lower():
            center_pos = np.array([(self.surroundings.shape[0]-1)/2, (self.surroundings.shape[1]-1)/2]).astype(int)
            center_val = self.surroundings[center_pos[0]][center_pos[1]]

            angle_radians = np.radians(self.rotation)
            # Calculate the vector for the given angle
            vector = np.array([3 * np.sin(angle_radians), 3 * np.cos(angle_radians)])
            # Calculate the position of the rotated vector
            rotated_vector = center_pos + vector
            # Round the rotated vector to the nearest integer to get the index of the matrix element
            index = np.round(rotated_vector).astype(int)

            target_val = self.surroundings[index[0], index[1]]

            self.center = center_val
            self.target = target_val

            if target_val > center_val:
                return [0.1, 0.9]
            else:
                return [0.9, 0.1]

        if MOTOR.lower() in name.lower():
            if self.flagella_prb == FLAGELLA_RUN_PRB:
                return [0.1, 0.9]
            else:
                return [0.9, 0.1]

        return [0.5, 0.5]

    def action(self, node, prediction):
        if np.argmax(prediction) == 1:
            self.flagella_prb = FLAGELLA_RUN_PRB
        else:
            self.flagella_prb = FLAGELLA_TUMBLE_PRB

    def update(self):
        self.generative_model.process()
        if random.random() < self.flagella_prb:
            # RUN
            self.rect.x += Bacteria.SPEED * math.cos(math.radians(self.rotation))
            self.rect.y += Bacteria.SPEED * math.sin(math.radians(self.rotation))
        else:
            # TUMBLE
            self.rotation += random.randint(0, 15)
            if self.rotation < 0:
                self.rotation = 360
            if self.rotation > 360:
                self.rotation = 0

        if self.graphical:
            self.image = pg.transform.rotate(self.image_original, -self.rotation)
            self.rect = self.image.get_rect(center=self.rect.center)

        if self.rect.x <= 0:
            self.rect.x = 3
        if self.rect.y <= 0:
            self.rect.y = 3
        if self.rect.x >= 800:
            self.rect.x = 797
        if self.rect.y >= 800:
            self.rect.y = 797

        self.health -= 1
        center_pos = np.array([(Bacteria.SIZE[0]-1)/2, (Bacteria.SIZE[1]-1)/2]).astype(int)
        center_val = self.surroundings[center_pos[0]][center_pos[1]]
        if center_val > 25:
            self.health += 2
        if self.health < 0:
            self.health = 0

    def draw(self, surface):
        label = self.fonty.render(f"C: {round(self.center, 4)} | T: {round(self.target, 4)} | D: {round(self.target - self.center, 4)}", True, (255, 255, 0))
        surface.blit(label, (0, 10))
        label2 = self.fonty.render(f"R: {self.rotation}", True, (255, 255, 0))
        surface.blit(label2, (700, 10))

        label3 = self.fonty.render(f"F: {self.flagella_prb}", True, (255, 255, 0))
        surface.blit(label3, (700, 30))

        label4 = self.fonty.render(f"T: {self.target > self.center}", True, (255, 255, 0))
        surface.blit(label4, (700, 70))

        end_pos = (self.rect.x + 10 * math.cos(math.radians(self.rotation)), self.rect.y + 10 * math.sin(math.radians(self.rotation)))

        #pg.draw.line(surface, (255, 255, 255), (self.rect.x, self.rect.y), end_pos, 2)

        surface.blit(self.image, self.rect)

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill((0, 0, 0, 0))
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image

    def get_pos(self):
        if self.rect.x <= 0 or self.rect.y <= 0:
            return self.pos
        return self.rect.x, self.rect.y

    def set_surroundings(self, surroundings):
        self.surroundings = surroundings
