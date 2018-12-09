#28/11
import math
import os
import random
import sys

import pygame as pg
import numpy as np

CAPTION = "Raptor 's World"
SCREEN_SIZE = (1400, 750)
WALL_SIZE = (SCREEN_SIZE[0], SCREEN_SIZE[1])
TRANSPARENT = (0, 0, 0, 0)





class PigeonActor(object):
    """ This class represents a Pigeon; the victim
        More than 1 Pigeon can be present but this is for later maybe"""

    SIZE = (40, 40)
    MAXSPEED = 2#5  # the speed will be different for each run

    def __init__(self, number_of_pigeons, wall):
        np.random.seed(9001)
        self.wall = wall
        self.speed = random.randint(1, PigeonActor.MAXSPEED)  # a random speed between 1 and MAXSPEED
        self.number_of_pigeons = number_of_pigeons
        self.tensor_of_pigeons = np.zeros(shape=(number_of_pigeons, 4))
        self.max_speed = PigeonActor.MAXSPEED
        self.stop = False
        self.first_tensor()
        self.pos_x = self.tensor_of_pigeons[0][0]
        self.pos_y = self.tensor_of_pigeons[0][1]
        self.trajectory = []
        self.trajectory.append((int(self.pos_x ), int(self.pos_y )))


    def first_tensor(self):
        direction = random.uniform(-1,1)
        if direction >= 0:
            direction = 1
        direction /= abs(direction)
        start = 0*WALL_SIZE[1]
        #direction = 1
        if direction < 0:
            start = 1*WALL_SIZE[1]
        for row in range(0, self.number_of_pigeons):
            self.tensor_of_pigeons[row][2] = direction*self.speed#the speed of the pigeons (uniform for all of them, for the moment being
            self.tensor_of_pigeons[row][3] = random.uniform(0.05 * math.pi, 0.99 * math.pi)#the flying angle between -180 and 180 degrees
            self.tensor_of_pigeons[row][0] = random.uniform(WALL_SIZE[0]/ 2, WALL_SIZE[0])# the Pigeon starts somewhere in the second halve of the width
            self.tensor_of_pigeons[row][1] = start#upper or lower wall
            self.pos_x = self.tensor_of_pigeons[row][0]
            self.pos_y = self.tensor_of_pigeons[row][1]

    def get_pigeons(self):
        return self.tensor_of_pigeons

    def get_pigeons_obstacles(self):
        obstacles = []
        for row in range(0, self.number_of_pigeons):
            obstacles.append(
                PigeonObject('target_' + str(row), (self.tensor_of_pigeons[row][0], self.tensor_of_pigeons[row][1])))
        return obstacles

    def update(self):
        for row in range(0, self.number_of_pigeons):
            self.tensor_of_pigeons[row][0] += self.tensor_of_pigeons[row][2] * math.cos(self.tensor_of_pigeons[row][3])
            self.tensor_of_pigeons[row][1] += self.tensor_of_pigeons[row][2] * math.sin(self.tensor_of_pigeons[row][3])
            self.pos_x = self.tensor_of_pigeons[row][0]
            self.pos_y = self.tensor_of_pigeons[row][1]
            if self.pos_x >= self.wall[2]:
                self.tensor_of_pigeons[row][3] = (math.pi / 2 + self.tensor_of_pigeons[row][3])
            if self.pos_y >= self.wall[3]:
                self.tensor_of_pigeons[row][3] = (math.pi / 2 + self.tensor_of_pigeons[row][3])
            if self.pos_x < self.wall[0]:
                self.tensor_of_pigeons[row][3] = (math.pi - self.tensor_of_pigeons[row][3])
            if self.pos_y < self.wall[1]:
                self.tensor_of_pigeons[row][3] = (math.pi / 2 + self.tensor_of_pigeons[row][3])
            self.trajectory.append((int(self.pos_x), int(self.pos_y)))


class PigeonObject(object):
    SIZE = (20, 20)

    def __init__(self, id, pos):
        self.rect = pg.Rect((0, 0), PigeonObject.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.id = id

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("blue"), image_rect.inflate(-2, -2))
        return image

    def update(self):
        pass

    def draw(self, surface):
        surface.blit(self.image, self.rect)