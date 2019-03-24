import math

import numpy as np
import pygame as pg

from peepo.playground.wandering.vision import collision, end_line
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput

LEFT = 'left'
RIGHT = 'right'

VISION = 'vision'
MOTOR = 'motor'
TRANSPARENT = (0, 0, 0, 0)


class Peepo:
    """
    This organism represents peepo. Each peepo takes as parameters a name, an initial position and the list of
    obstacles present in its environment.
    """

    SIZE = (4, 4)
    RADIUS = 50
    SPEED = 2

    def __init__(self, name, network, graphical, pos=(0, 0), obstacles=None):
        self.graphical = graphical
        self.name = name
        self.network = network
        self.rect = pg.Rect(pos, Peepo.SIZE)
        self.rect.center = pos
        self.rotation = 0
        self.edge_right = end_line(Peepo.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(Peepo.RADIUS, self.rotation - 30, self.rect.center)
        self.edge_middle = end_line(Peepo.RADIUS, self.rotation, self.rect.center)

        if self.graphical:
            self.image = self.make_image()
            self.image_original = self.image.copy()

        self.health = 0

        self.obstacles = obstacles or []
        self.motor = {
            LEFT: False,
            RIGHT: False
        }
        self.view = {
            '1': False,
            '2': False,
            '3': False,
            '4': False,
            '5': False,
            '6': False,
        }
        self.generative_model = GenerativeModel(network, SensoryInputPeepo(self), n_jobs=1)

    def update(self):
        self.generative_model.process()

        self.rect.x += Peepo.SPEED * math.cos(math.radians(self.rotation))
        self.rect.y += Peepo.SPEED * math.sin(math.radians(self.rotation))

        if self.motor[LEFT]:
            self.rotation -= 10
            if self.rotation < 0:
                self.rotation = 360
        if self.motor[RIGHT]:
            self.rotation += 10
            if self.rotation > 360:
                self.rotation = 0
        self.edge_middle = end_line(Peepo.RADIUS, self.rotation, self.rect.center)
        self.calculate_obstacles()

        if self.graphical:
            self.image = pg.transform.rotate(self.image_original, -self.rotation)
            self.rect = self.image.get_rect(center=self.rect.center)

        self.edge_right = end_line(Peepo.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(Peepo.RADIUS, self.rotation - 30, self.rect.center)

        if self.rect.x < 0 or self.rect.y < 0 or self.rect.x > 800 or self.rect.y > 800:
            self.rect.x, self.rect.y = 400, 400

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        pg.draw.line(surface, pg.Color("red"), self.rect.center, self.edge_right, 2)
        pg.draw.line(surface, pg.Color("green"), self.rect.center, self.edge_left, 2)

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image

    def calculate_obstacles(self):
        self.view = {x: False for x in self.view}

        peepo_vec = pg.math.Vector2(self.rect.center)
        min_distance = 1000
        closest_obstacle = None

        for obstacle in self.obstacles:
            if self.rect.colliderect(obstacle.rect):
                self.health += 1
                self.obstacles.remove(obstacle)

            if collision(obstacle.rect, peepo_vec, self.edge_left, self.edge_right, Peepo.RADIUS):
                distance = math.hypot(obstacle.rect.x - self.rect.x, obstacle.rect.y - self.rect.y)
                if distance <= min_distance:
                    closest_obstacle = obstacle
                    min_distance = distance

        if closest_obstacle:
            edge1 = end_line(Peepo.RADIUS, self.rotation - 30, self.rect.center)
            edge2 = end_line(Peepo.RADIUS, self.rotation - 20, self.rect.center)
            edge3 = end_line(Peepo.RADIUS, self.rotation - 10, self.rect.center)
            edge4 = end_line(Peepo.RADIUS, self.rotation, self.rect.center)
            edge5 = end_line(Peepo.RADIUS, self.rotation + 10, self.rect.center)
            edge6 = end_line(Peepo.RADIUS, self.rotation + 20, self.rect.center)
            edge7 = end_line(Peepo.RADIUS, self.rotation + 30, self.rect.center)

            self.view['1'] = collision(closest_obstacle.rect, peepo_vec, edge1, edge2, Peepo.RADIUS)
            self.view['2'] = collision(closest_obstacle.rect, peepo_vec, edge2, edge3, Peepo.RADIUS)
            self.view['3'] = collision(closest_obstacle.rect, peepo_vec, edge3, edge4, Peepo.RADIUS)
            self.view['4'] = collision(closest_obstacle.rect, peepo_vec, edge4, edge5, Peepo.RADIUS)
            self.view['5'] = collision(closest_obstacle.rect, peepo_vec, edge5, edge6, Peepo.RADIUS)
            self.view['6'] = collision(closest_obstacle.rect, peepo_vec, edge6, edge7, Peepo.RADIUS)


class SensoryInputPeepo(SensoryInput):

    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def value(self, name):
        if VISION.lower() in name.lower():
            return [0.1, 0.9] if self.peepo.view[self.get_quadrant(name)] else [0.9, 0.1]
        if MOTOR.lower() in name.lower():
            return [0.1, 0.9] if self.peepo.motor[self.get_direction(name)] else [0.9, 0.1]

    def action(self, node, prediction):
        if np.argmax(prediction) == 0:
            self.peepo.motor[self.get_direction(node)] = False
        else:
            self.peepo.motor[self.get_direction(node)] = True

    @staticmethod
    def get_quadrant(name):
        for quad in ['1', '2', '3', '4', '5', '6']:
            if quad.lower() in name.lower():
                return quad
        raise ValueError('Unexpected node name %s, could not find 1,2,3,4,5,6', name)

    @staticmethod
    def get_direction(name):
        for direction in [LEFT, RIGHT]:
            if direction.lower() in name.lower():
                return direction
        raise ValueError('Unexpected node name %s, could not find LEFT, RIGHT', name)


class Obstacle:
    """
    This organism represents an obstacle. It is an inanimate object and therefore only takes a position parameter.
    """

    SIZE = (10, 10)

    def __init__(self, name, pos, graphical):
        self.rect = pg.Rect((0, 0), Obstacle.SIZE)
        self.rect.center = pos
        if graphical:
            self.image = self.make_image()
        self.name = name

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("pink"), image_rect.inflate(-2, -2))
        return image

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def __str__(self):
        return '[' + str(self.rect.x) + ', ' + str(self.rect.y) + ']'

    def __repr__(self):
        return '[' + str(self.rect.x) + ', ' + str(self.rect.y) + ']'
