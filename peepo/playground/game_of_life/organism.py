import math

import numpy as np
import pygame as pg
from peepo.pp.v3.sensory_input import SensoryInput

from peepo.playground.game_of_life.vision import collision, end_line
from peepo.pp.generative_model import GenerativeModel

LEFT = 'left'
RIGHT = 'right'

VISION = 'vision'
MOTOR = 'motor'

ID_ENNEMY = 'ennemy'
ID_FOOD = 'food'

TRANSPARENT = (0, 0, 0, 0)


class Peepo:
    """
    This organism represents peepo. Each peepo takes as parameters a name, an initial position and the list of
    obstacles present in its environment.
    """

    SIZE = (4, 4)
    RADIUS = 50
    SPEED = 2

    def __init__(self, name, network, graphical, pos=(0, 0), ennemies=None, food = None):
        self.graphical = graphical
        self.name = name
        self.network = network
        self.rect = pg.Rect(pos, Peepo.SIZE)
        self.rect.center = pos
        self.rotation = 0
        self.edge_right = end_line(Peepo.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(Peepo.RADIUS, self.rotation - 30, self.rect.center)

        if self.graphical:
            self.image = self.make_image()
            self.image_original = self.image.copy()

        self.stomach = 0
        self.bang = 0

        self.ennemies = ennemies or []
        self.food = food or []
        self.obstacles = []
        self.assemble_obstacles()
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
        self.is_an_enemy = False
        self.is_food = False
        self.generative_model = GenerativeModel(network, SensoryInputPeepo(self), n_jobs=1)

    def assemble_obstacles(self):
        for i,x in enumerate(self.food):
            self.obstacles.append([x,0,i])
        for i,x in enumerate(self.ennemies):
            self.obstacles.append([x,1,i])

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
        self.is_an_enemy = False
        self.is_food = False
        observations = self.obstacles
        for obstacle in observations:
            if self.rect.colliderect(obstacle[0].rect):
                if obstacle[1] == 0:
                    self.stomach += 1
                    self.obstacles.remove(obstacle)
                else:
                    self.bang += 1
                    # self.obstacles.remove(obstacle)

            peepo_vec = pg.math.Vector2(self.rect.center)
            if collision(obstacle[0].rect, peepo_vec, self.edge_left, self.edge_right, Peepo.RADIUS):
                edge1 = end_line(Peepo.RADIUS, self.rotation - 30, self.rect.center)
                edge2 = end_line(Peepo.RADIUS, self.rotation - 20, self.rect.center)
                edge3 = end_line(Peepo.RADIUS, self.rotation - 10, self.rect.center)
                edge4 = end_line(Peepo.RADIUS, self.rotation, self.rect.center)
                edge5 = end_line(Peepo.RADIUS, self.rotation + 10, self.rect.center)
                edge6 = end_line(Peepo.RADIUS, self.rotation + 20, self.rect.center)
                edge7 = end_line(Peepo.RADIUS, self.rotation + 30, self.rect.center)

                self.view['1'] = collision(obstacle[0].rect, peepo_vec, edge1, edge2, Peepo.RADIUS)
                self.view['2'] = collision(obstacle[0].rect, peepo_vec, edge2, edge3, Peepo.RADIUS)
                self.view['3'] = collision(obstacle[0].rect, peepo_vec, edge3, edge4, Peepo.RADIUS)
                self.view['4'] = collision(obstacle[0].rect, peepo_vec, edge4, edge5, Peepo.RADIUS)
                self.view['5'] = collision(obstacle[0].rect, peepo_vec, edge5, edge6, Peepo.RADIUS)
                self.view['6'] = collision(obstacle[0].rect, peepo_vec, edge6, edge7, Peepo.RADIUS)
                if obstacle[1] == 1:
                    self.is_an_enemy = True
                if obstacle[1] == 0:
                    self.is_food = True
        self.food = []
        [self.food.append(x[0]) for x in self.obstacles if x[1] == 0]
        # self.ennemies = []
        # [self.ennemies.append(x[0]) for x in self.obstacles if x[1] == 1]

class SensoryInputPeepo(SensoryInput):

    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def value(self, name):
        if VISION.lower() in name.lower():
            return [0.1, 0.9] if self.peepo.view[self.get_quadrant(name)] else [0.9, 0.1]
        if MOTOR.lower() in name.lower():
            return [0.1, 0.9] if self.peepo.motor[self.get_direction(name)] else [0.9, 0.1]
        if ID_ENNEMY.lower() in name.lower():
            return [0.1, 0.9] if self.peepo.is_an_enemy else [0.9, 0.1]
        if ID_FOOD.lower() in name.lower():
            return [0.1, 0.9] if self.peepo.is_food else [0.9, 0.1]

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


class Ennemies:
    """
    This organism represents an obstacle. It is an inanimate object and therefore only takes a position parameter.
    """

    SIZE = (10, 10)

    def __init__(self, name, pos, graphical):
        self.rect = pg.Rect((0, 0),Ennemies.SIZE)
        self.rect.center = pos
        if graphical:
            self.image = self.make_image()
        self.name = name

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("darkgrey"), image_rect.inflate(-2, -2))
        return image

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def __str__(self):
        return '[' + str(self.rect.x) + ', ' + str(self.rect.y) + ']'

    def __repr__(self):
        return '[' + str(self.rect.x) + ', ' + str(self.rect.y) + ']'


class Food:
    """
    This organism represents an obstacle. It is an inanimate object and therefore only takes a position parameter.
    """

    SIZE = (10, 10)

    def __init__(self, name, pos, graphical):
        self.rect = pg.Rect((0, 0), Food.SIZE)
        self.rect.center = pos
        if graphical:
            self.image = self.make_image()
        self.name = name

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("palegreen"), image_rect.inflate(-2, -2))
        return image

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def __str__(self):
        return '[' + str(self.rect.x) + ', ' + str(self.rect.y) + ']'

    def __repr__(self):
        return '[' + str(self.rect.x) + ', ' + str(self.rect.y) + ']'