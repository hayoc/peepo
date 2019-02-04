import math
import random
import numpy as np
import pygame as pg

from peepo.playground.game_of_life.vision import collision, end_line
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput

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

    def __init__(self, name, network, graphical, pos=(0, 0), ennemies=None, food=None):
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
        self.obstacles = []
        for i, x in enumerate(self.food):
            self.obstacles.append([x, 0, i])
        for i, x in enumerate(self.ennemies):
            self.obstacles.append([x, 1, i])

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
        pg.draw.line(surface, pg.Color("blue"), self.rect.center, self.edge_middle, 4)
        pg.draw.circle(surface, pg.Color("grey"), self.rect.center, Peepo.RADIUS, 2)

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
        for obstacle in self.obstacles:
            if self.rect.colliderect(obstacle[0].rect):
                if obstacle[1] == 0:
                    self.stomach += 1
                    self.obstacles.remove(obstacle)
                else:
                    self.bang += 1
                    # self.obstacles.remove(obstacle)
        self.food = []
        [self.food.append(x[0]) for x in self.obstacles if x[1] == 0]
        # self.ennemies = []
        # [self.ennemies.append(x[0]) for x in self.obstacles if x[1] == 1]
        self.assemble_obstacles()
        observations = []
        for obstacle in self.obstacles:
            distance = math.sqrt((obstacle[0].rect.center[0] - self.rect.center[0])**2 +(obstacle[0].rect.center[1] - self.rect.center[1])**2)
            if distance <= Peepo.RADIUS:
                observations.append(obstacle)
        edge1 = end_line(Peepo.RADIUS, self.rotation - 30, self.rect.center)
        edge2 = end_line(Peepo.RADIUS, self.rotation - 20, self.rect.center)
        edge3 = end_line(Peepo.RADIUS, self.rotation - 10, self.rect.center)
        edge4 = end_line(Peepo.RADIUS, self.rotation, self.rect.center)
        edge5 = end_line(Peepo.RADIUS, self.rotation + 10, self.rect.center)
        edge6 = end_line(Peepo.RADIUS, self.rotation + 20, self.rect.center)
        edge7 = end_line(Peepo.RADIUS, self.rotation + 30, self.rect.center)
        sectors = [[edge1, edge2], [edge2, edge3], [edge3, edge4], [edge4, edge5], [edge5, edge6], [edge6, edge7]]
        peepo_vec = pg.math.Vector2(self.rect.center)
        relevant_sector = ["0", self.rect.center, Peepo.RADIUS]
        closest_distance = 10000.
        for index, sector in enumerate(sectors):
            lower_edge = sector[0]
            upper_edge = sector[1]
            for obstacle in observations:
                is_collision = collision(obstacle[0].rect, peepo_vec, lower_edge, upper_edge, Peepo.RADIUS)
                if is_collision:
                    distance = math.sqrt((obstacle[0].rect.center[0] - peepo_vec[0]) ** 2 + (
                                obstacle[0].rect.center[1] - peepo_vec[1]) ** 2)
                    if distance <= closest_distance:
                        closest_distance = distance
                        relevant_sector[0] = str(index + 1)
                        relevant_sector[1] = obstacle[0].rect.center
                        relevant_sector[2] = closest_distance
                        if obstacle[1] == 1:
                            self.is_an_enemy = True
                            self.is_food = False
                        if obstacle[1] == 0:
                            self.is_food = True
                            self.is_an_enemy = False
        self.view = {x: False for x in self.view}
        self.view[relevant_sector[0]] = True
        sight_angle = 0
        only_true = relevant_sector[0]
        if only_true == '0':
            sight_angle = self.rotation
        if only_true == '1':
            sight_angle = self.rotation - 25
        if only_true == '2':
            sight_angle = self.rotation - 15
        if only_true == '3':
            sight_angle = self.rotation - 5
        if only_true == '4':
            sight_angle = self.rotation + 5
        if only_true == '5':
            sight_angle = self.rotation + 15
        if only_true == '6':
            sight_angle = self.rotation + 25
        self.edge_middle = end_line(relevant_sector[2] / 20, sight_angle, relevant_sector[1])


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
        self.rect = pg.Rect((0, 0), Ennemies.SIZE)
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