import math
import os
import random
import sys

import pygame as pg
import numpy as np
from peepoHawk.playground.models.wandering_obstacle_avoidance_model import PeepoModel
from peepoHawk.playground.models.wandering_obstacle_avoidance_peepo import Peepo

from peepoHawk.playground.util.vision import end_line
from peepoHawk.playground.Performance.performance import \
    Metrics  # NOT used for the moment: intended to measure te effectiveness of the leanning rate

vec = pg.math.Vector2

CAPTION = "Peepo 's World"
SCREEN_SIZE = (1400, 750)
WALL_SIZE = (SCREEN_SIZE[0], SCREEN_SIZE[1])
SCREEN_CENTER = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2)
TRANSPARENT = (0, 0, 0, 0)
DIRECT_DICT = {pg.K_LEFT: (-1, 0),
               pg.K_RIGHT: (1, 0),
               pg.K_UP: (0, -1),
               pg.K_DOWN: (0, 1)}


class PoopieActor(object):
    """ This class represents a Poopie; the victim
        More than 1 Poopie can be present but this is for later maybe"""

    SIZE = (40, 40)
    MAXSPEED = 5  # the speed will be different for each run

    def __init__(self, number_of_poopies, screen_rect):
        self.speed = random.randint(1, PoopieActor.MAXSPEED)  # a random speed between 1 and MAXSPEED
        self.number_of_poopies = number_of_poopies
        self.screen_rect = screen_rect
        self.tensor_of_poopies = np.zeros(shape=(number_of_poopies, 4))
        self.first_tensor()

    def first_tensor(self):
        for row in range(0, self.number_of_poopies):
            self.tensor_of_poopies[row][
                2] = self.speed  # the speed of the poopies (uniform for all of them, for the moment being
            self.tensor_of_poopies[row][3] = random.uniform(0.05 * math.pi, 0.99 * math.pi)
            # the Poopie start at the upper side, somewhere in the second halve of the width
            self.tensor_of_poopies[row][0] = random.uniform(WALL_SIZE[0] / 2, WALL_SIZE[0])
            self.tensor_of_poopies[row][1] = 0  # random.uniform(0, WALL_SIZE[1])

    def get_poopies(self):
        return self.tensor_of_poopies

    def get_poopies_obstacles(self):
        obstacles = []
        for row in range(0, self.number_of_poopies):
            obstacles.append(
                PoopieObject('obj_' + str(row), (self.tensor_of_poopies[row][0], self.tensor_of_poopies[row][1])))
        return obstacles

    def update(self, screen_rect):

        for row in range(0, self.number_of_poopies):
            self.tensor_of_poopies[row][0] += self.tensor_of_poopies[row][2] * math.cos(self.tensor_of_poopies[row][3])
            self.tensor_of_poopies[row][1] += self.tensor_of_poopies[row][2] * math.sin(self.tensor_of_poopies[row][3])
            # once the Poopie has reached safely a wall, he rests and stays there
            if self.tensor_of_poopies[row][0] >= WALL_SIZE[0]:
                self.speed = 0
                self.tensor_of_poopies[row][2] = self.speed
            if self.tensor_of_poopies[row][1] >= WALL_SIZE[1]:
                self.speed = 0
                self.tensor_of_poopies[row][2] = self.speed
            if self.tensor_of_poopies[row][0] <= 0:
                self.speed = 0
                self.tensor_of_poopies[row][2] = self.speed
            if self.tensor_of_poopies[row][1] <= 0:
                self.speed = 0
                self.tensor_of_poopies[row][2] = self.speed


class PoopieObject(object):
    SIZE = (20, 20)

    def __init__(self, id, pos):
        self.rect = pg.Rect((0, 0), PoopieObject.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.id = id

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("red"), image_rect.inflate(-2, -2))
        return image

    def update(self):
        pass

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class PeepoActor(object):
    """ This class represents peepo """

    SIZE = (40, 40)
    SPEED = 3

    def __init__(self, pos, actors):
        self.model = PeepoModel(self, actors)
        self.rect = pg.Rect((0, 0), PeepoActor.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.image_original = self.image.copy()
        self.peepo = Peepo()
        self.rotation = 0
        self.edge_right = end_line(PeepoModel.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(PeepoModel.RADIUS, self.rotation - 30, self.rect.center)

    def update(self, screen_rect):
        self.model.process()

        self.rect.x += PeepoActor.SPEED * math.cos(math.radians(self.rotation))
        self.rect.y += PeepoActor.SPEED * math.sin(math.radians(self.rotation))

        if self.model.motor_output[pg.K_LEFT]:
            self.rotation -= random.randint(10, 30)
            if self.rotation < 0:
                self.rotation = 360
        if self.model.motor_output[pg.K_RIGHT]:
            self.rotation += random.randint(10, 30)
            if self.rotation > 360:
                self.rotation = 0

        self.image = pg.transform.rotate(self.image_original, -self.rotation)
        self.rect = self.image.get_rect(center=self.rect.center)

        self.edge_right = end_line(PeepoModel.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(PeepoModel.RADIUS, self.rotation - 30, self.rect.center)

        self.rect.clamp_ip(screen_rect)
        self.peepo.update(self.model)

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


class Wall(object):

    def __init__(self, id, pos, size):
        self.id = id
        self.rect = pg.Rect((0, 0), size)
        self.rect.center = pos
        self.image = self.make_image()

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("brown"), image_rect)
        pg.draw.rect(image, pg.Color("brown"), image_rect.inflate(-1, -1))
        return image

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class PeeposWorld(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """

    def __init__(self, peepo, objects, poopies, metrics):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.keys = pg.key.get_pressed()
        self.peepo = peepo
        self.objects = objects
        self.poopies = poopies
        self.metrics = metrics

    def event_loop(self):
        """
        One event loop. Never cut your game off from the event loop.
        Your OS may decide your program has hung if the event queue is not
        accessed for a prolonged period of time.
        Stuff like user input can be processed here
        """
        for event in pg.event.get():
            if event.type == pg.QUIT or self.keys[pg.K_ESCAPE]:
                self.done = True
            elif event.type in (pg.KEYUP, pg.KEYDOWN):
                self.keys = pg.key.get_pressed()

    def render(self):
        """
        Perform all necessary drawing and update the screen.
        """
        self.screen.fill(pg.Color("white"))
        for obj in self.objects:
            obj.draw(self.screen)
        self.peepo.draw(self.screen)

        pg.display.update()

    def main_loop(self):
        """
        Game loop
        """
        while not self.done:
            self.event_loop()
            self.peepo.update(self.screen_rect)
            self.poopies.update(self.screen_rect)
            wall1 = Wall('wall_up', (0, 0), (WALL_SIZE[0] * 2, 5))
            wall2 = Wall('wall_left', (0, 0), (5, WALL_SIZE[1] * 2))
            wall3 = Wall('wall_right', (WALL_SIZE[0], 0), (5, WALL_SIZE[1] * 2))
            wall4 = Wall('wall_down', (0, WALL_SIZE[1]), (WALL_SIZE[0] * 2, 5))
            obstacles = self.poopies.get_poopies_obstacles()
            obstacles.extend([wall1, wall2, wall3, wall4])
            self.objects = obstacles
            self.render()
            self.clock.tick(self.fps)


def main():
    """
    Prepare our environment, create a display, and start the program (pygame).

    Initialize the game screen with the actors: walls, obstacles and peepo
    """
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    wall1 = Wall('wall_up', (0, 0), (WALL_SIZE[0], 5))
    wall2 = Wall('wall_left', (0, 0), (5, WALL_SIZE[1]))
    wall3 = Wall('wall_right', (WALL_SIZE[0], 0), (5, WALL_SIZE[1]))
    wall4 = Wall('wall_down', (0, WALL_SIZE[1]), (WALL_SIZE[0], 5))
    Max_Epochs = 1
    Epoch = 0
    metrics = Metrics(Epoch,
                      Max_Epochs)  # not used for the moment, intented to run Max_Epochs runs to assess statistically the effectiveness of the model
    poopies = PoopieActor(1, WALL_SIZE)  # class adress for the poopies
    obstacles = poopies.get_poopies_obstacles()
    obstacles.extend([wall1, wall2, wall3, wall4])
    peepo = PeepoActor((0, WALL_SIZE[1] / 2), obstacles)
    world = PeeposWorld(peepo, obstacles, poopies, metrics)
    world.main_loop()

    pg.quit()
    sys.exit()


"""
####################################################################################
############################### BEGIN HERE #########################################
####################################################################################
"""
if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
