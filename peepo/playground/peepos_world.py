import math
import logging
import os, sys

import pygame as pg
import numpy as np

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput

CAPTION = "Peepo 's World"
SCREEN_SIZE = (800, 600)
TRANSPARENT = (0, 0, 0, 0)
DIRECT_DICT = {pg.K_LEFT: (-1, 0),
               pg.K_RIGHT: (1, 0),
               pg.K_UP: (0, -1),
               pg.K_DOWN: (0, 1)}


class Human(object):
    """ This class represents a human """

    SIZE = (20, 20)

    def __init__(self, pos, speed):
        self.rect = pg.Rect((0, 0), Human.SIZE)
        self.rect.center = pos
        self.speed = speed
        self.image = self.make_image()

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("red"), image_rect.inflate(-2, -2))
        return image

    def update(self, keys, screen_rect):
        for key in DIRECT_DICT:
            if keys[key]:
                self.rect.x += DIRECT_DICT[key][0] * self.speed
                self.rect.y += DIRECT_DICT[key][1] * self.speed
        self.rect.clamp_ip(screen_rect)  # Keep player on screen.

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class Peepo(object):
    """ This class represents peepo """

    SIZE = (40, 40)
    DISTANCE = 75

    def __init__(self, pos, speed, actors):
        self.rect = pg.Rect((0, 0), Peepo.SIZE)
        self.rect.center = pos
        self.speed = speed
        self.image = self.make_image()
        self.actors = actors
        self.movingleft = False
        self.movingright = False
        self.movingup = False
        self.movingdown = False
        self.obstacleleft = False
        self.obstacleright = False
        self.obstacleup = False
        self.obstacledown = False

        self.modelLeft = GenerativeModel(SensoryInputVirtualPeepoLeft(self), self.create_network())
        self.modelRight = GenerativeModel(SensoryInputVirtualPeepoRight(self), self.create_network())
        self.modelUp = GenerativeModel(SensoryInputVirtualPeepoUp(self), self.create_network())
        self.modelDown = GenerativeModel(SensoryInputVirtualPeepoDown(self), self.create_network())

    def create_network(self):
        network = BayesianModel([('hypo', 'infrared'), ('hypo', 'motor')])
        cpd_a = TabularCPD(variable='hypo', variable_card=2, values=[[0.7, 0.3]])
        cpd_b = TabularCPD(variable='infrared', variable_card=2, values=[[0.9, 0.1],
                                                                         [0.1, 0.9]],
                           evidence=['hypo'],
                           evidence_card=[2])
        cpd_c = TabularCPD(variable='motor', variable_card=2, values=[[0.9, 0.1],
                                                                      [0.1, 0.9]],
                           evidence=['hypo'],
                           evidence_card=[2])
        network.add_cpds(cpd_a, cpd_b, cpd_c)
        network.check_model()
        return network

    def update(self, screen_rect):
        self.calculate_obstacles()

        self.modelLeft.process()
        self.modelRight.process()
        self.modelUp.process()
        self.modelDown.process()

        if self.movingup:
            self.rect.y += DIRECT_DICT[pg.K_UP][1] * self.speed
        if self.movingdown:
            self.rect.y += DIRECT_DICT[pg.K_DOWN][1] * self.speed
        if self.movingleft:
            self.rect.x += DIRECT_DICT[pg.K_LEFT][0] * self.speed
        if self.movingright:
            self.rect.x += DIRECT_DICT[pg.K_RIGHT][0] * self.speed

        self.rect.clamp_ip(screen_rect)  # Keep player on screen.

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def calculate_obstacles(self):
        for actor in self.actors:
            self.obstacleleft = actor.rect.x > self.rect.x and math.hypot(actor.rect.x - self.rect.x, actor.rect.y - self.rect.y) < float(Peepo.DISTANCE)
            self.obstacleright = actor.rect.x < self.rect.x and math.hypot(actor.rect.x - self.rect.x, actor.rect.y - self.rect.y) < float(Peepo.DISTANCE)
            self.obstacleup = actor.rect.y > self.rect.y and math.hypot(actor.rect.x - self.rect.x, actor.rect.y - self.rect.y) < float(Peepo.DISTANCE)
            self.obstacledown = actor.rect.y < self.rect.y and math.hypot(actor.rect.x - self.rect.x, actor.rect.y - self.rect.y) < float(Peepo.DISTANCE)


    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image


class SensoryInputVirtualPeepoLeft(SensoryInput):

    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def action(self, node, prediction_error, prediction):
        # if prediction = [0.1, 0.9] (= moving) then move else stop
        if np.argmax(prediction) > 0:  # predicted moving
            self.peepo.movingleft = True
        else:  # predicted stopped
            self.peepo.movingleft = False

    def value(self, name):
        if name == 'infrared':
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            return np.array([0.1, 0.9] if self.peepo.obstacleleft else np.array([0.9, 0.1]))
        else:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            return np.array([0.1, 0.9]) if self.peepo.movingleft else np.array([0.9, 0.1])


class SensoryInputVirtualPeepoRight(SensoryInput):

    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def action(self, node, prediction_error, prediction):
        # if prediction = [0.1, 0.9] (= moving) then move else stop
        if np.argmax(prediction) > 0:  # predicted moving
            self.peepo.movingright = True
        else:  # predicted stopped
            self.peepo.movingright = False

    def value(self, name):
        if name == 'infrared':
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            return np.array([0.1, 0.9] if self.peepo.obstacleright else np.array([0.9, 0.1]))
        else:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            return np.array([0.1, 0.9]) if self.peepo.movingright else np.array([0.9, 0.1])


class SensoryInputVirtualPeepoUp(SensoryInput):

    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def action(self, node, prediction_error, prediction):
        # if prediction = [0.1, 0.9] (= moving) then move else stop
        if np.argmax(prediction) > 0:  # predicted moving
            self.peepo.movingup = True
        else:  # predicted stopped
            self.peepo.movingup = False

    def value(self, name):
        if name == 'infrared':
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            return np.array([0.1, 0.9] if self.peepo.obstacleup else np.array([0.9, 0.1]))
        else:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            return np.array([0.1, 0.9]) if self.peepo.movingup else np.array([0.9, 0.1])


class SensoryInputVirtualPeepoDown(SensoryInput):

    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def action(self, node, prediction_error, prediction):
        # if prediction = [0.1, 0.9] (= moving) then move else stop
        if np.argmax(prediction) > 0:  # predicted moving
            self.peepo.movingdown = True
        else:  # predicted stopped
            self.peepo.movingdown = False

    def value(self, name):
        if name == 'infrared':
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            return np.array([0.1, 0.9] if self.peepo.obstacledown else np.array([0.9, 0.1]))
        else:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            return np.array([0.1, 0.9]) if self.peepo.movingdown else np.array([0.9, 0.1])


class PeeposWorld(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """

    def __init__(self):
        """
        Get a reference to the display surface; set up required attributes;
        and create a Player instance.
        """
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.keys = pg.key.get_pressed()
        self.player = Human((0, 0), 5)
        self.peepo = Peepo(self.screen_rect.center, 5, [self.player])

    def event_loop(self):
        """
        One event loop. Never cut your game off from the event loop.
        Your OS may decide your program has hung if the event queue is not
        accessed for a prolonged period of time.
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
        self.player.draw(self.screen)
        self.peepo.draw(self.screen)
        pg.display.update()

    def main_loop(self):
        """
        One game loop. Simple and clean.
        """
        while not self.done:
            self.event_loop()
            self.player.update(self.keys, self.screen_rect)
            self.peepo.update(self.screen_rect)
            self.render()
            self.clock.tick(self.fps)


def main():
    """
    Prepare our environment, create a display, and start the program.
    """
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)
    PeeposWorld().main_loop()
    pg.quit()
    sys.exit()


if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
