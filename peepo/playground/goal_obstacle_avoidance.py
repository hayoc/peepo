import math
import os
import random
import sys

import pygame as pg

from peepo.playground.models.goal_obstacle_advoidance_model import PeepoModel
from peepo.playground.models.goal_obstacle_avoidance_peepo import Peepo
from peepo.playground.util.vision import end_line

vec = pg.math.Vector2

CAPTION = "Peepo 's World"
SCREEN_SIZE = (1600, 1000)
SCREEN_CENTER = (800, 500)
TRANSPARENT = (0, 0, 0, 0)
DIRECT_DICT = {pg.K_LEFT: (-1, 0),
               pg.K_RIGHT: (1, 0),
               pg.K_UP: (0, -1),
               pg.K_DOWN: (0, 1)}

START_POS = (50, 50)
GOAL_POS = (1500, 800)


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
        self.rotation = 0
        self.edge_right = end_line(PeepoModel.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(PeepoModel.RADIUS, self.rotation - 30, self.rect.center)
        self.peepo = Peepo(self, GOAL_POS)

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


class ObjectActor(object):

    def __init__(self, id, pos, size, color):
        self.id = id
        self.rect = pg.Rect((0, 0), size)
        self.rect.center = pos
        self.color = color
        self.image = self.make_image()

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, self.color, image_rect)
        pg.draw.rect(image, self.color, image_rect.inflate(-2, -2))
        return image

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class PeeposWorld(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """

    def __init__(self, peepo, objects):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.keys = pg.key.get_pressed()
        self.peepo = peepo
        self.objects = objects

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
        for obj in self.objects:
            obj.draw(self.screen)
        self.peepo.draw(self.screen)

        pg.display.update()

    def main_loop(self):
        """
        One game loop. Simple and clean.
        """
        while not self.done:
            self.event_loop()
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

    wall1 = ObjectActor('wall_up', (0, 0), (3200, 5), pg.Color('brown'))
    wall2 = ObjectActor('wall_left', (0, 0), (5, 2000), pg.Color('brown'))
    wall3 = ObjectActor('wall_right', (1598, 0), (5, 2000), pg.Color('brown'))
    wall4 = ObjectActor('wall_down', (0, 998), (3200, 5), pg.Color('brown'))

    obstacles = []
    for x in range(0, 50):
        obstacles.append(ObjectActor('obj_' + str(x), (random.randint(100, 1500), random.randint(100, 900)), (20, 20),
                                     pg.Color('red')))
    goal = ObjectActor('goal', GOAL_POS, (50, 50), pg.Color('blue'))
    obstacles.extend([wall1, wall2, wall3, wall4, goal])

    peepo = PeepoActor(START_POS, obstacles)

    world = PeeposWorld(peepo, obstacles)

    world.main_loop()
    pg.quit()
    sys.exit()


if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
