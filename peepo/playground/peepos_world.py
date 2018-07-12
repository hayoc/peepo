import os, sys

import pygame as pg

from peepo.playground.peepos_model import PeepoModel

CAPTION = "Peepo 's World"
SCREEN_SIZE = (800, 600)
SCREEN_CENTER = (400, 300)
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
        self.rect.clamp_ip(screen_rect)

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class Peepo(object):
    """ This class represents peepo """

    def __init__(self, pos, speed, model):
        self.model = model
        self.rect = self.model.rect
        self.rect.center = pos
        self.speed = speed
        self.image = self.make_image()

    def update(self, screen_rect):
        self.model.process()
        for key in DIRECT_DICT:
            if self.model.motor_output[key]:
                self.rect.x += DIRECT_DICT[key][0] * self.speed
                self.rect.y += DIRECT_DICT[key][1] * self.speed
        self.rect.clamp_ip(screen_rect)

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image


class PeeposWorld(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """

    def __init__(self, peepo, human):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.keys = pg.key.get_pressed()
        self.human = human
        self.peepo = peepo

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
        self.human.draw(self.screen)
        self.peepo.draw(self.screen)
        pg.display.update()

    def main_loop(self):
        """
        One game loop. Simple and clean.
        """
        while not self.done:
            self.event_loop()
            self.human.update(self.keys, self.screen_rect)
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

    human = Human((0, 0), 5)
    peepo = Peepo(SCREEN_CENTER, 5, PeepoModel([human]))
    world = PeeposWorld(peepo, human)

    world.main_loop()
    pg.quit()
    sys.exit()


if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
