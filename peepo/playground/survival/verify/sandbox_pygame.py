import json
import logging
import os
import sys

import pygame as pg

from peepo.playground.survival.verify.organism_pygame import Food, Peepo
from peepo.predictive_processing.v3 import peepo_network

CAPTION = "Survival"
SCREEN_SIZE = (800, 800)
SCREEN_CENTER = (400, 400)


def read_food():
    food_stuff = []
    with open('../food.json') as json_data:
        for f in json.load(json_data):
            food_stuff.append(Food(f['name'], (f['x'], f['y'])))
    return food_stuff


class World(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """

    def __init__(self):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.keys = pg.key.get_pressed()
        self.food = read_food()
        self.peepo = Peepo('peepo', peepo_network.read_from_file('best_survival_network'), (5, 5), self.food)

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
        for obj in self.food:
            obj.draw(self.screen)
        self.peepo.draw(self.screen)

        pg.display.update()

    def main_loop(self):
        while not self.done:
            self.event_loop()
            self.peepo.update()
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

    world = World()

    world.main_loop()

    pg.quit()
    sys.exit()


"""
####################################################################################
############################### BEGIN HERE #########################################
####################################################################################
"""
if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
