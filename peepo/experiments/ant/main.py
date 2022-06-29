import logging
import os
import sys

import pygame as pg

from peepo.experiments.ant.organism import AntPeepo
from peepo.experiments.base.world import World
from peepo.pp.peepo_network import read_from_file


CAPTION = "survival"
SCREEN_SIZE = (800, 800)
SCREEN_CENTER = (400, 400)


def run():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    peepo_network = read_from_file("ant_peepo")

    peepo = AntPeepo('ant-peepo', peepo_network, (5, 400))
    world = World(peepo)

    world.main_loop()

    pg.quit()
    sys.exit()


if __name__ == "__main__":
    run()