import sys, os
import pygame as pg

from peepo.experiments.base.world import World
from peepo.experiments.portia.organism import Portia
from peepo.pp.peepo_network import read_from_file

CAPTION = "Portia"
SCREEN_SIZE = (800, 800)


def run():
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    network = read_from_file('best_survival_network')

    portia = Portia("portia", network, (400, 400))
    world = World(portia)

    world.main_loop()

    pg.quit()
    sys.exit()


if __name__ == '__main__':
    run()
