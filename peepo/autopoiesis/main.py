import pygame as pg
import sys

from peepo.autopoiesis.particle import Particle
from peepo.autopoiesis.world import World

CAPTION = "Autopoiesis"
SCREEN_SIZE = (800, 800)


def run():
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    particles = []
    # for i in range(1000):
    #     particles.append(Particle("S", particles, "S-{}".format(i)))
    # for i in range(50):
    #     particles.append(Particle("K", particles, "K-{}".format(i)))

    world = World(particles)

    world.main_loop()

    pg.quit()
    sys.exit()


if __name__ == "__main__":
    run()