import pygame as pg
import numpy as np

from peepo.experiments.chemotaxis.organism import Bacteria

plate_length = 80

alpha = 2
delta_x = 1

delta_t = (delta_x ** 2) / (4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)


class World:

    def __init__(self, peepos, graphical=True):
        self.graphical = graphical
        if self.graphical:
            self.screen = pg.display.get_surface()
            self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False

        self.u = np.empty((plate_length, plate_length))
        self.u.fill(0)

        self.source_1 = (25, 20)
        self.u[self.source_1] = 100.

        self.source_2 = (70, 61)
        self.u[self.source_2] = 100.

        self.u[(plate_length - 1):, :] = 0.
        self.u[:, :1] = 0.
        self.u[:1, 1:] = 0.
        self.u[:, (plate_length - 1):] = 0.

        self.peepos = peepos

    def main_loop(self, max_age):
        loop = 0

        while not self.done:
            for peepo in self.peepos:
                peepo.set_surroundings(self.get_surroundings(peepo))
                peepo.update()

            if self.graphical:
                self.event_loop()
                self.render()
                self.clock.tick(self.fps)

            loop += 1
            print(loop)
            if loop > max_age:
                for peepo in self.peepos:
                    print(peepo.health)
                break


    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

        for i in range(1, plate_length - 1, delta_x):
            for j in range(1, plate_length - 1, delta_x):
                if (i, j) == self.source_1 or (i, j) == self.source_2:
                    continue
                self.u[i, j] = gamma * (
                            self.u[i + 1][j] + self.u[i - 1][j] + self.u[i][j + 1] + self.u[i][j - 1] - 4 * self.u[i][
                        j]) + self.u[i][j]

    def render(self):
        a = np.repeat(np.repeat(self.u, 10, 0), 10, 1)
        pg.surfarray.blit_array(self.screen, a)

        for peepo in self.peepos:
            peepo.draw(self.screen)

        pg.display.update()

    def get_surroundings(self, peepo):
        x, y = peepo.get_pos()
        m = np.repeat(np.repeat(self.u, 10, 0), 10, 1)
        surrounding_size = 6
        surrounding_values = m[max(0, y - surrounding_size // 2):min(m.shape[0], y + surrounding_size // 2 + 1),
                               max(0, x - surrounding_size // 2):min(m.shape[1], x + surrounding_size // 2 + 1)]
        return surrounding_values
