import pygame as pg
import numpy as np



CAPTION = "Bacterial Chemotaxis"
SCREEN_SIZE = (800, 800)
plate_length = 80

alpha = 2
delta_x = 1

delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)


class World:

    def __init__(self):
        pg.init()
        pg.display.set_caption(CAPTION)
        pg.display.set_mode(SCREEN_SIZE)

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

    def main_loop(self):
        while not self.done:
            self.event_loop()
            self.render()
            self.clock.tick(self.fps)

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                if (i, j) == self.source_1 or (i, j) == self.source_2:
                    continue
                self.u[i, j] = gamma * (self.u[i+1][j] + self.u[i-1][j] + self.u[i][j+1] + self.u[i][j-1] - 4*self.u[i][j]) + self.u[i][j]

    def render(self):
        a = np.repeat(np.repeat(self.u, 10, 0), 10, 1)
        pg.surfarray.blit_array(self.screen, a)

        pg.display.update()


if __name__ == "__main__":
    world = World()
    world.main_loop()