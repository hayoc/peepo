import pygame as pg

from peepo.pp.peepo import Peepo


class World:

    def __init__(self, organism: Peepo):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False

        self.organism = organism

    def main_loop(self):
        while not self.done:
            self.organism.update()

            self.event_loop()
            self.render()
            self.clock.tick(self.fps)

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

    def render(self):
        self.screen.fill(pg.Color("white"))
        self.organism.draw(self.screen)

        pg.display.update()
