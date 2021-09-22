import pygame as pg
import math
import random


SCREEN_SIZE = (800, 800)


class Particle:

    SIZE = (4, 4)
    SPEED = 2

    KIND_COLOR = {
        "S": "grey",
        "K": "green",
        "L": "blue"
    }

    KIND_SIZE = {
        "S": (4, 4),
        "K": (6, 6),
        "L": (4, 4)
    }

    def __init__(self, kind):
        self.kind = kind

        self.pos = (random.randint(0, 800), random.randint(0, 800))

        self.rect = pg.Rect(self.pos, Particle.KIND_SIZE[self.kind])
        self.rect.center = self.pos
        self.rotation = random.randint(0, 360)

        self.image = self.make_image()
        self.image_original = self.image.copy()

        self.timestep = 0

    def update(self):
        self.rotation += random.randint(-5, 5)
        if self.rotation < 0:
            self.rotation = 360
        if self.rotation > 360:
            self.rotation = 0

        self.timestep += 1
        if self.timestep > 4:
            self.rect.x += Particle.SPEED * math.cos(math.radians(self.rotation))
            self.rect.y += Particle.SPEED * math.sin(math.radians(self.rotation))
            self.timestep = 0

        if self.rect.x < 0:
            self.rect.x = 800
        if self.rect.y < 0:
            self.rect.y = 800
        if self.rect.x > 800:
            self.rect.x = 0
        if self.rect.y > 800:
            self.rect.y = 0

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill((0, 0, 0, 0))
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color(Particle.KIND_COLOR[self.kind]), image_rect.inflate(-2, -2))
        return image

