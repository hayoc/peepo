import pygame as pg
import math
import random


SCREEN_SIZE = (800, 800)


class Particle:

    SIZE = (4, 4)
    SPEED = 2

    KIND_COLOR = {
        "S": "yellow",
        "K": "green",
        "L": "blue"
    }

    KIND_SIZE = {
        "S": (4, 4),
        "K": (6, 6),
        "L": (6, 6)
    }

    def __init__(self, kind, others, pos=None):
        self.kind = kind
        self.others = others

        if pos is None:
            pos = (random.randint(0, 800), random.randint(0, 800))

        self.rect = pg.Rect(pos, Particle.KIND_SIZE[self.kind])
        self.rect.center = pos
        self.rotation = random.randint(0, 360)

        self.image = self.make_image()
        self.image_original = self.image.copy()

        self.timestep = 0
        self.disintegration_chance = 1  # / 10000
        self.bondLeft = None
        self.bondRight = None

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
            self.image = pg.transform.rotate(self.image_original, -self.rotation)
            self.rect = self.image.get_rect(center=self.rect.center)
            self.timestep = 0

        if self.rect.x < 0:
            self.rect.x = 800
        if self.rect.y < 0:
            self.rect.y = 800
        if self.rect.x > 800:
            self.rect.x = 0
        if self.rect.y > 800:
            self.rect.y = 0

        if self.kind == "K":
            self.production()
        if self.kind == "L":
            self.bonding()
            self.disintegration()

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def production(self):
        collided = []

        for particle in list(self.others):
            if particle.kind == "S":
                collide = self.rect.colliderect(particle.rect)
                if collide:
                    collided.append(particle)

                    if len(collided) >= 2:
                        sub0 = collided[0]
                        sub1 = collided[1]

                        dist_x = abs(sub0.rect.x - sub1.rect.x)
                        dist_y = abs(sub0.rect.y - sub1.rect.y)

                        start_x = min(sub0.rect.x, sub1.rect.x)
                        start_y = min(sub0.rect.y, sub1.rect.y)

                        new_x = start_x + dist_x/2
                        new_y = start_y + dist_y/2

                        self.others.remove(sub0)
                        self.others.remove(sub1)
                        self.others.append(Particle("L", self.others, (new_x, new_y)))

                        collided.clear()

    def bonding(self):
        collided = []

        for particle in list(self.others):
            if particle.kind == "L":
                collide = self.rect.colliderect(particle.rect)

    def disintegration(self):
        self.disintegration_chance += 1
        disintegrate = random.choices([True, False], weights=[self.disintegration_chance, 10000000], k=1)[0]

        if disintegrate:
            self.others.remove(self)
            self.others.append(Particle("S", self.others, (self.rect.x, self.rect.y)))
            self.others.append(Particle("S", self.others, (self.rect.x, self.rect.y)))

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill((0, 0, 0, 0))
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color(Particle.KIND_COLOR[self.kind]), image_rect.inflate(-2, -2))
        return image

