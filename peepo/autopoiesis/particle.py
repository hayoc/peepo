import pygame as pg
import math
import random

vec = pg.math.Vector2

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

    def __init__(self, kind, others, id, pos=None):
        self.kind = kind
        self.others = others
        self.id = id

        if pos is None:
            pos = (random.randint(0, 800), random.randint(0, 800))

        self.rect = pg.Rect(pos, Particle.KIND_SIZE[self.kind])
        self.rect.center = pos
        self.rotation = random.randint(0, 360)

        self.image = self.make_image()
        self.image_original = self.image.copy()

        self.timestep = 0
        self.disintegration_chance = 1  # 1 / 10,000,000 chance


        self.bonded_link = None
        self.bonded = False
        self.is_starter = False
        self.start_bond = None

        self.bond_prev = None
        self.bond_next = None
        self.bonds = []
        self.stop_bonding = False
        self.bond_left = None
        self.bond_right = None

        self.clamp = False

        if self.kind == "L":
            self.edge_left = end_line(5, self.rotation - 90, self.rect.center)
            self.edge_right = end_line(5, self.rotation + 90, self.rect.center)
            self.edge_center = end_line(10, self.rotation, self.rect.center)

    def update(self):
        if not self.bonded:
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

            if self.kind == "L":
                self.edge_center = end_line(10, self.rotation, self.rect.center)
                self.edge_left = end_line(10, self.rotation - 90, self.rect.center)
                self.edge_right = end_line(10, self.rotation + 90, self.rect.center)

        if self.kind == "K":
            self.production()
        if self.kind == "L":
            self.bonding()
            self.disintegration()

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        if self.kind == "L":
            # myfont = pg.font.SysFont("Comic Sans MS", 15)
            # # label = myfont.render("{}r - {}p".format(self.rotation, (self.rect.x, self.rect.y)), True, pg.Color("red"))
            # label = myfont.render("{}".format(self.id), True, pg.Color("red"))
            # surface.blit(label, self.rect)

            if not self.bond_right:
                pg.draw.line(surface, pg.Color("pink"), self.rect.center, self.edge_right, 2)
            else:
                pg.draw.line(surface, pg.Color("green"), self.rect.center, self.edge_right, 2)

            if not self.bond_left:
                pg.draw.line(surface, pg.Color("pink"), self.rect.center, self.edge_left, 2)
            else:
                pg.draw.line(surface, pg.Color("green"), self.rect.center, self.edge_left, 2)

            pg.draw.line(surface, pg.Color("red"), self.rect.center, self.edge_center, 2)

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

                        new_x = start_x + dist_x / 2
                        new_y = start_y + dist_y / 2

                        self.others.remove(sub0)
                        self.others.remove(sub1)
                        self.others.append(Particle("L", self.others, "L-{}".format(len(self.others)), (new_x, new_y)))

                        collided.clear()

    def bonding(self):
        # so basically whenever a particle collides with another, we push that other all round to the last
        # particle that has an open edge
        for particle in list(self.others):
            if particle.kind == "L" and particle is not self:
                if self.rect.colliderect(particle.rect):
                    # We close the circle here
                    if (self.bonded and not self.bonded_link) and particle.bonded_link:
                        self.bonded_link = particle
                        particle.is_starter = True

                    # Open particle to be bonded
                    if not self.bonded:
                        self.do_bond(self, particle)

    def do_bond(self, first, second):
        source, other = get_bonded_or_none(first, second)

        if source == "Fuck":
            return

        if source:
            if not source.is_starter:
                self.do_bond(source.bonded_link, other)
        else:
            print("{} bonded to {}".format(first.id, second.id))
            first.bonded_link = second
            first.bonded, second.bonded = (True, True)

            angle = 25  # +25 for a left attach, -25 for a right attach
            radius = 50  # radius of the theoretical circle of cell

            origin_x = first.rect.centerx - radius * math.cos(math.radians(first.rotation))
            origin_y = first.rect.centery - radius * math.sin(math.radians(first.rotation))

            second.rotation = modify_degrees(first.rotation, angle)
            second.rect.centerx = origin_x + radius * math.cos(math.radians(second.rotation))
            second.rect.centery = origin_y + radius * math.sin(math.radians(second.rotation))

            second.edge_left = end_line(10, second.rotation - 90, second.rect.center)
            second.edge_right = end_line(10, second.rotation + 90, second.rect.center)
            second.edge_center = end_line(10, second.rotation, second.rect.center)

    def disintegration(self):
        self.disintegration_chance += 1
        disintegrate = random.choices([True, False], weights=[self.disintegration_chance, 1000000000], k=1)[0]

        if disintegrate:
            if self.bond_left:
                self.bond_left.bond_left = None
            if self.bond_right:
                self.bond_right.bond_right = None
            self.others.remove(self)
            self.others.append(Particle("S", self.others, "S-{}".format(len(self.others)), (self.rect.x, self.rect.y)))
            self.others.append(Particle("S", self.others, "S-{}".format(len(self.others)), (self.rect.x, self.rect.y)))

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill((0, 0, 0, 0))
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color(Particle.KIND_COLOR[self.kind]), image_rect.inflate(-2, -2))
        return image


def get_bonded_or_none(one: Particle, two: Particle):
    if one.bonded_link and two.bonded_link:
        print("Fuck: {} - {}".format(one.id, two.id))
        return "Fuck", "This"
    if one.bonded_link:
        return one, two
    if two.bonded_link:
        return two, one
    return None, None


def end_line(radius, rotation, center):
    center_rotate = vec(radius, 0).rotate(rotation)
    return center_rotate + center


def identify_lone_link(one: Particle, two: Particle):
    if not one.bond_left and not one.bond_right:
        return two, one
    if not two.bond_left and not two.bond_right:
        return one, two
    return None, None


def modify_degrees(start, add):
    if add > 0:
        if start + add > 360:
            return start + add - 360
        return start + add
    elif add < 0:
        if start + add < 0:
            return start + add + 360
        return start + add
    else:
        return start
