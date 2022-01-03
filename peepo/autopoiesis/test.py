import pygame as pg
import sys
import math


CAPTION = "Autopoiesis"
SCREEN_SIZE = (800, 800)
vec = pg.math.Vector2


class World:

    def __init__(self):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False

        self.first = Block((400, 400), 0)
        self.second = self.create_new(self.first)
        self.third = self.create_new(self.second)
        self.fourth = self.create_new(self.third)
        self.fifth = self.create_new(self.fourth)
        self.sixth = self.create_new(self.fifth)
        self.seventh = self.create_new(self.sixth)
        self.eight = self.create_new(self.seventh)
        self.ninth = self.create_new(self.eight)
        self.ten = self.create_new(self.ninth)
        self.eleven = self.create_new(self.ten)
        self.twelve = self.create_new(self.eleven)
        self.thirteen = self.create_new(self.twelve)
        self.fourteen = self.create_new(self.thirteen)
        self.fifteen = self.create_new(self.fourteen)

    def create_new(self, base_block):
        angle = 25  # +25 for a left attach, -25 for a right attach
        radius = 50  # radius of circle of cell

        origin_x = base_block.rect.centerx - radius * math.cos(math.radians(base_block.rotation))
        origin_y = base_block.rect.centery - radius * math.sin(math.radians(base_block.rotation))

        new_rotation = modify_degrees(base_block.rotation, angle)
        new_x = origin_x + radius * math.cos(math.radians(new_rotation))
        new_y = origin_y + radius * math.sin(math.radians(new_rotation))

        second = Block((new_x, new_y), new_rotation)
        return second


    def main_loop(self):
        while not self.done:
            self.event_loop()
            self.render()
            self.clock.tick(self.fps)

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

    def render(self):
        self.screen.fill(pg.Color("white"))
        self.first.draw(self.screen)
        self.second.draw(self.screen)
        self.third.draw(self.screen)
        self.fourth.draw(self.screen)
        self.fifth.draw(self.screen)
        self.sixth.draw(self.screen)
        self.seventh.draw(self.screen)
        self.eight.draw(self.screen)
        self.ninth.draw(self.screen)
        self.ten.draw(self.screen)
        self.eleven.draw(self.screen)
        self.twelve.draw(self.screen)
        self.thirteen.draw(self.screen)
        self.fourteen.draw(self.screen)
        self.fifteen.draw(self.screen)


        pg.display.update()


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


class Block:

    def __init__(self, pos, rotation):
        self.rect = pg.Rect(pos, (10, 10))
        self.rect.center = pos
        self.rotation = rotation

        self.image = self.make_image()

        self.edge_center = end_line(10, self.rotation, self.rect.center)

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        # myfont = pg.font.SysFont("Comic Sans MS", 8)
        # label = myfont.render("{}r - {}p".format(self.rotation, (self.rect.x, self.rect.y)), True, pg.Color("red"))
        # surface.blit(label, self.rect)
        pg.draw.line(surface, pg.Color("red"), self.rect.center, self.edge_center, 2)

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill((0, 0, 0, 0))
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image


def end_line(radius, rotation, center):
    center_rotate = vec(radius, 0).rotate(rotation)
    return center_rotate + center


def run():
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    world = World()
    world.main_loop()

    pg.quit()
    sys.exit()



if __name__ == "__main__":
    run()