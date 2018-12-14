import json
import logging
import math
import os
import random
import sys

import pygame as pg

from config import ROOT_DIR
from peepo.playground.util.vision import end_line
from peepo.playground.wandering.wandering_obstacle_avoidance_model import PeepoModel
from peepo.playground.wandering.wandering_obstacle_avoidance_peepo import Peepo
from peepo.utilities.bayesian_network import json_to_bayesian_network, random_mutation, bayesian_network_to_json

vec = pg.math.Vector2

CAPTION = "Peepo 's World"
SCREEN_SIZE = (1600, 1000)
SCREEN_CENTER = (800, 500)
TRANSPARENT = (0, 0, 0, 0)
DIRECT_DICT = {pg.K_LEFT: (-1, 0),
               pg.K_RIGHT: (1, 0),
               pg.K_UP: (0, -1),
               pg.K_DOWN: (0, 1)}


class PeepoActor(object):
    """ This class represents peepo """

    SIZE = (40, 40)
    SPEED = 4

    def __init__(self, pos, actors, generation, individual):
        self.model = PeepoModel(self, actors, generation, individual)
        self.rect = pg.Rect((0, 0), PeepoActor.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.image_original = self.image.copy()
        self.peepo = Peepo()
        self.rotation = 0
        self.edge_right = end_line(PeepoModel.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(PeepoModel.RADIUS, self.rotation - 30, self.rect.center)
        self.run = True

    def update(self, screen_rect):
        if self.run:
            self.model.process()

            self.rect.x += PeepoActor.SPEED * math.cos(math.radians(self.rotation))
            self.rect.y += PeepoActor.SPEED * math.sin(math.radians(self.rotation))

            if self.model.motor_output[pg.K_LEFT]:
                self.rotation -= 10
                if self.rotation < 0:
                    self.rotation = 360
            if self.model.motor_output[pg.K_RIGHT]:
                self.rotation += 10
                if self.rotation > 360:
                    self.rotation = 0

            self.image = pg.transform.rotate(self.image_original, -self.rotation)
            self.rect = self.image.get_rect(center=self.rect.center)

            self.edge_right = end_line(PeepoModel.RADIUS, self.rotation + 30, self.rect.center)
            self.edge_left = end_line(PeepoModel.RADIUS, self.rotation - 30, self.rect.center)

            self.rect.clamp_ip(screen_rect)
            # self.peepo.update(self.model)

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        pg.draw.line(surface, pg.Color("red"), self.rect.center, self.edge_right, 2)
        pg.draw.line(surface, pg.Color("green"), self.rect.center, self.edge_left, 2)

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        return image


class ObjectActor(object):
    SIZE = (20, 20)

    def __init__(self, id, pos):
        self.rect = pg.Rect((0, 0), ObjectActor.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.id = id

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("pink"), image_rect.inflate(-2, -2))
        return image

    def update(self):
        pass

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class Wall(object):

    def __init__(self, id, pos, size):
        self.id = id
        self.rect = pg.Rect((0, 0), size)
        self.rect.center = pos
        self.image = self.make_image()

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("brown"), image_rect)
        pg.draw.rect(image, pg.Color("brown"), image_rect.inflate(-1, -1))
        return image

    def draw(self, surface):
        surface.blit(self.image, self.rect)


def generate_obstacles():
    objects = []
    for x in range(0, 50):
        objects.append({
            'id': 'obj_' + str(x),
            'x': random.randint(20, 1580),
            'y': random.randint(20, 980)
        })
    with open('obstacles.json', 'w') as outfile:
        json.dump(objects, outfile)


class PeeposWorld(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """

    def __init__(self, objects):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.keys = pg.key.get_pressed()
        self.objects = objects
        self.current_generation = 0
        logging.info('New generation ' + str(self.current_generation))

        self.score0 = 0
        self.score1 = 0
        self.score2 = 0
        self.score3 = 0
        self.score4 = 0
        self.peepo0 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id0')
        self.peepo1 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id1')
        self.peepo2 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id2')
        self.peepo3 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id3')
        self.peepo4 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id4')

    def new_peepo(self):
        self.write_score(0, self.score0)
        self.write_score(1, self.score1)
        self.write_score(2, self.score2)
        self.write_score(3, self.score3)
        self.write_score(4, self.score4)
        self.score0 = 0
        self.score1 = 0
        self.score2 = 0
        self.score3 = 0
        self.score4 = 0

        self.peepo0 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id0')
        self.peepo1 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id1')
        self.peepo2 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id2')
        self.peepo3 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id3')
        self.peepo4 = PeepoActor((50, 500), self.objects,
                                 'gen' + str(self.current_generation), 'id4')

        self.new_generation()
        self.current_generation += 1
        logging.info('New generation ' + str(self.current_generation))

    def new_generation(self):
        best_score = 1000000
        best_individual = 0

        gen_dir = ROOT_DIR + '/resources/bn/' + 'gen' + str(self.current_generation)
        for individual in range(0, 5):
            with open(gen_dir + '/' + 'id' + str(individual) + '.json', 'r') as f:
                score = json.load(f)['score']
                if score < best_score:
                    best_score = score
                    best_individual = individual

        parent = json_to_bayesian_network('gen' + str(self.current_generation), 'id' + str(best_individual))
        for x in range(0, 5):
            child = random_mutation(parent)
            bayesian_network_to_json(child, 'gen' + str(self.current_generation + 1), 'id' + str(x))

    def write_score(self, individual, score):
        with open(ROOT_DIR + '/resources/bn/' + 'gen' + str(self.current_generation)
                  + '/' + 'id' + str(individual) + '.json') as f:
            json_object = json.load(f)

        json_object['score'] = score

        with open(ROOT_DIR + '/resources/bn/' + 'gen' + str(self.current_generation)
                  + '/' + 'id' + str(individual) + '.json', 'w') as f:
            json.dump(json_object, f)

    def event_loop(self):
        """
        One event loop. Never cut your game off from the event loop.
        Your OS may decide your program has hung if the event queue is not
        accessed for a prolonged period of time.
        Stuff like user input can be processed here
        """
        for event in pg.event.get():
            if event.type == pg.QUIT or self.keys[pg.K_ESCAPE]:
                self.done = True
            elif event.type in (pg.KEYUP, pg.KEYDOWN):
                self.keys = pg.key.get_pressed()

    def render(self):
        """
        Perform all necessary drawing and update the screen.
        """
        self.screen.fill(pg.Color("white"))
        for obj in self.objects:
            obj.draw(self.screen)
        self.peepo0.draw(self.screen)
        self.peepo1.draw(self.screen)
        self.peepo2.draw(self.screen)
        self.peepo3.draw(self.screen)
        self.peepo4.draw(self.screen)

        pg.display.update()

    def main_loop(self):
        """
        Game loop
        """
        loop = 0
        start = pg.time.get_ticks()
        while not self.done:
            self.event_loop()
            self.peepo0.update(self.screen_rect)
            self.peepo1.update(self.screen_rect)
            self.peepo2.update(self.screen_rect)
            self.peepo3.update(self.screen_rect)
            self.peepo4.update(self.screen_rect)
            self.render()
            self.clock.tick(self.fps)

            for obj in self.objects:
                if self.peepo0.rect.colliderect(obj.rect):
                    self.score0 += 0.1
                if self.peepo1.rect.colliderect(obj.rect):
                    self.score1 += 0.1
                if self.peepo2.rect.colliderect(obj.rect):
                    self.score2 += 0.1
                if self.peepo3.rect.colliderect(obj.rect):
                    self.score3 += 0.1
                if self.peepo4.rect.colliderect(obj.rect):
                    self.score4 += 0.1

            if self.peepo0.rect.x > 1450:
                self.peepo0.run = False
            if self.peepo1.rect.x > 1450:
                self.peepo1.run = False
            if self.peepo2.rect.x > 1450:
                self.peepo2.run = False
            if self.peepo3.rect.x > 1450:
                self.peepo3.run = False
            if self.peepo4.rect.x > 1450:
                self.peepo4.run = False

            loop += 1
            print(loop)
            if loop > 280:
                loop = 0
                self.new_peepo()


def main():
    """
    Prepare our environment, create a display, and start the program (pygame).

    Initialize the game screen with the actors: walls, obstacles and peepo
    """
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    wall1 = Wall('wall_up', (0, 0), (3200, 5))
    wall2 = Wall('wall_left', (0, 0), (5, 2000))
    wall3 = Wall('wall_right', (1598, 0), (5, 2000))
    wall4 = Wall('wall_down', (0, 998), (3200, 5))

    generate_obstacles()
    obstacles = []
    with open('obstacles.json') as json_data:
        for obs in json.load(json_data):
            obstacles.append(ObjectActor(obs['id'], (obs['x'], obs['y'])))
    obstacles.extend([wall1, wall2, wall3, wall4])

    world = PeeposWorld(obstacles)

    world.main_loop()

    pg.quit()
    sys.exit()


"""
####################################################################################
############################### BEGIN HERE #########################################
####################################################################################
"""
if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
