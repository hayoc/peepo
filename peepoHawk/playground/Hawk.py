#3/11/2018
import math
import os
import random
import sys

import pygame as pg
import numpy as np
import scipy.optimize as opt
from  scipy.optimize import minimize
from peepoHawk.playground.models.Hawk_model import PeepoModel
from peepoHawk.playground.models.Hawk_peepo import Peepo

from peepoHawk.playground.util.vision import end_line
from peepoHawk.playground.Performance.performance import  Metrics  # NOT used for the moment: intended to measure te effectiveness of the leanning rate

vec = pg.math.Vector2

CAPTION = "Peepo 's World"
SCREEN_SIZE = (1400, 750)
WALL_SIZE = (SCREEN_SIZE[0], SCREEN_SIZE[1])
SCREEN_CENTER = (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2)
GAMMA = SCREEN_SIZE[1]/SCREEN_SIZE[0]
TRANSPARENT = (0, 0, 0, 0)
DIRECT_DICT = {pg.K_LEFT: (-1, 0),
               pg.K_RIGHT: (1, 0),
               pg.K_UP: (0, -1),
               pg.K_DOWN: (0, 1)}

# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED   =   (255,   0,   0)
GREY =  (225,220,225)

class Analytic_Solution(object):

    def __init__(self,Apx,Apy,Ahx,Ahy,beta,vp,vh):
        self.Apx = Apx
        print("Apx = ", self.Apx)
        self.Apy = Apy
        self.Ahx = Ahx
        self.Ahy = Ahy
        self.beta = beta
        self.vp = vp
        self.vpx = self.vp*math.cos(beta)
        self.vpy = self.vp*math.sin(beta)
        self.vh = vh
        self.alpha = 0
        self.tm = 0


    def func(self,variables):
        alpha = variables[0]
        tm = variables [1]
        f = np.zeros(2)
        f[0] = self.Apx + self.vpx*tm - self.Ahx - self.vh*math.cos(alpha)*tm
        f[1] = self.Apy + self.vpy*tm - self.Ahy - self.vh*math.sin(alpha)*tm
        return np.dot(f,f)

    def constraint(self,variables):
        alpha = variables[0]
        tm = variables [1]
        f = np.zeros(2)
        xtp = self.Apx + self.vp*math.cos(self.beta)*tm
        ytp = self.Apy + self.vp*math.sin(self.beta)*tm
        xth = self.Ahx + self.vh*math.cos(alpha)*tm
        yth = self.Ahy + self.vh*math.sin(alpha)*tm
        f[0] = 0.000001 - abs(xtp-xth)
        f[1] = 0.000001 - abs(ytp-yth)
        return f

    @property
    def get_analytical_solution(self):
        #solution = opt.fsolve(self.func, (100, 100))
        #cons = {'type': 'ineq', 'fun': self.constraint}
        #solution = opt.fsolve(self.func, (0, 100), constraints = cons)
        res = minimize(self.func, (0, 50), method = 'Nelder-Mead')#, constraints = cons)
        solution = np.zeros(2)
        solution[0] = res.x[0]
        solution[1] = res.x[1]
        self.alpha = solution[0]
        self.tm = solution[1]
        xy = solution
        xy[0] = self.Ahx + self.vh*math.cos(self.alpha)*self.tm
        xy[1] = self.Ahy + self.vh*math.sin(self.alpha)*self.tm
        return xy

    def draw(self,screen):
        xy = self.get_analytical_solution
        pg.draw.line(screen,GREEN, [0,self.Ahy], [xy[0], xy[1]],2)
        pg.draw.line(screen, RED, [self.Apx, 0] , [self.Apx + self.vp*math.cos(self.beta)*self.tm, self.Apy + self.vp*math.sin(self.beta)*self.tm], 2)
        pg.draw.circle(screen,BLUE, [int(xy[0]), int(xy[1])],9)

class PoopieActor(object):
    """ This class represents a Poopie; the victim
        More than 1 Poopie can be present but this is for later maybe"""

    SIZE = (40, 40)
    MAXSPEED = 2  # the speed will be different for each run

    def __init__(self, number_of_poopies):
        self.speed = random.randint(1, PoopieActor.MAXSPEED)  # a random speed between 1 and MAXSPEED
        self.number_of_poopies = number_of_poopies
        self.tensor_of_poopies = np.zeros(shape=(number_of_poopies, 4))
        self.beta = 0
        self.Apx = 0
        self.first_tensor()
        self.max_speed = PoopieActor.MAXSPEED
        self.stop = False


    def first_tensor(self):
        for row in range(0, self.number_of_poopies):
            self.tensor_of_poopies[row][2] = self.speed# the speed of the poopies (uniform for all of them, for the moment being
            self.tensor_of_poopies[row][3] = random.uniform(0.05 * math.pi, 0.99 * math.pi)
            # the Poopie start at the upper side, somewhere in the second halve of the width
            self.tensor_of_poopies[row][0] = random.uniform(WALL_SIZE[0] / 2, WALL_SIZE[0])
            self.tensor_of_poopies[row][1] = 0  # random.uniform(0, WALL_SIZE[1])
            self.beta = self.tensor_of_poopies[row][3]
            self.Apx = self.tensor_of_poopies[row][0]

    def get_Apx(self):
        return self.Apx

    def get_beta(self):
        return self.beta

    def get_poopies(self):
        return self.tensor_of_poopies

    def get_poopies_obstacles(self):
        obstacles = []
        for row in range(0, self.number_of_poopies):
            obstacles.append(
                PoopieObject('obj_' + str(row), (self.tensor_of_poopies[row][0], self.tensor_of_poopies[row][1])))
        return obstacles

    def update(self):

        for row in range(0, self.number_of_poopies):
            self.tensor_of_poopies[row][0] += self.tensor_of_poopies[row][2] * math.cos(self.tensor_of_poopies[row][3])
            self.tensor_of_poopies[row][1] += self.tensor_of_poopies[row][2] * math.sin(self.tensor_of_poopies[row][3])
            # once the Poopie has reached safely a wall, he rests and stays there
            if self.tensor_of_poopies[row][0] >= WALL_SIZE[0]:
                self.speed = 0
                self.tensor_of_poopies[row][2] = self.speed
                self.stop = True
            if self.tensor_of_poopies[row][1] >= WALL_SIZE[1]:
                self.speed = 0
                self.tensor_of_poopies[row][2] = self.speed
                self.stop = True
            if self.tensor_of_poopies[row][0] <= 0:
                self.speed = 0
                self.tensor_of_poopies[row][2] = self.speed
                self.stop = True
            if self.tensor_of_poopies[row][1] <= 0:
                self.speed = 0
                self.tensor_of_poopies[row][2] = self.speed
                self.stop = True



class PoopieObject(object):
    SIZE = (20, 20)

    def __init__(self, id, pos):
        self.rect = pg.Rect((0, 0), PoopieObject.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.id = id

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("red"), image_rect.inflate(-2, -2))
        return image

    def update(self):
        pass

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class PeepoActor(object):
    """ This class represents peepo """

    SIZE = (40, 40)
    SPEED = 10

    def __init__(self, pos, target):
        self.model = PeepoModel(self, target)
        self.rect = pg.Rect((0, 0), PeepoActor.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.image_original = self.image.copy()
        self.peepo = Peepo()
        self.rotation = 0
        self.edge_right = end_line(PeepoModel.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(PeepoModel.RADIUS, self.rotation - 30, self.rect.center)
        self.speed = PeepoActor.SPEED
        self.trajectory = []
        self.trajectory.append((int(self.rect.x + PeepoActor.SIZE[0]/2), int(self.rect.y +  PeepoActor.SIZE[1]/2)))

    def render_traject(self, screen, state):
        for p in range(0, len(self.trajectory)):
            #print([int(self.trajectory[p][0]), int(self.trajectory[p][1])])
            pg.draw.circle(screen,GREY, [int(self.trajectory[p][0]), int(self.trajectory[p][1])], 2)
        pg.display.update()
        count = 0
        if not state:
            while count == 0:
                for event in pg.event.get():
                    if event.type == pg.QUIT or self.keys[pg.K_ESCAPE]:
                        self.done = True
                    elif event.type in (pg.KEYUP, pg.KEYDOWN):
                        self.keys = pg.key.get_pressed()
                count = 0

    def update(self, screen_rect):
        self.model.process()
        self.rotation = -self.rotation
        self.rect.x += PeepoActor.SPEED * math.cos(math.radians(self.rotation))
        self.rect.y += PeepoActor.SPEED * math.sin(math.radians(self.rotation))
        if self.rect.x >= SCREEN_SIZE[0] or self.rect.y >= SCREEN_SIZE[0] or self.rect.x <= 0 or self.rect.y<= 0:
            self.rotation = self.rotation + 180
        #print(self.rect.x, "/", self.rect.y)
        self.trajectory.append(( int(self.rect.x + PeepoActor.SIZE[0]/2), int(self.rect.y +  PeepoActor.SIZE[1]/2)))
        if self.model.motor_output[pg.K_LEFT]:
            self.rotation -= random.randint(10, 30)
            if self.rotation < 0:
                self.rotation = 360
        if self.model.motor_output[pg.K_RIGHT]:
            self.rotation += random.randint(10, 30)
            if self.rotation > 360:
                self.rotation = 0

        self.image = pg.transform.rotate(self.image_original, -self.rotation)
        self.rect = self.image.get_rect(center=self.rect.center)

        self.edge_right = end_line(PeepoModel.RADIUS, self.rotation + 30, self.rect.center)
        self.edge_left = end_line(PeepoModel.RADIUS, self.rotation - 30, self.rect.center)

        self.rect.clamp_ip(screen_rect)
        self.peepo.update(self.model)

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


class PeeposWorld(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """

    def __init__(self, peepo, objects, poopies, metrics, analytical):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.keys = pg.key.get_pressed()
        self.peepo = peepo
        self.objects = objects
        self.poopies = poopies
        self.metrics = metrics
        self.analytical = analytical

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
        self.analytical.draw(self.screen)
        for obj in self.objects:
            obj.draw(self.screen)
        self.peepo.draw(self.screen)
        if self.poopies.stop:
            self.peepo.render_traject(self.screen, False)
        pg.display.update()

    def main_loop(self):
        """
        Game loop
        """
        while not self.done:
            self.event_loop()
            self.peepo.update(self.screen_rect)
            self.poopies.update()
            wall1 = Wall('wall_up', (0, 0), (WALL_SIZE[0] * 2, 5))
            wall2 = Wall('wall_left', (0, 0), (5, WALL_SIZE[1] * 2))
            wall3 = Wall('wall_right', (WALL_SIZE[0], 0), (5, WALL_SIZE[1] * 2))
            wall4 = Wall('wall_down', (0, WALL_SIZE[1]), (WALL_SIZE[0] * 2, 5))
            obstacles = self.poopies.get_poopies_obstacles()
            obstacles.extend([wall1, wall2, wall3, wall4])
            self.objects = obstacles
            self.render()
            self.clock.tick(self.fps)


def main():
    """
    Prepare our environment, create a display, and start the program (pygame).

    Initialize the game screen with the actors: walls, obstacles and peepo
    """
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)

    wall1 = Wall('wall_up', (0, 0), (WALL_SIZE[0], 5))
    wall2 = Wall('wall_left', (0, 0), (5, WALL_SIZE[1]))
    wall3 = Wall('wall_right', (WALL_SIZE[0], 0), (5, WALL_SIZE[1]))
    wall4 = Wall('wall_down', (0, WALL_SIZE[1]), (WALL_SIZE[0], 5))
    Max_Epochs = 1
    Epoch = 0
    metrics = Metrics(Epoch, Max_Epochs)  # not used for the moment, intented to run Max_Epochs runs to assess statistically the effectiveness of the model
    poopies = PoopieActor(1)  # class adress for the poopies
    obstacles = poopies.get_poopies_obstacles()
    obstacles.extend([wall1, wall2, wall3, wall4])
    peepo = PeepoActor((0, WALL_SIZE[1] / 2), obstacles)
    Apx = poopies.Apx
    beta = poopies.beta
    print("Apx = ", Apx, " and beta = ", beta)
    vp = poopies.max_speed
    vh = peepo.speed
    analytic_solution = Analytic_Solution(Apx,0,0,WALL_SIZE[1]/2,beta,vp,vh)
    world = PeeposWorld(peepo, obstacles, poopies, metrics,analytic_solution )
    world.main_loop()
    pg.quit()
    sys.exit()


"""
####################################################################################
############################### BEGIN HERE #########################################
####################################################################################
"""
if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)
    main()
