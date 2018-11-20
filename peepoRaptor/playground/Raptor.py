#15/11/2018
import math
import os
import random
import sys

import pygame as pg
import numpy as np
import scipy.optimize as opt
from  scipy.optimize import minimize
import matplotlib.pyplot as plt
from peepoRaptor.playground.models.Raptor_model import RaptorModel
from peepoRaptor.playground.models.Raptor_peepo import Raptor
from peepoRaptor.playground.models.Raptor_model import normalize_angle
vec = pg.math.Vector2

CAPTION = "Raptor 's World"
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

def normalize_angle(angle):
    #return angle
    if angle < 0:
        angle = 2 * math.pi + angle

    if angle >= 2 * math.pi:
        angle -= 2 * math.pi

    if angle <= -2 * math.pi:
        angle += 2 * math.pi

    return angle

class Analytic_Solution(object):

    def __init__(self,Apx,Apy,Ahx,Ahy,beta,vp,vh):
        self.Apx = Apx
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
        self.analytical_solution = np.zeros(2)


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

    def get_analytical_solution(self):
        res = minimize(self.func, (0, 50), method = 'Nelder-Mead')
        solution = np.zeros(2)
        solution[0] = res.x[0]
        solution[1] = res.x[1]
        self.alpha = solution[0]
        self.tm = solution[1]
        xy = solution
        self.analytical_solution[0] = 1*(self.Ahx + self.vh*math.cos(self.alpha)*self.tm)
        self.analytical_solution[1] = 1*(self.Ahy + self.vh*math.sin(self.alpha)*self.tm)
        #print("Analytical solution = ", self.analytical_solution)
        return self.analytical_solution

    def draw(self,screen):
        xy = self.analytical_solution
        #pg.draw.line(screen,GREY, [0,self.Ahy], [xy[0], xy[1]],2)
        #pg.draw.line(screen, GREY, [self.Apx, 0] , [self.Apx + self.vp*math.cos(self.beta)*self.tm, self.Apy + self.vp*math.sin(self.beta)*self.tm], 2)
        #pg.draw.circle(screen,BLUE, [int(xy[0]), int(xy[1])],9)

class PigeonActor(object):
    """ This class represents a Pigeon; the victim
        More than 1 Pigeon can be present but this is for later maybe"""

    SIZE = (40, 40)
    MAXSPEED = 5  # the speed will be different for each run

    def __init__(self, number_of_pigeons, wall):
        np.random.seed(9001)
        self.wall = wall
        self.speed = random.randint(1, PigeonActor.MAXSPEED)  # a random speed between 1 and MAXSPEED
        self.number_of_pigeons = number_of_pigeons
        self.tensor_of_pigeons = np.zeros(shape=(number_of_pigeons, 4))
        self.beta = 0
        self.Apx = 0
        self.max_speed = PigeonActor.MAXSPEED
        self.stop = False
        self.first_tensor()
        self.pos_x = self.tensor_of_pigeons[0][0]
        self.pos_y = self.tensor_of_pigeons[0][1]
        self.trajectory = []
        self.trajectory.append((int(self.pos_x ), int(self.pos_y )))


    def first_tensor(self):
        for row in range(0, self.number_of_pigeons):
            self.tensor_of_pigeons[row][2] = self.speed# the speed of the pigeons (uniform for all of them, for the moment being
            self.tensor_of_pigeons[row][3] = random.uniform(0.05 * math.pi, 0.99 * math.pi)
            # the Pigeon start at the upper side, somewhere in the second halve of the width
            self.tensor_of_pigeons[row][0] = random.uniform(WALL_SIZE[0] / 2, WALL_SIZE[0])
            self.tensor_of_pigeons[row][1] = 0*WALL_SIZE[1] +  0*random.uniform(0, WALL_SIZE[1])
            self.beta = self.tensor_of_pigeons[row][3]
            self.Apx = self.tensor_of_pigeons[row][0]
            self.pos_x = self.tensor_of_pigeons[row][0]
            self.pos_y = self.tensor_of_pigeons[row][1]

    def get_Apx(self):
        return self.Apx

    def get_beta(self):
        return self.beta

    def get_pigeons(self):
        return self.tensor_of_pigeons

    def get_pigeons_obstacles(self):
        obstacles = []
        for row in range(0, self.number_of_pigeons):
            obstacles.append(
                PigeonObject('target_' + str(row), (self.tensor_of_pigeons[row][0], self.tensor_of_pigeons[row][1])))
        return obstacles

    def update(self):

        for row in range(0, self.number_of_pigeons):
            self.tensor_of_pigeons[row][0] += self.tensor_of_pigeons[row][2] * math.cos(self.tensor_of_pigeons[row][3])
            self.tensor_of_pigeons[row][1] += self.tensor_of_pigeons[row][2] * math.sin(self.tensor_of_pigeons[row][3])
            self.pos_x = self.tensor_of_pigeons[row][0]
            self.pos_y = self.tensor_of_pigeons[row][1]
            if self.pos_x >= self.wall[2]:
                self.tensor_of_pigeons[row][3] = normalize_angle(math.pi / 2 + self.tensor_of_pigeons[row][3])
            if self.pos_y >= self.wall[3]:
                self.tensor_of_pigeons[row][3] = normalize_angle(math.pi / 2 + self.tensor_of_pigeons[row][3])
            if self.pos_x < self.wall[0]:
                self.tensor_of_pigeons[row][3] = normalize_angle(math.pi - self.tensor_of_pigeons[row][3])
            if self.pos_y < self.wall[1]:
                self.tensor_of_pigeons[row][3] = normalize_angle(math.pi / 2 + self.tensor_of_pigeons[row][3])
            self.trajectory.append((int(self.pos_x), int(self.pos_y)))
            #print("Pigeon position : ", self.pos_x, "/", self.pos_y)
            # once the Pigeon has reached safely a wall, he rests and stays there
            '''
            if self.tensor_of_pigeons[row][0] >= WALL_SIZE[0]:
                self.speed = 0
                self.tensor_of_pigeons[row][2] = self.speed
                self.stop = True
            if self.tensor_of_pigeons[row][1] >= WALL_SIZE[1]:
                self.speed = 0
                self.tensor_of_pigeons[row][2] = self.speed
                self.stop = True
            if self.tensor_of_pigeons[row][0] <= 0:
                self.speed = 0
                self.tensor_of_pigeons[row][2] = self.speed
                self.stop = True
            if self.tensor_of_pigeons[row][1] <= 0:
                self.speed = 0
                self.tensor_of_pigeons[row][2] = self.speed
                self.stop = True'''



class PigeonObject(object):
    SIZE = (20, 20)

    def __init__(self, id, pos):
        self.rect = pg.Rect((0, 0), PigeonObject.SIZE)
        self.rect.center = pos
        self.image = self.make_image()
        self.id = id

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        pg.draw.rect(image, pg.Color("black"), image_rect)
        pg.draw.rect(image, pg.Color("blue"), image_rect.inflate(-2, -2))
        return image

    def update(self):
        pass

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class RaptorActor(object):
    """ This class represents raptor """

    SIZE = (40, 40)
    SPEED = 10
    EYES_DISTANCE = 40
    NUMBER_OF_SECTORS = 7

    def __init__(self, pos,pigeon, target, wall):
        self.wall = wall
        self.pos_x = pos[0]
        self.pos_y = pos[1]
        self.rect = pg.Rect((0, 0), RaptorActor.SIZE)
        self.rect.center = pos
        self.pos_pigeon = pigeon.tensor_of_pigeons[0]
        self.eye_distance = RaptorActor.EYES_DISTANCE
        self.n_sectors = RaptorActor.NUMBER_OF_SECTORS
        self.angle =3*math.pi/4 +  0*random.uniform(- math.pi*0.75, math.pi*0.75)#
        self.speed = RaptorActor.SPEED
        self.trajectory = []
        self.trajectory.append((int(self.pos_x + RaptorActor.SIZE[0]/2), int(self.pos_y +  RaptorActor.SIZE[1]/2)))
        self.max_sector = 130/180*math.pi
        self.sector_L = np.zeros(self.n_sectors+1)
        self.sector_R = np.zeros(self.n_sectors+1)
        self.quadrants = np.zeros(self.n_sectors)
        self.sector = np.zeros(8)
        self.make_quadrants()
        self.update_sectors()
        self.model = RaptorModel(self, pigeon, wall)  # see Raptor_model.py
        self.image = self.make_image()
        self.image_original = self.image.copy()
        self.raptor = Raptor()#see Raptor_peepo.py
        self.keys = pg.key.get_pressed()
        self.alpha_L = []
        self.alpha_R = []
        self.d_L = []
        self.d_R = []
        self.dist = []
        self.left_eye = [0,0]
        self.right_eye = [0,0]
        self.pos_target = [0,0]

    def make_quadrants(self):
        gamma = self.max_sector/self.n_sectors
        for i in range(0,self.n_sectors):
            self.quadrants[i] = gamma
        norm = 0
        for i in range(0,self.n_sectors):
            norm += self.quadrants[i]
        for i in range(0,self.n_sectors):
            self.quadrants[i] *= self.max_sector/norm
        #print("Quadrants : ", self.quadrants*180/math.pi, " degrees. Giving a sum of : ", np.sum(self.quadrants)*180/math.pi)

    def update_sectors(self):
        self.angle = normalize_angle(self.angle)
        #print("self.angle in update sector = ", self.angle*180/math.pi," degrees")
        #left sector
        self.sector_L[0] = self.angle - self.max_sector/2
        for alfa in range(1, len(self.sector_L)):
            self.sector_L[alfa] = self.sector_L[alfa-1] + self.quadrants[alfa-1]
        for alfa in range(0, len(self.sector_L)):
            self.sector_L[alfa] = normalize_angle(self.sector_L[alfa])
        # right sector
        self.sector_R[self.n_sectors-1] = self.angle - self.max_sector/2
        for beta in range(1, len(self.sector_R)):
            alfa = self.n_sectors - 1 - beta
            self.sector_R[alfa] = self.sector_R[alfa - 1] + self.quadrants[alfa - 1]
        for alfa in range(0, len(self.sector_R)):
            self.sector_R[alfa] = normalize_angle(self.sector_R[alfa])
        self.angle = normalize_angle(self.angle)
        # print("self.angle in update sector = ", self.angle*180/math.pi," degrees")
        self.sector[0] = self.angle - self.quadrants[3] / 2 - self.quadrants[2] - self.quadrants[1] - self.quadrants[0]
        for alfa in range(1, len(self.sector)):
            self.sector[alfa] = self.sector[alfa - 1] + self.quadrants[alfa - 1]
        for alfa in range(0, len(self.sector)):
            self.sector[alfa] = normalize_angle(self.sector[alfa])

    def render_traject(self, screen, pigeon ,state):
        for p in range(0, len(self.trajectory)):
            pg.draw.circle(screen,GREEN, [int(self.trajectory[p][0]), int(self.trajectory[p][1])], 2)
        for p in range(0, len(pigeon.trajectory)):
            pg.draw.circle(screen,BLUE, [int(pigeon.trajectory[p][0]), int(pigeon.trajectory[p][1])], 2)
        pg.display.update()
        '''count = 0
        if not state:
            while count == 0:
                for event in pg.event.get():
                    if event.type == pg.QUIT or self.keys[pg.K_ESCAPE]:
                        self.done = True
                    elif event.type in (pg.KEYUP, pg.KEYDOWN):
                        self.keys = pg.key.get_pressed()
                count = 0'''

    def update(self, pos_pigeon, screen_rect):
        self.model.process()
        #self.angle = normalize_angle(self.angle)
        self.pos_x += self.speed * math.cos(self.angle)
        self.pos_y += self.speed * math.sin(self.angle)
        self.rect.center = [self.pos_x, self.pos_y]
        if self.pos_x >= self.wall[2] :
            self.angle = normalize_angle(math.pi/2 + self.angle)
        if self.pos_y >= self.wall[3] :
            self.angle = normalize_angle(math.pi/2 + self.angle)
        if self.pos_x < self.wall[0] :
            self.angle = normalize_angle(math.pi - self.angle)
        if  self.pos_y <self.wall[1]:
            self.angle = normalize_angle(math.pi/2 + self.angle)
        self.trajectory.append(( int(self.pos_x), int(self.pos_y)))
        self.image = pg.transform.rotate(self.image_original, -self.angle*180/math.pi)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.clamp_ip(screen_rect)
        #self.update_sectors()
        X_L = [self.pos_x - self.eye_distance/2*math.sin(self.angle), self.pos_y + self.eye_distance/2*math.cos(self.angle)]
        X_R = [self.pos_x + self.eye_distance/2*math.sin(self.angle), self.pos_y - self.eye_distance/2*math.cos(self.angle)]
        tg_alfa_L = (self.pos_pigeon[1] - X_L[1])/(self.pos_pigeon[0] - X_L[0])
        tg_alfa_R = (self.pos_pigeon[1] - X_R[1])/(self.pos_pigeon[0] - X_R[0])
        dis_L = math.sqrt((self.pos_pigeon[1] - X_L[1])*(self.pos_pigeon[1] - X_L[1]) + (self.pos_pigeon[0] - X_L[0])*(self.pos_pigeon[0] - X_L[0]))
        dis_R = math.sqrt((self.pos_pigeon[1] - X_R[1])*(self.pos_pigeon[1] - X_R[1]) + (self.pos_pigeon[0] - X_R[0]) * (self.pos_pigeon[0] - X_R[0]))
        self.alpha_L.append(math.atan(tg_alfa_L))
        self.alpha_R.append(math.atan(tg_alfa_R))
        self.d_L.append(dis_L)
        self.d_R.append(dis_R)
        self.dist.append((dis_L+dis_R)/2)
        self.left_eye[0]  = self.pos_x - self.eye_distance/2*math.sin(self.angle)
        self.left_eye[1]  = self.pos_y + self.eye_distance/2*math.cos(self.angle)
        self.right_eye[0] = self.pos_x + self.eye_distance/2*math.sin(self.angle)
        self.right_eye[1] = self.pos_y - self.eye_distance/2*math.cos(self.angle)
        self.pos_target[0] = self.pos_pigeon[0]
        self.pos_target[1] = self.pos_pigeon[1]
        #self.rect.clamp_ip(screen_rect)
        self.update_sectors()
        #self.raptor.update(self.model)

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        pg.draw.circle(surface, BLACK, [int(self.pos_x), int(self.pos_y)], int(RaptorActor.SIZE[0]/2), 2)
        pg.draw.line(surface, pg.Color("green"), self.left_eye, self.pos_target, 2)
        pg.draw.line(surface, pg.Color("red"), self.right_eye, self.pos_target, 2)
        #pg.draw.circle(surface, GREY, [int(self.pos_x), int(self.pos_y)], 20)
        #pg.draw.line(surface, pg.Color("blue"), self.rect.center, self.edge_direction, 1)'''

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        #pg.draw.rect(image, pg.Color("black"), image_rect)
        #pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        #pg.draw.circle(image, GREY, [int(self.pos_x), int(self.pos_y)], 20, 3)
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


class RaptorsWorld(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """

    def __init__(self, raptor_actor,  target, pigeons, analytical):
        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.fps = 60
        self.done = False
        self.keys = pg.key.get_pressed()
        self.raptor_actor = raptor_actor
        self.pigeons = pigeons
        self.target = target
        self.target_pos = []
        self.analytical = analytical
        #print(self.pigeons.tensor_of_pigeons)
        #print([self.raptor_actor.pos_x, self.raptor_actor.pos_y] )
        #aim_at = self.pigeons.tensor_of_pigeons[0]
        #aim_at = self.analytical.analytical_solution
        #self.raptor_actor.angle = 1*RaptorsWorld.calculate_angle(self, aim_at, self.raptor_actor.pos_x, self.raptor_actor.pos_y) + 0*random.uniform(-math.pi/5, math.pi/5)

    def render(self):
        """
        Perform all necessary drawing and update the screen.
        """
        self.screen.fill(pg.Color("white"))
        #self.analytical.draw(self.screen)
        for obj in self.target:
            obj.draw(self.screen)
        self.raptor_actor.draw(self.screen)
        #if self.pigeons.stop:
            #self.raptor_actor.render_traject(self.screen, False)
            #self.pigeon_actor.render_traject(self.screen, False)
        pg.display.update()

    def calculate_angle( self,target, raptor_x, raptor_y):
        #print(" target :::: " ,target)
        #print(raptor_x)
        tg_alfa = (target[1] - raptor_y)/(target[0] - raptor_x)
        return normalize_angle(math.atan(tg_alfa))

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


    def main_loop(self):
        """
        Game loop
        """
        while not self.done:
            self.event_loop()
            self.pigeons.update()
            self.target = self.pigeons.get_pigeons_obstacles()
            self.target_pos = self.pigeons.tensor_of_pigeons[0]
            self.raptor_actor.update(self.pigeons.tensor_of_pigeons,self.screen_rect)
            #print("Target position: ",self.pigeons.tensor_of_pigeons )
            #print("Raptor position:", self.raptor_actor.pos_x,self.raptor_actor.pos_y )
            xa = self.target_pos[0]
            ya = self.target_pos[1]
            xb = self.raptor_actor.pos_x
            yb = self.raptor_actor.pos_y
            d = math.sqrt((ya -yb)*(ya - yb) + (xa - xb)*(xa - xb))
            self.raptor_actor.dist.append(d)
            delta = RaptorActor.SIZE[0]/2
            print("Distance :", d)
            if d <= delta:
                print("Raptor captured pigeon")
                self.raptor_actor.render_traject(self.screen,self.pigeons, False)
                #self.pigeon_actor.render_traject(self.screen,self.pigeon_actor False)
                self.done = True
                break
            if self.pigeons.stop:
                print("Pigeon reached bounds")
                self.raptor_actor.render_traject(self.screen, self.pigeons,False)
                #self.pigeon_actor.render_traject(self.screen, False)
                self.done = True
                break
            self.render()
            self.clock.tick(self.fps)
        print("out of main loop")

class Draw_Graphics(object):

    def __init__(self, raptor):
        self.alfa_L = raptor.alpha_L
        self.alfa_R = raptor.alpha_R
        self.d_L    = raptor.d_L
        self.d_R    = raptor.d_R
        self.collision = raptor.dist
        self.draw_graphics()

    def draw_graphics(self):
        print("Draw graphics called")
        t = []
        tt = []
        delta_alfa = []
        delta_d =  []
        for i in range(0,len(self.alfa_L)):
            t.append(i)
            self.alfa_L[i] =  180/math.pi*normalize_angle(self.alfa_L[i])
            self.alfa_R[i] = 180/math.pi*normalize_angle(self.alfa_R[i])
            delta_alfa.append((self.alfa_L[i] -self.alfa_R[i]))
            delta_d.append((self.d_L[i] -self.d_R[i]))
            #print(" alfa_L = ",self.alfa_L[i]," alfa_R = ",self.alfa_R[i]," d_L = ",self.d_L[i]," d_RL = ",self.d_R[i])
        #t = np.arange(0.0, len(self.alfa_L),5)
        for i in range(0, len(self.collision)):
            tt.append(i)

        s1 = self.alfa_L
        s2 = self.alfa_R
        s3 = self.collision
        s4 = self.d_R


        fig, axs = plt.subplots(3, 1, sharex=True)
        # Remove horizontal space between axes
        #fig.subplots_adjust(hspace=0)

        # Plot each graph, and manually set the y tick values
        axs[0].plot(t, s1)
        #axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
        #axs[0].set_ylim(-1, 1)

        axs[1].plot(t, s2)
        axs[2].plot(tt, s3)
        #axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
        #axs[1].set_ylim(0, 1)
        '''
        axs[2].plot(t, s3)
        #axs[2].set_yticks(np.arange(-0.9, 1.0, 0.4))
        #axs[2].set_ylim(-1, 1)

        axs[3].plot(t, s4)
        #axs[2].set_yticks(np.arange(-0.9, 1.0, 0.4))
        #axs[2].set_ylim(-1, 1'''

        plt.show()


def main():
    """
    Prepare our environment, create a display, and start the program (pygame).

    Initialize the game screen with the actors: walls, obstacles and raptor
    """
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pg.init()
    pg.display.set_caption(CAPTION)
    pg.display.set_mode(SCREEN_SIZE)
    wall1 = Wall('wall_up', (0, 0), (WALL_SIZE[0], 5))
    wall2 = Wall('wall_left', (0, 0), (5, WALL_SIZE[1]))
    wall3 = Wall('wall_right', (WALL_SIZE[0], 0), (5, WALL_SIZE[1]))
    wall4 = Wall('wall_down', (0, WALL_SIZE[1]), (WALL_SIZE[0], 5))
    wall = [0, 0, WALL_SIZE[0], WALL_SIZE[1]]
    pigeons = PigeonActor(1,wall)  # class adress for the pigeons
    target = pigeons.get_pigeons_obstacles()
    raptor_actor = RaptorActor((0, WALL_SIZE[1] / 2), pigeons, target, wall)
    Apx = pigeons.Apx
    beta = pigeons.beta
    vp = pigeons.max_speed
    vh = raptor_actor.speed
    analytic_solution = Analytic_Solution(Apx,0,0,WALL_SIZE[1]/2,beta,vp,vh)
    xy = analytic_solution.get_analytical_solution()
    world = RaptorsWorld(raptor_actor, target, pigeons,analytic_solution )
    world.main_loop()
    Draw_Graphics(raptor_actor)
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
