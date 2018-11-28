#28/11/2018
import math
import os
import random
import sys

import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from peepoHawk.playground.models.Raptor_model import RaptorModel
from peepoHawk.playground.Pigeon import PigeonActor
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



class RaptorActor(object):
    """ This class represents raptor """

    SIZE = (40, 40)
    RESOLUTIONSLOPE = 45#dergrees
    SPEED = 10#7#10
    EYES_DISTANCE =40
    ANGLE_INCREMENT = 10# degrees
    NUMBER_OF_SECTORS = 16#15#71#7#125 #!!! TO GET THE SYSTEM STABLE THE RESOLUTION OF THE SECTORS SHOULD BE GREATER THEN THE ANGLE_INCREMENT
    MAXALFASECTOR = 130#degrees
    ALFAINNERSECTOR = 90#85#degrees


    def __init__(self, pos,pigeon, target, wall):
        self.alfa_increment = math.pi / 180 * RaptorActor.ANGLE_INCREMENT
        self.alfa_inner_sector = math.pi/180*RaptorActor.ALFAINNERSECTOR
        self.max_sector = RaptorActor.MAXALFASECTOR / 180 * math.pi  # 130° is a normal sight for a falcon
        #self.n_sectors = RaptorActor.Set_Number_of_Sectors(self)
        self.n_sectors =  RaptorActor.NUMBER_OF_SECTORS
        self.wall = wall
        self.pos_x = pos[0]
        self.pos_y = pos[1]
        self.rect = pg.Rect((0, 0), RaptorActor.SIZE)
        self.rect.center = pos
        self.pos_pigeon = pigeon.tensor_of_pigeons[0]
        self.eye_distance = RaptorActor.EYES_DISTANCE
        self.angle =  np.exp(1*1j*math.pi/4 +  0*1j*random.uniform(- math.pi*0.75, math.pi*0.75) + 0*1j)#
        self.speed = RaptorActor.SPEED
        self.trajectory = []
        self.trajectory.append((int(self.pos_x + RaptorActor.SIZE[0]/2), int(self.pos_y +  RaptorActor.SIZE[1]/2)))
        self.sector_L = np.full(self.n_sectors+1, 0+0*1j)
        self.sector_R = np.full(self.n_sectors+1, 0+0*1j)
        self.zero_sector_L = np.full(self.n_sectors+1, 0+0*1j)
        self.zero_sector_R = np.full(self.n_sectors+1, 0+0*1j)
        self.make_quadrants()
        self.update_sectors()
        self.image = self.make_image()
        self.image_original = self.image.copy()
        self.keys = pg.key.get_pressed()
        self.alpha_L = []
        self.alpha_R = []
        self.d_L = []
        self.d_R = []
        self.dist = []
        self.left_eye = [0,0]
        self.right_eye = [0,0]
        self.pos_target = [0,0]
        self.choosen_sector_L = 0
        self.choosen_sector_R = 0
        self.model = RaptorModel(self, pigeon, wall)  # see Raptor_model.py


    def Set_Number_of_Sectors(obj):
        wide = (obj.max_sector/2 + (math.pi/180*obj.alfa_inner_sector) )
        resolution = 2*obj.alfa_increment
        n_sectors = int(wide/resolution)
        print("Number of sectors : " , n_sectors, " with a resolution of ",  180/math.pi*wide/n_sectors, ' to compare with an increment of ',180/math.pi*obj.alfa_increment, " degrees" )
        return n_sectors

    def Check_Walls(obj):
        if obj.pos_x >= obj.wall[2] or obj.pos_x <= obj.wall[0] :
            if abs(np.angle(obj.angle)) == 0:
                obj.angle *= np.exp(1j*math.pi/20)
            obj.angle = -obj.angle.real + 1j*obj.angle.imag
        if obj.pos_y >= obj.wall[3] or obj.pos_y <= obj.wall[1] :
            if abs(np.angle(obj.angle)) == math.pi/2:
                obj.angle *= np.exp(1j*math.pi/20)
            obj.angle = obj.angle.real - 1j*obj.angle.imag

    def make_quadrants(self):
        wide = (self.max_sector/2 + (math.pi/180*RaptorActor.ALFAINNERSECTOR) )
        gamma = wide/(self.n_sectors)
        for alfa in range(0, len(self.zero_sector_L)):
            self.zero_sector_L[alfa] = np.exp(-1j*self.max_sector/2)*np.exp(1j*alfa*gamma)
        # right sector is the anti-symmetric to the left
        for alfa in range(0, len(self.zero_sector_R)):
            self.zero_sector_R[alfa] =  np.exp(1j*(self.max_sector/2 - wide ))*np.exp(1j*alfa*gamma)

        ##print("Left zero sector  : ", 180/math.pi*np.angle(self.zero_sector_L))
        ##print("Right zero sector : ", 180/math.pi*np.angle(self.zero_sector_R))

    def update_sectors(self):
        #print("self.angle in update sector = ", self.angle*180/math.pi," degrees")
        #left sector
        for alfa in range(0,len(self.sector_L)):
            self.sector_L[alfa] = self.zero_sector_L[alfa]*self.angle
            self.sector_R[alfa] = self.zero_sector_R[alfa]*self.angle

        #print("Left  sector  : ", 180 / math.pi * np.angle(self.sector_L))
        #print("Right sector : ", 180 / math.pi * np.angle(self.sector_R))

    def render_traject(self, screen, pigeon ,state):
        #print("RENDER TRAJECT CALLED")
        for p in range(0, len(self.trajectory)):
            pg.draw.circle(screen,BLACK, [int(self.trajectory[p][0]), int(self.trajectory[p][1])], 2)
        for p in range(0, len(pigeon.trajectory)):
            pg.draw.circle(screen,BLUE, [int(pigeon.trajectory[p][0]), int(pigeon.trajectory[p][1])], 2)
        pg.display.update()


    def update(self, pos_pigeon, screen_rect):
        #self.angle = np.exp(1j * 0)
        print("----------------------------------BEGIN  ITERATION ----------------------------------------------")
        #print(" ANGEL VECTOR : ", self.angle, " for an angle of : ", 180/math.pi*np.angle(self.angle))
        self.pos_x += self.speed * self.angle.real
        self.pos_y += self.speed * self.angle.imag
        self.update_sectors()
        self.rect.center = [self.pos_x, self.pos_y]
        #Check wether Raptor hits the wall
        RaptorActor.Check_Walls(self)
        self.trajectory.append(( int(self.pos_x), int(self.pos_y)))
        self.image = pg.transform.rotate(self.image_original, - np.angle(self.angle)*180/math.pi)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.clamp_ip(screen_rect)
        #self.update_sectors()
        self.left_eye[0]  = self.pos_x + self.eye_distance/2*self.angle.imag
        self.left_eye[1]  = self.pos_y - self.eye_distance/2*self.angle.real
        self.right_eye[0] = self.pos_x - self.eye_distance/2*self.angle.imag
        self.right_eye[1] = self.pos_y + self.eye_distance/2*self.angle.real
        X_L = [self.left_eye[0], self.left_eye[1]]
        X_R = [self.right_eye[0], self.right_eye[1]]
        '''print("Angle : " , 180/math.pi*np.angle(self.angle))
        print("Eye distance : ",self.eye_distance )
        print(" **************************************** Distance of eyes :", math.sqrt((self.right_eye[1] - self.left_eye[1])**2 + (self.right_eye[0] - self.left_eye[0])**2))'''
        tg_alfa_L = (self.pos_pigeon[1] - X_L[1])/(self.pos_pigeon[0] - X_L[0])
        tg_alfa_R = (self.pos_pigeon[1] - X_R[1])/(self.pos_pigeon[0] - X_R[0])
        #print("EXACT ANGLE CALCULATION")
        #print("From Left eye : ", 180/math.pi*math.atan(tg_alfa_L), "       ", "From Right eye : ", 180/math.pi*math.atan(tg_alfa_R))
        dis_L = math.sqrt((self.pos_pigeon[1] - X_L[1])*(self.pos_pigeon[1] - X_L[1]) + (self.pos_pigeon[0] - X_L[0])*(self.pos_pigeon[0] - X_L[0]))
        dis_R = math.sqrt((self.pos_pigeon[1] - X_R[1])*(self.pos_pigeon[1] - X_R[1]) + (self.pos_pigeon[0] - X_R[0]) * (self.pos_pigeon[0] - X_R[0]))
        self.alpha_L.append(math.atan(tg_alfa_L))
        self.alpha_R.append(math.atan(tg_alfa_R))
        self.alpha_L.append(self.choosen_sector_L)
        self.alpha_R.append(self.choosen_sector_R)
        self.d_L.append(dis_L)
        self.d_R.append(dis_R)
        self.dist.append((dis_L+dis_R)/2)
        self.model.process()
        print("-----------------------------------------------------------------------------------------------")

    def draw(self, surface):
        surface.blit(self.image, self.rect)
        pg.draw.circle(surface, BLACK, [int(self.pos_x), int(self.pos_y)], int(RaptorActor.SIZE[0]/2), 2)
        #if self.angle + self.sector_L[self.n_sectors-1] < math.atan((self.pos_target[1] - self.left_eye[1])/(self.pos_target[0] - self.left_eye[0])) and self.angle + self.sector_R[0] < math.atan((self.pos_target[1] - self.right_eye[1])/(self.pos_target[0] - self.right_eye[0])):
        #pg.draw.line(surface, pg.Color("green"), self.left_eye, self.pos_target, 2)
        #pg.draw.line(surface, pg.Color("red"), self.right_eye, self.pos_target, 2)
        pg.draw.line(surface, pg.Color("BLUE"), [self.pos_x,self.pos_y],  [self.pos_x + 25 * self.angle.real,  self.pos_y + 25 *self.angle.imag], 2)
        #pg.draw.line(surface, pg.Color("BLUE"), self.left_eye, [self.left_eye[0] + 100 * math.cos(self.angle), self.left_eye[1] + 100 * math.sin(self.angle)], 2)
        #pg.draw.line(surface, pg.Color("BLUE"), self.right_eye, [self.right_eye[0] + 100 * math.cos(self.angle), self.right_eye[1] + 100 * math.sin(self.angle)], 2)
        pg.draw.line(surface, pg.Color("GREEN"), self.left_eye,[self.left_eye[0] + 700*self.sector_L[self.choosen_sector_L].real, self.left_eye[1] + 700*self.sector_L[self.choosen_sector_L].imag], 2)
        pg.draw.line(surface, pg.Color("RED")  , self.right_eye,[self.right_eye[0] + 700*self.sector_R[self.choosen_sector_R].real, self.right_eye[1] + 700*self.sector_R[self.choosen_sector_R].imag], 2)
        #pg.draw.line(surface, pg.Color("GREY"), self.left_eye,[self.left_eye[0] + 50*self.sector_L[self.n_sectors].real,self.left_eye[1] + 50*self.sector_L[self.n_sectors].imag], 2)
        pg.draw.line(surface, pg.Color("GREY"), self.left_eye,[self.left_eye[0] + 60*self.sector_L[0].real,self.left_eye[1] + 60* self.sector_L[0].imag], 2)
        #pg.draw.line(surface, pg.Color("GREY"), self.right_eye,[self.right_eye[0] + 50*self.sector_R[0].real, self.right_eye[1] + 50*self.sector_R[0].imag], 2)
        pg.draw.line(surface, pg.Color("GREY"), self.right_eye,[self.right_eye[0] + 60*self.sector_R[self.n_sectors].real, self.right_eye[1] + 60*self.sector_R[self.n_sectors].imag], 2)
        #pg.draw.circle(surface, GREY, [int(self.pos_x), int(self.pos_y)], 20)
        #pg.draw.line(surface, pg.Color("blue"), self.rect.center, self.edge_direction, 1)'''
        myfont = pg.font.SysFont('Arial', 30)
        text1 = myfont.render('Sector', False, (0, 0, 0))
        text2 = myfont.render("[" + str(self.choosen_sector_L) +", " + str(self.choosen_sector_R)+ "]", False, (0, 0, 0))
        surface.blit(text1, (0, 0))
        surface.blit(text2, (0, 40))

    def make_image(self):
        image = pg.Surface(self.rect.size).convert_alpha()
        image.fill(TRANSPARENT)
        image_rect = image.get_rect()
        #pg.draw.rect(image, pg.Color("black"), image_rect)
        #pg.draw.rect(image, pg.Color("green"), image_rect.inflate(-2, -2))
        #pg.draw.circle(image, GREY, [int(self.pos_x), int(self.pos_y)], 20, 3)
        return image



class RaptorsWorld(object):
    """
    A class to manage our event, game loop, and overall program flow.
    """
    CATCH_DISTANCE = 20

    def __init__(self, raptor_actor,  target, pigeons):
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

    def render(self):
        """
        Perform all necessary drawing and update the screen.
        """
        self.screen.fill(pg.Color("white"))
        for obj in self.target:
            obj.draw(self.screen)
        self.raptor_actor.draw(self.screen)
        pg.display.update()

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
            xa = self.target_pos[0]
            ya = self.target_pos[1]
            xb = self.raptor_actor.pos_x
            yb = self.raptor_actor.pos_y
            d = math.sqrt((ya -yb)*(ya - yb) + (xa - xb)*(xa - xb))
            self.raptor_actor.dist.append(d)
            delta = RaptorsWorld.CATCH_DISTANCE#10*RaptorActor.SIZE[0]/2
            #print("Distance :", d)
            if d <= delta:
                print("")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("+                                                     +")
                print("+                 Raptor captured pigeon              +")
                print("+                                                     +")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("")
                self.raptor_actor.render_traject(self.screen,self.pigeons, False)
                #self.pigeon_actor.render_traject(self.screen,self.pigeon_actor False)
                self.done = True
                break
            self.render()
            self.clock.tick(self.fps)

class Draw_Graphics(object):

    def __init__(self, raptor):
        self.alfa_L = raptor.alpha_L
        self.alfa_R = raptor.alpha_R
        self.d_L    = raptor.d_L
        self.d_R    = raptor.d_R
        self.collision = raptor.dist
        self.draw_graphics()

    def draw_graphics(self):
        t = []
        tt = []
        delta_alfa = []
        delta_d =  []
        for i in range(0,len(self.alfa_L )):
            t.append(i)

            #self.alfa_L[i] =  180/math.pi*normalize_angle(self.alfa_L[i])
            #self.alfa_R[i] = 180/math.pi*normalize_angle(self.alfa_R[i])
            delta_alfa.append((self.alfa_L[i] -self.alfa_R[i]))
            #delta_d.append((self.d_L[i] -self.d_R[i]))
            #print(" alfa_L = ",self.alfa_L[i]," alfa_R = ",self.alfa_R[i]," d_L = ",self.d_L[i]," d_RL = ",self.d_R[i])
        #t = np.arange(0.0, len(self.alfa_L),5)
        for i in range(0, len(self.collision)):
            tt.append(i)

        s1 = delta_alfa
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
    pg.font.init()  # you have to call this at the start,
    # if you want to use this module.
    #myfont = pygame.font.SysFont('Comic Sans MS', 30)
    wall = [0, 0, WALL_SIZE[0], WALL_SIZE[1]]
    pigeons = PigeonActor(1,wall)  # class adress for the pigeons
    target = pigeons.get_pigeons_obstacles()
    raptor_actor = RaptorActor((0, WALL_SIZE[1] / 2), pigeons, target, wall)
    world = RaptorsWorld(raptor_actor, target, pigeons)
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
