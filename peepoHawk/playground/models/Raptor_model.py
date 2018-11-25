#  version 25/11/2018


import math
import random
import numpy as np
from numpy import array
import pygame as pg
from pgmpy.models import BayesianModel

from peepoHawk.predictive_processing.v3.generative_model import GenerativeModel
from peepoHawk.predictive_processing.v3.sensory_input import SensoryInput
from peepoHawk.playground.models.CeePeeDees import CPD
from peepoHawk.visualize.graph import draw_network

vec = pg.math.Vector2

def normalize_angle(angle):
    '''if angle < 0:
        angle = 2 * math.pi + angle
    if angle >= 2 * math.pi:
        angle -= 2 * math.pi
    if angle <= -2 * math.pi:
        angle += 2 * math.pi'''
    return angle





class RaptorModel:
    RADIUS = 100

    def __init__(self, raptor_actor, Pigeons, wall):
        self.raptor_actor = raptor_actor
        self.Pigeons = Pigeons
        self.wall = wall
        self.n_sectors = raptor_actor.n_sectors
        self.cardinality_vision = self.n_sectors
        self.cardinality_delta_azimuth  = 2
        self.cardinality_correction = 3
        self.cardinality_direction = 2
        self.target = self.Pigeons.tensor_of_pigeons[0]
        self.R_previous = 0#raptor_actor.R_previous
        self.R_now = 0#raptor_actor.R_now
        self.motor_output = {pg.K_LEFT: False, pg.K_RIGHT: False}
        self.network = BayesianModel()
        self.models = self.create_networks()#see generative_model.py
        self.correction = 0
        self.cpd_left = CPD.create_fixed_parent((self.n_sectors), int(self.n_sectors/2))
        self.cpd_right = CPD.create_fixed_parent((self.n_sectors), int(self.n_sectors/3))
        self.cpd_left_previous = self.cpd_right.copy()
        self.cpd_right_previous = self.cpd_left.copy()



    def create_networks(self):
        gamma = 1 # this controls how steep the discrimination will be between the classes (gamma << 1 low discrimination, gamma >> 1 : high discrimination
        sigma = 1 # this controls how steep the squeezing of the action will be

        ParentNodes = []
        ParentNodes.append("Vision_left")
        ParentNodes.append("Vision_right")
        ParentNodes.append("Vision_left_previous")
        ParentNodes.append("Vision_right_previous")

        #ParentNodes.append("Reward_Predicted")
        count = 0
        while count < len(ParentNodes):
            self.network.add_node(ParentNodes[count])
            count = count+1

        LatentNodes = []
        LatentNodes.append("Delta_alfa_left")
        LatentNodes.append("Delta_alfa_right")
        LatentNodes.append("Direction")
        while count < len(LatentNodes):
            self.network.add_node(LatentNodes[count])
            count = count + 1

        LeafNodes = []
        LeafNodes.append("Correction")
        count = 0
        while count < len(LeafNodes):
            self.network.add_node(LeafNodes[count])
            count = count + 1
        self.network.add_edge(ParentNodes[0], LatentNodes[0])
        self.network.add_edge(ParentNodes[1], LatentNodes[1])
        self.network.add_edge(ParentNodes[2], LatentNodes[0])
        self.network.add_edge(ParentNodes[3], LatentNodes[1])
        self.network.add_edge(ParentNodes[0], LatentNodes[2])
        self.network.add_edge(ParentNodes[1], LatentNodes[2])
        self.network.add_edge(ParentNodes[2], LatentNodes[2])
        self.network.add_edge(ParentNodes[3], LatentNodes[2])
        self.network.add_edge(LatentNodes[0], LeafNodes[0])
        self.network.add_edge(LatentNodes[1], LeafNodes[0])
        self.network.add_edge(LatentNodes[2], LeafNodes[0])



        CPD_Parents = []
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[0],self.cardinality_vision, int(random.uniform(0,self.cardinality_vision-0.4)), sigma/2, "fixed"))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[1],self.cardinality_vision, int(random.uniform(0,self.cardinality_vision-0.4)), sigma/2, "fixed"))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[2],self.cardinality_vision, int(random.uniform(0,self.cardinality_vision-0.4)), sigma/2, "fixed"))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[3],self.cardinality_vision, int(random.uniform(0,self.cardinality_vision-0.4)), sigma/2, "fixed"))
        for n in range(0, len(CPD_Parents)):
            self.network.add_cpds(CPD_Parents[n])
        CPD_Latents = []
        CPD_Latents.append(CPD.latent_cpd(LatentNodes[0],self.cardinality_delta_azimuth,[self.cardinality_vision,self.cardinality_vision],[ParentNodes[0],ParentNodes[2]] ,'delta_alfa', gamma))
        CPD_Latents.append(CPD.latent_cpd(LatentNodes[1],self.cardinality_delta_azimuth, [self.cardinality_vision,self.cardinality_vision], [ParentNodes[1],ParentNodes[3]],'delta_alfa', gamma))
        CPD_Latents.append(CPD.latent_cpd(LatentNodes[2], self.cardinality_direction, [self.cardinality_vision,self.cardinality_vision,self.cardinality_vision,self.cardinality_vision], [ParentNodes[0], ParentNodes[2],ParentNodes[1], ParentNodes[3]], 'direction', gamma))
        for n in range(0,len(CPD_Latents)):
            self.network.add_cpds(CPD_Latents[n])
        CPD_Leafs = []
        CPD_Leafs.append(CPD.latent_cpd(LeafNodes[0],self.cardinality_correction,[self.cardinality_delta_azimuth,self.cardinality_delta_azimuth,self.cardinality_direction],[LatentNodes[0], LatentNodes[1], LatentNodes[2]], 'correction', gamma))

        for n in range(0,len(CPD_Leafs)):
            self.network.add_cpds(CPD_Leafs[n])
        self.network.check_model()
        '''draw_network(self.network)
        for n in range(0,len(CPD_Latents)):
            print("Latents :")
            print(CPD_Latents[n])
        for n in range(0,len(CPD_Leafs)):
            print("Leafs :")
            print(CPD_Leafs[n])'''
        #wait = input("PRESS ENTER TO CONTINUE.")
        return {'main': GenerativeModel(SensoryInputVirtualPeepo(self), self.network)}


    def process(self):
        self.calculate_environment()
        self.network.get_cpds('Vision_left').values = self.cpd_left
        self.network.get_cpds('Vision_right').values = self.cpd_right
        self.network.get_cpds('Vision_left_previous').values = self.cpd_left_previous
        self.network.get_cpds('Vision_right_previous').values = self.cpd_right_previous
        '''print("PARENT CPD's")
        print("******************************************************************************************************************")
        print("Previous left :", self.cpd_left_previous, " Previous right : ", self.cpd_right_previous)
        print("New      left :", self.cpd_left, " New      right : ", self.cpd_right)'''
        for key in self.models:

            er,  self.correction = self.models[key].process()
            index_correction_angle  = np.argmax(self.correction)
            angle_increment = 1 + 1j
            if index_correction_angle == 0:
                angle_increment  = np.exp(-1j*self.raptor_actor.alfa_increment)
                print("Turning right")
            if index_correction_angle == 1:
                angle_increment = np.exp(0*1j*self.raptor_actor.alfa_increment)
                print("No Turning")
            if index_correction_angle == 2:
                angle_increment = np.exp(1j*self.raptor_actor.alfa_increment)
                print("Turning left")
            if index_correction_angle == 3:
                angle_increment = np.exp(0*1j*self.raptor_actor.alfa_increment)
                print("Undecided - > No Turning")
            #print("Incrementing angle with ",angle_increment)
            #print("Angle from  " ,self.raptor_actor.angle)
            self.raptor_actor.angle *= angle_increment
            #print ( "to : ", self.raptor_actor.angle)
            #print("Peepo's new moving direction :  ", np.angle(self.raptor_actor.angle)*180/math.pi," degrees")
            self.cpd_left_previous = self.cpd_left
            self.cpd_right_previous = self.cpd_right
            '''print("Parent nodes ")
            print(self.network.get_cpds('Delta_alfa_left').values)
            print(self.network.get_cpds('Delta_alfa_right').values)
            print("Leaf nodes ")
            #print(self.network.get_cpds('Correction'))'''
            #print("Inference ----> ",self.correction)

    def get_sector_index(angle, sector, side):
        index = 0
        aim = angle.real/abs(angle.real)
        #print( " Angle vector : ", angle," and aim = ", aim, " for angle of : ", 180/math.pi*np.angle(angle) )

        #print("In get_sector_index")
        if side == 'Left':
            for sec in range(0,len(sector)-1):
                #print("Angle :" , 180/math.pi*angle, " for sectors ", sec , " to ", sec+1 ," (",180/math.pi*np.angle(sector[sec]),",", 180/math.pi*np.angle(sector[sec+1]),")")
                if  (np.angle(angle) >= np.angle(sector[sec]) and np.angle(angle) < np.angle(sector[sec+1])) :
                    index = sec
            if (np.angle(angle) >= np.angle(sector[len(sector)-2]) and np.angle(angle) < np.angle(sector[len(sector)-1])):
                index = len(sector)-2
        if side == 'Right':
            for sec in range(1,len(sector)-1):
                #print("Angle :" , 180/math.pi*angle, " for sectors ", sec , " to ", sec+1 ," (",180/math.pi*np.angle(sector[sec]),",", 180/math.pi*np.angle(sector[sec+1]),")")
                if  (np.angle(angle) >np.angle(sector[sec]) and np.angle(angle) <= np.angle(sector[sec+1])) :
                    index = sec
            if (np.angle(angle) >= np.angle(sector[0]) and np.angle(angle) <= np.angle(sector[1])):
                index = len(sector)-2
            '''if np.angle(angle) > np.angle(sector[len(sector)-1]):
                index = len(sector) - 2'''
        if side == 'Left':
            if aim*angle.real <= aim*sector[0].real and aim*angle.real >= aim*sector[len(sector)-1].real:
                index = int((len(sector) - 2)/2)
        if side == 'Right':
            if aim * angle.real >= aim * sector[0].real and aim * angle.real <= aim * sector[len(sector) - 1].real:
                index = int((len(sector) - 2) / 2)
            #index = 0
            '''if side == 'Left':
                index = len(sector)-2#int((len(sector)-1)/2)#0
            if side == 'Right':
                index = 0
        if np.angle(angle) >= np.angle(sector[len(sector)-1]):
            if side == 'Right':
                index = len(sector)-2#int((len(sector)-1)/2)#0
            if side == 'Left':
                index = 0'''

        #print("Angle :", 180 / math.pi * angle, " for sectors ", len(sector)-2, " to ", len(sector)-2 + 1, " (",  180 / math.pi * sector[len(sector)-2], ",", 180 / math.pi * sector[len(sector)-2+ 1], ")")
        #print("Index for angle ",angle , " and sector : ", sector, " : ", index)
        #print(side , " sector Index for angle ", 180/math.pi*np.angle(angle), " ------> ",  index, "   for sectors angles(", 180/math.pi*np.angle(sector[index]),", ", 180/math.pi*np.angle(sector[index+1]),")")
        return index

    def normalize(s):
        som = 0
        for sec in range(0, len(s)):
            som +=s[sec]
        for sec in range(0, len(s)):
            s[sec] /= som
        return s

    def calculate_environment(obj):
        pos_target = [obj.target[0], obj.target[1]]
        pos_raptor_left_eye = [obj.raptor_actor.left_eye[0], obj.raptor_actor.left_eye[1]]
        pos_raptor_right_eye = [obj.raptor_actor.right_eye[0], obj.raptor_actor.right_eye[1]]
        observed_left_angle  = (pos_target[0] - pos_raptor_left_eye[0] ) + 1j * (pos_target[1] - pos_raptor_left_eye[1])
        observed_right_angle = (pos_target[0] - pos_raptor_right_eye[0]) + 1j * (pos_target[1] - pos_raptor_right_eye[1])
        index_sector_left  = RaptorModel.get_sector_index(observed_left_angle, obj.raptor_actor.sector_L, 'Left')
        obj.raptor_actor.choosen_sector_L = index_sector_left
        index_sector_right = RaptorModel.get_sector_index(observed_right_angle, obj.raptor_actor.sector_R, 'Right')
        obj.raptor_actor.choosen_sector_R = index_sector_right
        obj.cpd_left_previous  = obj.cpd_left
        obj.cpd_right_previous = obj.cpd_right
        obj.cpd_left  = CPD.create_fixed_parent(obj.cardinality_vision,index_sector_left)
        obj.cpd_right = CPD.create_fixed_parent(obj.cardinality_vision,index_sector_right)
        #obj.cpd_left_previous  = obj.cpd_left
        #obj.cpd_right_previous = obj.cpd_right



class SensoryInputVirtualPeepo(SensoryInput):
    def __init__(self, raptor):
        super().__init__()
        self.raptor = raptor



    def value(self, name):
        #print("In sensory input")
        if 'Azimuth' in name:
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if 'next_cycle' in name:
                #print("with Azimuth_next_cycle")
                if self.raptor.target_sector == 0:
                    return np.array([1, 0, 0, 0, 0, 0, 0])
                if self.raptor.target_sector == 1:
                    return np.array([0, 1, 0, 0, 0, 0, 0])
                if self.raptor.target_sector == 2:
                    return np.array([0, 0, 1, 0, 0, 0, 0])
                if self.raptor.target_sector == 3:
                    return np.array([0, 0, 0, 1, 0, 0, 0])
                if self.raptor.target_sector == 4:
                    return np.array([0, 0, 0, 0, 1, 0, 0])
                if self.raptor.target_sector == 5:
                    return np.array([0, 0, 0, 0, 0, 1, 0])
                if self.raptor.target_sector == 6:
                    return np.array([0, 0, 0, 0, 0, 0, 1])


        if 'Reward' in name:
            #print("with Reward_next_cycle")
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if 'next_cycle' in name:
                if self.raptor.Reward == 0:
                    return np.array([1, 0, 0])
                if self.raptor.Reward == 1:
                    return np.array([0, 1, 0])
                if self.raptor.Reward == 2:
                    return np.array([0, 0, 1])

