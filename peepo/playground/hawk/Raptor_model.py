#version 28/11/2018


import random

import numpy as np
import pygame as pg
from peepo.pp.v3.sensory_input import SensoryInput
from pgmpy.models import BayesianModel

from peepo.playground.hawk.CeePeeDees import CPD
from peepo.pp.v3.generative_model import GenerativeModel

vec = pg.math.Vector2


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
        self.cardinality_direction = 3
        self.resolution = abs(np.angle(self.raptor_actor.sector_L[1]) - np.angle(self.raptor_actor.sector_L[0]))
        self.target = self.Pigeons.tensor_of_pigeons[0]
        self.R_previous = 0#raptor_actor.R_previous
        self.R_now = 0#raptor_actor.R_now
        self.observations = []
        self.network = BayesianModel()
        self.models = self.create_networks()#see generative_model.py
        self.correction = 0
        self.direction = 0
        self.compass = 0
        self.cpd_left  = CPD.create_fixed_parent((self.n_sectors), int(self.n_sectors/2))
        self.cpd_right = CPD.create_fixed_parent((self.n_sectors), int(self.n_sectors/3))
        self.cpd_left_previous = self.cpd_right.copy()
        self.cpd_right_previous = self.cpd_left.copy()
        self.cpd_left_observed = self.cpd_right_previous.copy()
        self.cpd_right_observed = self.cpd_left_previous.copy()
        self.cpd_action =  CPD.create_fixed_parent(self.cardinality_correction, int(self.cardinality_correction/2))
        self.cpd_direction =  CPD.create_fixed_parent(self.cardinality_direction , int(self.cardinality_direction / 2))





    def create_networks(self):
        gamma = 1 # this controls how steep the discrimination will be between the classes (gamma << 1 low discrimination, gamma >> 1 : high discrimination
        sigma = 1 # this controls how steep the squeezing of the action will be
        index_jump = int(self.raptor_actor.alfa_increment/self.resolution)#calculates the number of "sector  jumps"  dpeending on the tuple alfa_ainvrement and sector_resolution
        index_jump = 2
        #         print("Index gamma ", index_gamma)
        ParentNodes = []
        ParentNodes.append("MEM_vision_left")
        ParentNodes.append("MEM_vision_right")
        ParentNodes.append("BEN_Correction")
        ParentNodes.append("BEN_Direction")


        #ParentNodes.append("Reward_Predicted")
        count = 0
        while count < len(ParentNodes):
            self.network.add_node(ParentNodes[count])
            count = count+1


        LeafNodes = []
        LeafNodes.append("LEN_vision_left")
        LeafNodes.append("LEN_vision_right")
        LeafNodes.append("LEN_motor_Correction")
        LeafNodes.append("LEN_motor_Direction")

        count = 0
        while count < len(LeafNodes):
            self.network.add_node(LeafNodes[count])
            count = count + 1
        self.network.add_edge(ParentNodes[0], LeafNodes[0])
        self.network.add_edge(ParentNodes[2], LeafNodes[0])
        self.network.add_edge(ParentNodes[3], LeafNodes[0])
        self.network.add_edge(ParentNodes[1], LeafNodes[1])
        self.network.add_edge(ParentNodes[2], LeafNodes[1])
        self.network.add_edge(ParentNodes[3], LeafNodes[1])
        self.network.add_edge(ParentNodes[2], LeafNodes[2])
        self.network.add_edge(ParentNodes[3], LeafNodes[3])

        CPD_Parents = []
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[0],self.cardinality_vision, int(random.uniform(0,self.cardinality_vision-0.4)), sigma/2, "fixed"))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[1], self.cardinality_vision,int(random.uniform(0, self.cardinality_vision - 0.4)), sigma / 2, "fixed"))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[2],self.cardinality_correction, int(random.uniform(0,self.cardinality_correction-0.4)), sigma/2, "fixed"))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[3],self.cardinality_direction, int(random.uniform(0,self.cardinality_direction-0.4)), sigma/2, "fixed"))

        for n in range(0, len(CPD_Parents)):
            self.network.add_cpds(CPD_Parents[n])

        CPD_Leafs = []
        CPD_Leafs.append(CPD.leaf_cpd(LeafNodes[0],self.cardinality_vision,[self.cardinality_vision,self.cardinality_correction,self.cardinality_direction],[ParentNodes[0],ParentNodes[2],ParentNodes[3]],'left_vision',index_jump))
        CPD_Leafs.append(CPD.leaf_cpd(LeafNodes[1],self.cardinality_vision,[self.cardinality_vision,self.cardinality_correction,self.cardinality_direction],[ParentNodes[1],ParentNodes[2],ParentNodes[3]],'right_vision',index_jump))
        CPD_Leafs.append(CPD.leaf_cpd(LeafNodes[2],self.cardinality_correction,[self.cardinality_correction],[ParentNodes[2]],'one_2_one',index_jump))
        CPD_Leafs.append(CPD.leaf_cpd(LeafNodes[3], self.cardinality_direction, [self.cardinality_direction], [ParentNodes[3]],'one_2_one', index_jump))
        for n in range(0,len(CPD_Leafs)):
            self.network.add_cpds(CPD_Leafs[n])
        self.network.check_model()
        #draw_network(self.network)
        '''for n in range(0,len(CPD_Latents)):
            print("Latents :")
            print(CPD_Latents[n])'''
        for n in range(0,len(CPD_Leafs)):
            print("Leafs :")
            print(CPD_Leafs[n])
        #wait = input("PRESS ENTER TO CONTINUE.")
        relevant_parent_nodes = [2,3]

        return {'main': GenerativeModel(SensoryInputVirtualPeepo(self), self.network)}


    def process(self):
        self.calculate_environment()
        self.network.get_cpds('BEN_Correction').values = self.cpd_action
        self.network.get_cpds('BEN_Direction').values = self.cpd_direction
        self.network.get_cpds('MEM_vision_left').values = self.cpd_left_previous
        self.network.get_cpds('MEM_vision_right').values = self.cpd_right_previous
        '''print("PARENT CPD's")
        print("******************************************************************************************************************")
        print("Previous left :", self.cpd_left_previous, " Previous right : ", self.cpd_right_previous)
        print("New  correction :", self.action, " New   direction : ", self.direction)'''

        for key in self.models:
            err = self.models[key].process()
            self.correction = self.network.get_cpds('BEN_Correction').values
            self.direction = self.network.get_cpds('BEN_Direction').values
            self.cpd_action = self.correction
            self.cpd_direction = self.direction
            index_direction_angle = np.argmax(self.direction)
            index_correction_angle  = np.argmax(self.correction)
            angle_increment =  1j
            if  index_direction_angle >=0:
                if index_correction_angle == 0:
                    angle_increment  = np.exp(-1j*self.raptor_actor.alfa_increment)
                    print("Turning right")
                if index_correction_angle == 1:
                    angle_increment = np.exp(0*1j*self.raptor_actor.alfa_increment)
                    print("No Turning")
                if index_correction_angle == 2:
                    angle_increment = np.exp(1j*self.raptor_actor.alfa_increment)
                    print("Turning left")
            if  index_direction_angle > 3:
                if index_correction_angle == 2:
                    angle_increment  = np.exp(-1j*self.raptor_actor.alfa_increment)
                    print("Turning right")
                if index_correction_angle == 1:
                    angle_increment = np.exp(0*1j*self.raptor_actor.alfa_increment)
                    print("No Turning")
                if index_correction_angle == 0:
                    angle_increment = np.exp(1j*self.raptor_actor.alfa_increment)
                    print("Turning left")

            #print("Incrementing angle with ",angle_increment)
            #print("Angle from  " ,self.raptor_actor.angle)
            self.raptor_actor.angle *= angle_increment
            #print ( "to : ", self.raptor_actor.angle)
            #print("Peepo's new moving direction :  ", np.angle(self.raptor_actor.angle)*180/math.pi," degrees")
            self.cpd_left_previous = self.cpd_left_observed
            self.cpd_right_previous = self.cpd_right_observed
            #self.action = CPD.create_fixed_parent(self.cardinality_correction, self.correction)
            #self.direction = CPD.create_fixed_parent(self.cardinality_correction, self.compass)
            '''print("Parent nodes ")
            print(self.network.get_cpds('Delta_alfa_left').values)
            print(self.network.get_cpds('Delta_alfa_right').values)
            print("Leaf nodes ")
            #print(self.network.get_cpds('Correction'))'''
            #print("Inference ----> ",self.correction)


    def calculate_environment(obj):
        pos_target = [obj.target[0], obj.target[1]]
        pos_raptor_left_eye = [obj.raptor_actor.left_eye[0], obj.raptor_actor.left_eye[1]]
        pos_raptor_right_eye = [obj.raptor_actor.right_eye[0], obj.raptor_actor.right_eye[1]]
        observed_left_angle  = (pos_target[0] - pos_raptor_left_eye[0] ) + 1j * (pos_target[1] - pos_raptor_left_eye[1])
        observed_right_angle = (pos_target[0] - pos_raptor_right_eye[0]) + 1j * (pos_target[1] - pos_raptor_right_eye[1])
        index_sector_left  = RaptorModel.get_sector_index(observed_left_angle, obj.raptor_actor.sector_L, 'Left')
        obj.raptor_actor.choosen_sector_L = index_sector_left
        if obj.raptor_actor.choosen_sector_L == len(obj.raptor_actor.sector_L) -2:
            obj.raptor_actor.choosen_sector_L = len(obj.raptor_actor.sector_L) - 1
        index_sector_right = RaptorModel.get_sector_index(observed_right_angle, obj.raptor_actor.sector_R, 'Right')
        obj.raptor_actor.choosen_sector_R = index_sector_right
        if obj.raptor_actor.choosen_sector_R == 0*(len(obj.raptor_actor.sector_R) -2):
            obj.raptor_actor.choosen_sector_R = (len(obj.raptor_actor.sector_R) - 1)
        obj.cpd_left_observed  = CPD.create_fixed_parent(obj.cardinality_vision,index_sector_left)
        obj.cpd_right_observed = CPD.create_fixed_parent(obj.cardinality_vision,index_sector_right)
        obj.observations.clear()
        obj.observations.append(obj.cpd_left_observed)
        obj.observations.append(obj.cpd_right_observed)
        #print("Observations -------------------", obj.observations)

    def get_sector_index(angle, sector, side):
        index = 0
        aim = angle.real/abs(angle.real)
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

        if side == 'Left':
            if aim*angle.real <= aim*sector[0].real and aim*angle.real >= aim*sector[len(sector)-1].real:
                index = int((len(sector) - 2)/2)
        if side == 'Right':
            if aim * angle.real >= aim * sector[0].real and aim * angle.real <= aim * sector[len(sector) - 1].real:
                index = int((len(sector) - 2) / 2)

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


class SensoryInputVirtualPeepo(SensoryInput):
    def __init__(self, obj):
        super().__init__()
        self.peepo = obj
    def action(self, node, prediction):
        # if prediction = [0.9, 0.1] (= moving) then move else stop
        if 'Correction' in node:
            return self.peepo.cpd_action
        if 'Direction' in node:
            return self.peepo.cpd_direction

    def value(self, name):
        if 'vision_left' in name:
            return self.peepo.cpd_left_observed
        if 'vision_right' in name:
            return self.peepo.cpd_right_observed
        if 'Correction' in name:
            return self.peepo.cpd_action
        if 'Direction' in name:
            return self.peepo.cpd_direction