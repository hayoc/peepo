#version 15/11/2018


import math
import random
import numpy as np
from numpy import array
import pygame as pg
from pgmpy.models import BayesianModel

from predictive_processing.v3.generative_model import GenerativeModel
from predictive_processing.v3.sensory_input import SensoryInput
from playground.models.CeePeeDees import CPD
from visualize.graph import draw_network

vec = pg.math.Vector2

def normalize_angle(angle):
    return angle
    if angle < 0:
        angle = 2 * math.pi + angle
    if angle >= 2 * math.pi:
        angle -= 2 * math.pi
    if angle <= -2 * math.pi:
        angle += 2 * math.pi
    return angle



class RaptorModel:
    RADIUS = 100
    ANGLE_INCREMENT = 10#degrees

    def __init__(self, raptor_actor, Pigeons, wall):
        self.raptor_actor = raptor_actor
        self.Pigeons = Pigeons
        self.wall = wall
        self.n_sectors = raptor_actor.n_sectors
        print(self.Pigeons)
        self.target = self.Pigeons.tensor_of_pigeons[0]
        self.R_previous = 0#raptor_actor.R_previous
        self.R_now = 0#raptor_actor.R_now
        self.motor_output = {pg.K_LEFT: False, pg.K_RIGHT: False}
        self.target_sector = 3
        self.distance_now = 10000
        self.distance_previous = self.distance_now
        self.Reward = 0
        self.network = BayesianModel()
        self.models = self.create_networks()#see generative_model.py
        self.observed_left_angle = 0
        self.correction = 0
        self.observed_right_angle = 0
        self.previous_left_angle = 0
        self.previous_correction = 0
        self.previous_right_angle = 0
        self.cpd_left = [0.5,0.5]
        self.cpd_right = [0.5, 0.5]


    def create_networks(self):
        gamma = 1 # this controls how steep the discrimination will be between the classes (gamma << 1 low discrimination, gamma >> 1 : high discrimination
        sigma = 1 # this controls how steep the squeezing of the action will be

        ParentNodes = []
        ParentNodes.append("Delta_alfa_left")
        ParentNodes.append("Delta_alfa_right")
        #ParentNodes.append("Reward_Predicted")
        count = 0
        while count < len(ParentNodes):
            self.network.add_node(ParentNodes[count])
            count = count+1

        #LatentNodes = []
        #LatentNodes.append("Delta_Azimuth")
        #LatentNodes.append("Delta_Reward")
        #LatentNodes.append("Action")
        count = 0
        '''while count < len(LatentNodes):
            self.network.add_node(LatentNodes[count])
            count = count + 1'''

        LeafNodes = []
        LeafNodes.append("Correction")
        count = 0
        while count < len(LeafNodes):
            self.network.add_node(LeafNodes[count])
            count = count + 1

        self.network.add_edge(ParentNodes[0], LeafNodes[0])
        self.network.add_edge(ParentNodes[1], LeafNodes[0])

        '''self.network.add_edge(LatentNodes[0], LeafNodes[0])
        self.network.add_edge(LatentNodes[0], LeafNodes[1])
        self.network.add_edge(LatentNodes[1], LeafNodes[0])
        self.network.add_edge(LatentNodes[1], LeafNodes[1])
        self.network.add_edge(LatentNodes[2], LeafNodes[0])
        self.network.add_edge(LatentNodes[2], LeafNodes[1])'''

        cardinality_azimuth = self.n_sectors
        cardinality_delta_azimuth  = 2
        cardinality_correction = 2

        CPD_Parents = []
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[0],cardinality_delta_azimuth, int(random.uniform(0,cardinality_delta_azimuth+0.4)), sigma/2))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[1],cardinality_delta_azimuth, int(random.uniform(0,cardinality_delta_azimuth+0.4)), sigma/2))

        for n in range(0, len(CPD_Parents)):
            self.network.add_cpds(CPD_Parents[n])
        count = 0
        '''CPD_Latents = []
        CPD_Latents.append(CPD.latent_cpd(LatentNodes[0],cardinality_azimuth,[cardinality_azimuth,cardinality_azimuth],[ParentNodes[0],ParentNodes[1]], 'fixed', gamma))
        for n in range(0,len(CPD_Latents)):
            self.network.add_cpds(CPD_Latents[n])'''
        CPD_Leafs = []
        CPD_Leafs.append(CPD.latent_cpd(LeafNodes[0],cardinality_correction,[cardinality_delta_azimuth,cardinality_delta_azimuth],[ParentNodes[0], ParentNodes[1]], 'angle_correction', gamma))

        for n in range(0,len(CPD_Leafs)):
            self.network.add_cpds(CPD_Leafs[n])
        self.network.check_model()
        draw_network(self.network)
        for n in range(0,len(CPD_Leafs)):
            print("Leafs :")
            print(CPD_Leafs[n])
        #wait = input("PRESS ENTER TO CONTINUE.")
        return {'main': GenerativeModel(SensoryInputVirtualPeepo(self), self.network)}

    def Raptor_Correction(self, index_correction):
        sigma = 1
        x = (3 - index_correction)/sigma
        correction = 1 - math.exp(x)
        return correction

    def process(self):
        self.calculate_environment()
        for key in self.models:
            er,  self.correction = self.models[key].process()
            index_correction_angle  = np.argmax(self.correction)
            sign_corr = 1
            print("index correction :", index_correction_angle)
            if index_correction_angle == 0:
                sign_corr  *= -1
            angle_increment = sign_corr*RaptorModel.ANGLE_INCREMENT*math.pi/180
            self.raptor_actor.angle = normalize_angle(self.raptor_actor.angle + angle_increment)
            print("Peepo's new moving direction :  ", self.raptor_actor.angle*180/math.pi," degrees")
            self.raptor_actor.update_sectors()
            self.network.get_cpds('Delta_alfa_left').values = self.cpd_left
            self.network.get_cpds('Delta_alfa_right').values = self.cpd_right
            print("Parent nodes ")
            print(self.network.get_cpds('Delta_alfa_left').values)
            print(self.network.get_cpds('Delta_alfa_right').values)
            print("Leaf nodes ")
            #print(self.network.get_cpds('Correction'))
            print("Inference")
            print(self.correction)

    def calculate_environment(self):
        raptor_vec = vec(self.raptor_actor.rect.center)

        #first check if no collision with the wall occurred
        if raptor_vec[0] <= self.wall[0]:
            if math.cos(self.raptor_actor.angle) == 0:
                self.raptor_actor.angle = 0.009*math.pi/2
            self.raptor_actor.angle = math.atan(math.sin(self.raptor_actor.angle)/math.cos(-self.raptor_actor.angle))
        if raptor_vec[1] <= self.wall[1]:
            if math.cos(self.raptor_actor.angle) == 0:
                self.raptor_actor.angle = 0.009 * math.pi / 2
            self.raptor_actor.angle = math.atan(math.sin(-self.raptor_actor.angle)/math.cos(self.raptor_actor.angle))
        if raptor_vec[0] >= self.wall[2]:
            if math.cos(self.raptor_actor.angle) == 0:
                self.raptor_actor.angle = 0.009*math.pi/2
            self.raptor_actor.angle = math.atan(math.sin(self.raptor_actor.angle)/math.cos(-self.raptor_actor.angle))
        if raptor_vec[1] >= self.wall[3]:
            if math.cos(self.raptor_actor.angle) == 0:
                self.raptor_actor.angle = 0.009 * math.pi / 2
            self.raptor_actor.angle = math.atan(math.sin(-self.raptor_actor.angle)/math.cos(self.raptor_actor.angle))

        #Calculate delta_angle left and right
        self.previous_left_angle = self.observed_left_angle
        self.previous_right_angle = self.observed_right_angle
        pos_target = [self.target[0], self.target[1]]
        pos_raptor = [self.raptor_actor.pos_x,self.raptor_actor.pos_y ]
        pos_raptor_left_eye = [self.raptor_actor.left_eye[0], self.raptor_actor.left_eye[1]]
        pos_raptor_right_eye = [self.raptor_actor.right_eye[0], self.raptor_actor.right_eye[1]]
        tan_left = (pos_target[1] - pos_raptor_left_eye[1])/(pos_target[0] - pos_raptor_left_eye[0])
        tan_right = (pos_target[1] - pos_raptor_right_eye[1]) / (pos_target[0] - pos_raptor_right_eye[0])
        self.observed_left_angle = math.atan(tan_left)
        self.observed_right_angle = math.atan(tan_right)
        delta_alfa_left = self.observed_left_angle -  self.previous_left_angle
        delta_alfa_right= self.observed_right_angle - self.previous_right_angle
        self.cpd_left = [0.5, 0.5]
        self.cpd_right = [0.5, 0.5]
        if  delta_alfa_left < 0:
            self.cpd_left = [0.9, 0.1]
        if  delta_alfa_left > 0:
            self.cpd_left = [0.1, 0.9]
        if  delta_alfa_right < 0:
            self.cpd_right = [0.9, 0.1]
        if  delta_alfa_right > 0:
            self.cpd_right = [0.1, 0.9]
        #we suppose Peepo has the capability to assess the direction and the distance of the prey
        for target in self.target:
            #calculate in which sector the target is
            absolute_angle_target = normalize_angle(math.atan((self.Pigeons.pos_y - raptor_vec[1])/(self.Pigeons.pos_x - raptor_vec[0])))
            if absolute_angle_target < self.raptor_actor.sector[1] or absolute_angle_target <= self.raptor_actor.sector[0]:
                self.target_sector = 0
            if absolute_angle_target >= self.raptor_actor.sector[1]:
                self.target_sector = 1
                if absolute_angle_target >= self.raptor_actor.sector[2]:
                    self.target_sector = 2
                    if absolute_angle_target >= self.raptor_actor.sector[3]:
                        self.target_sector = 3
                        if absolute_angle_target >= self.raptor_actor.sector[4]:
                            self.target_sector = 4
                            if absolute_angle_target >= self.raptor_actor.sector[5]:
                                self.target_sector = 5
                                if absolute_angle_target >= self.raptor_actor.sector[6]:
                                    self.target_sector = 6
                                    if absolute_angle_target >= self.raptor_actor.sector[7]:
                                        self.target_sector = 6
            #print("Absolute angle of the target viewed from Peepo : ", absolute_angle_target*180/math.pi, " and choosen sector has index   : ",self.target_sector, " with an angle of ", self.raptor_actor.sector[self.target_sector ]*180/math.pi, " degrees.")

            #self.observed_azimuth = self.target_sector


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

