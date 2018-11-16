#version 15/11/2018


import math
import random
import numpy as np
from numpy import array
import pygame as pg
from pgmpy.models import BayesianModel

from peepoHawk.playground.util.vision import observation, end_line
from peepoHawk.predictive_processing.v3.generative_model import GenerativeModel
from peepoHawk.predictive_processing.v3.sensory_input import SensoryInput
from peepoHawk.playground.models.CeePeeDees import CPD

from peepoHawk.visualize.graph import draw_network

vec = pg.math.Vector2

def normalize_angle(angle):
    if angle < 0:
        angle = 2 * math.pi + angle
    if angle >= 2 * math.pi:
        angle -= 2 * math.pi
    if angle <= -2 * math.pi:
        angle += 2 * math.pi
    return angle



class PeepoModel:
    RADIUS = 100

    def __init__(self, peepo_actor, Poopies, wall):
        self.peepo_actor = peepo_actor
        self.Poopies = Poopies
        self.wall = wall
        self.target = self.Poopies.get_poopies_obstacles()
        self.R_previous = peepo_actor.R_previous
        self.R_now = peepo_actor.R_now
        self.motor_output = {pg.K_LEFT: False, pg.K_RIGHT: False}
        self.target_sector = 3
        self.distance_now = 10000
        self.distance_previous = self.distance_now
        self.Reward = 0
        self.network = BayesianModel()
        self.models = self.create_networks()#see generative_model.py
        self.next_azimuth = 0
        self.observed_azimuth = 0
        self.next_reward = 0
        self.observed_reward = 0



    def create_networks(self):
        gamma = 1 # this controls how steep the discrimination will be between the classes (gamma << 1 low discrimination, gamma >> 1 : high discrimination
        sigma = 1 # this controls how steep the squeezing of the action will be

        ParentNodes = []
        ParentNodes.append("Azimuth_Belief")
        ParentNodes.append("Azimuth_Predicted")
        ParentNodes.append("Reward_Belief")
        ParentNodes.append("Reward_Predicted")
        count = 0
        while count < len(ParentNodes):
            self.network.add_node(ParentNodes[count])
            count = count+1

        LatentNodes = []
        LatentNodes.append("Delta_Azimuth")
        LatentNodes.append("Delta_Reward")
        LatentNodes.append("Action")
        count = 0
        while count < len(LatentNodes):
            self.network.add_node(LatentNodes[count])
            count = count + 1

        '''LeafNodes = []
        LeafNodes.append("Azimuth_next_cycle")
        LeafNodes.append("Reward_next_cycle")
        count = 0
        while count < len(LeafNodes):
            self.network.add_node(LeafNodes[count])
            count = count + 1'''

        self.network.add_edge(ParentNodes[0], LatentNodes[0])
        self.network.add_edge(ParentNodes[1], LatentNodes[0])
        self.network.add_edge(ParentNodes[2], LatentNodes[1])
        self.network.add_edge(ParentNodes[3], LatentNodes[1])
        self.network.add_edge(LatentNodes[0], LatentNodes[2])
        self.network.add_edge(LatentNodes[1], LatentNodes[2])

        '''self.network.add_edge(LatentNodes[0], LeafNodes[0])
        self.network.add_edge(LatentNodes[0], LeafNodes[1])
        self.network.add_edge(LatentNodes[1], LeafNodes[0])
        self.network.add_edge(LatentNodes[1], LeafNodes[1])
        self.network.add_edge(LatentNodes[2], LeafNodes[0])
        self.network.add_edge(LatentNodes[2], LeafNodes[1])'''

        cardinality_azimuth = 7
        cardinality_reward  = 3
        cardinality_action  = cardinality_azimuth#3
        CPD_Parents = []
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[0],cardinality_azimuth, int(cardinality_azimuth/2), sigma/2))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[1],cardinality_azimuth, int(cardinality_azimuth/2), sigma/2))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[2],cardinality_reward,  int(cardinality_reward/2), sigma/2))
        CPD_Parents.append(CPD.parent_cpd(ParentNodes[3],cardinality_reward,  int(cardinality_reward/2), sigma/2))
        for n in range(0, len(CPD_Parents)):
            self.network.add_cpds(CPD_Parents[n])
        count = 0
        CPD_Latents = []
        CPD_Latents.append(CPD.latent_cpd(LatentNodes[0],cardinality_azimuth,[cardinality_azimuth,cardinality_azimuth],[ParentNodes[0],ParentNodes[1]], 'fixed', gamma))
        CPD_Latents.append(CPD.latent_cpd(LatentNodes[1],cardinality_reward,[cardinality_reward,cardinality_reward],[ParentNodes[2],ParentNodes[3]],'reward', gamma))
        CPD_Latents.append(CPD.latent_cpd(LatentNodes[2],cardinality_action,[cardinality_azimuth, cardinality_reward], [LatentNodes[0], LatentNodes[1]], 'action', sigma))
        for n in range(0,len(CPD_Latents)):
            self.network.add_cpds(CPD_Latents[n])
        '''CPD_Leafs = []
        CPD_Leafs.append(CPD.leaf_cpd(LeafNodes[0],cardinality_azimuth,[cardinality_azimuth,cardinality_reward,cardinality_action ,],[LatentNodes[0], LatentNodes[1], LatentNodes[2]], 'azimuth', gamma))
        CPD_Leafs.append(CPD.leaf_cpd(LeafNodes[1],cardinality_reward ,[cardinality_azimuth,cardinality_reward,cardinality_action,],[LatentNodes[0], LatentNodes[1], LatentNodes[2]], 'reward', gamma))

        for n in range(0,len(CPD_Leafs)):
            self.network.add_cpds(CPD_Leafs[n])'''
        #draw_network(network)
        self.network.check_model()
        '''print("ROOTS")
        print(self.network.get_roots())
        print("LEAVES")
        print(self.network.get_leaves())'''
        '''for n in range(0,len(CPD_Parents)):
            print("Parents :")
            print(CPD_Parents[n])
        for n in range(0,len(CPD_Latents)):
            print("Latents :")
            print(CPD_Latents[n])
        for n in range(0,len(CPD_Leafs)):
            print("Leafs :")
            print(CPD_Leafs[n])'''
        #wait = input("PRESS ENTER TO CONTINUE.")
        return {'main': GenerativeModel(SensoryInputVirtualPeepo(self), self.network)}

    def Azimuth_Correction(self,action, alpha):
        delta = alpha - action
        correction = delta*1
        return correction

    def process(self):
        self.calculate_environment()
        for key in self.models:
            er, self.next_azimuth, self.next_reward = self.models[key].process()
            #print("Next azimuth ----> ", self.next_azimuth)
            #print("Next reward  ----> ", self.next_reward)
            #azimuth = self.network.get_cpds('Action').values
            #azimuth = azimuth.reshape((azimuth.shape[0],-1),order = 'F')
            index_action_angle  = np.argmax(self.next_azimuth)
            print("Action a s next azimuth :", self.next_azimuth, " giving an index_peepos_angle = ", index_action_angle, " and next_reward = ", self.next_reward)
            action = (self.peepo_actor.sector[index_action_angle] + self.peepo_actor.sector[index_action_angle+1])/2
            azimuth_correction = self.Azimuth_Correction(action, self.peepo_actor.angle)
            #print("Adapted sector = ",pos_azimuth)
            self.peepo_actor.angle = normalize_angle(self.peepo_actor.angle + azimuth_correction)
            print("Peepo's new moving direction :  ", self.peepo_actor.angle*180/math.pi," degrees")
            self.peepo_actor.update_sectors()
            self.network.get_cpds('Azimuth_Belief').values = CPD.updated_cpd(7, self.observed_azimuth)
            self.network.get_cpds('Reward_Belief').values = CPD.updated_cpd(3, self.observed_reward)
            self.network.get_cpds('Reward_Predicted').values = self.next_reward
            self.network.get_cpds('Azimuth_Predicted').values = self.next_azimuth
            '''print(self.network.get_cpds('Azimuth_Belief'))
            print(self.network.get_cpds('Reward_Belief'))
            print(self.network.get_cpds('Reward_Predicted'))
            print(self.network.get_cpds('Azimuth_Predicted'))'''

    def calculate_environment(self):
        peepo_vec = vec(self.peepo_actor.rect.center)
        #print("peepo_vec = ", peepo_vec)

        #first check if no collision with the wall occurred
        if peepo_vec[0] <= self.wall[0]:
            if math.cos(self.peepo_actor.angle) == 0:
                self.peepo_actor.angle = 0.009*math.pi/2
                self.peepo_actor.angle = math.atan(math.sin(self.peepo_actor.angle)/math.cos(-self.peepo_actor.angle))
        if peepo_vec[1] <= self.wall[1]:
            if math.cos(self.peepo_actor.angle) == 0:
                self.peepo_actor.angle = 0.009 * math.pi / 2
                self.peepo_actor.angle = math.atan(math.sin(-self.peepo_actor.angle)/math.cos(self.peepo_actor.angle))
        if peepo_vec[0] >= self.wall[2]:
            if math.cos(self.peepo_actor.angle) == 0:
                self.peepo_actor.angle = 0.009*math.pi/2
                self.peepo_actor.angle = math.atan(math.sin(self.peepo_actor.angle)/math.cos(-self.peepo_actor.angle))
        if peepo_vec[1] >= self.wall[3]:
            if math.cos(self.peepo_actor.angle) == 0:
                self.peepo_actor.angle = 0.009 * math.pi / 2
                self.peepo_actor.angle = math.atan(math.sin(-self.peepo_actor.angle)/math.cos(self.peepo_actor.angle))

        #Calculate distance and sector of the target
        #we suppose Peepo has the capability to assess the direction and the distance of the prey
        for target in self.target:
            #print("Target = ", self.Poopies.pos_x, self.Poopies.pos_y)
            #distance (is in fact the square of the distance but this doesn't matter here
            self.distance_now = (self.Poopies.pos_x - peepo_vec[0])*(self.Poopies.pos_x - peepo_vec[0])  + (self.Poopies.pos_y - peepo_vec[1])*(self.Poopies.pos_y - peepo_vec[1])
            #print( "Distance now = ", math.sqrt(self.distance_now), " and previous distance = ",  math.sqrt(self.distance_previous))
            if self.distance_now - self.distance_previous < 0:
                self.Reward = 2
            if self.distance_now - self.distance_previous > 0:
                self.Reward = 0
            if self.distance_now - self.distance_previous == 0:
                self.Reward = 1
            self.distance_previous = self.distance_now
            #calculate in which sector the target is
            absolute_angle_target = normalize_angle(math.atan((self.Poopies.pos_y - peepo_vec[1])/(self.Poopies.pos_x - peepo_vec[0])))
            if absolute_angle_target < self.peepo_actor.sector[1] or absolute_angle_target <= self.peepo_actor.sector[0]:
                self.target_sector = 0
            if absolute_angle_target >= self.peepo_actor.sector[1]:
                self.target_sector = 1
                if absolute_angle_target >= self.peepo_actor.sector[2]:
                    self.target_sector = 2
                    if absolute_angle_target >= self.peepo_actor.sector[3]:
                        self.target_sector = 3
                        if absolute_angle_target >= self.peepo_actor.sector[4]:
                            self.target_sector = 4
                            if absolute_angle_target >= self.peepo_actor.sector[5]:
                                self.target_sector = 5
                                if absolute_angle_target >= self.peepo_actor.sector[6]:
                                    self.target_sector = 6
                                    if absolute_angle_target >= self.peepo_actor.sector[7]:
                                        self.target_sector = 6
            print("Absolute angle of the target viewed from Peepo : ", absolute_angle_target*180/math.pi, " and choosen sector has index   : ",self.target_sector, " with an angle of ", self.peepo_actor.sector[self.target_sector ]*180/math.pi, " degrees.")
            #print("The sectors are : ", self.peepo_actor.sector*180/math.pi)
            #print("Target is sector ", self.target_sector)
            #print("Reward ", self.Reward)
            self.observed_azimuth = self.target_sector
            self.observed_reward = self.Reward


class SensoryInputVirtualPeepo(SensoryInput):
    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo



    def value(self, name):
        #print("In sensory input")
        if 'Azimuth' in name:
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if 'next_cycle' in name:
                #print("with Azimuth_next_cycle")
                if self.peepo.target_sector == 0:
                    return np.array([1, 0, 0, 0, 0, 0, 0])
                if self.peepo.target_sector == 1:
                    return np.array([0, 1, 0, 0, 0, 0, 0])
                if self.peepo.target_sector == 2:
                    return np.array([0, 0, 1, 0, 0, 0, 0])
                if self.peepo.target_sector == 3:
                    return np.array([0, 0, 0, 1, 0, 0, 0])
                if self.peepo.target_sector == 4:
                    return np.array([0, 0, 0, 0, 1, 0, 0])
                if self.peepo.target_sector == 5:
                    return np.array([0, 0, 0, 0, 0, 1, 0])
                if self.peepo.target_sector == 6:
                    return np.array([0, 0, 0, 0, 0, 0, 1])


        if 'Reward' in name:
            #print("with Reward_next_cycle")
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if 'next_cycle' in name:
                if self.peepo.Reward == 0:
                    return np.array([1, 0, 0])
                if self.peepo.Reward == 1:
                    return np.array([0, 1, 0])
                if self.peepo.Reward == 2:
                    return np.array([0, 0, 1])

