#version 11/11/2018


import math
import random
import numpy as np
from numpy import array
import pygame as pg
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepoHawk.playground.util.vision import observation, end_line
from peepoHawk.predictive_processing.v3.generative_model import GenerativeModel
from peepoHawk.predictive_processing.v3.sensory_input import SensoryInput

from peepoHawk.visualize.graph import draw_network

vec = pg.math.Vector2

def get_index_matrix(cardinality):#creates a matrix for the header of the contingency table (used  in create latent distribution with a fixed distibution)
    C = np.prod(cardinality)
    blocks = np.copy(cardinality)
    B = len(blocks)
    for b in range(1,B):
        index = B-1-b
        blocks[index] = blocks[index+1]*cardinality[index]
    M = np.zeros((len(cardinality), C))
    #construct first the lowest row
    block = np.zeros(cardinality[len(cardinality) - 1])
    for n in range(0,len(block)):
        block[n] = n
    #fill M  with the right number of blocks
    n_blocks = int(C/blocks[B-1])
    R = []
    for n in range(0,n_blocks):
        R.append(block)
    R = np.reshape(R, n_blocks*len(block))
    M[B-1,:] = R
    block_mem = int(C / blocks[B - 1])
    #now the rest of the rows
    for b in range(0, B-1):
        row = B - 2 - b
        block = np.zeros(blocks[row])
        block_mem /= cardinality[row]
        n_blocks = int(block_mem)
        #fill first the block
        index = 0
        index_ = 0
        for p in range(0,len(block)):
            block[p] = index
            index_ += 1
            if index_ > blocks[row+1]-1:
                index_ = 0
                index += 1
        #now create an R array with the right number of blocks
        R = []
        for n in range(0, n_blocks):
            R.append(block)
        R = np.reshape(R, n_blocks * len(block))
        M[row, :] = R
    return M

def create_action_distribution(card_latent, card_parent, sigma):
    #CREATES : a CPD with a distribution depending on the "distance" of the latent variable index to the indexes of the parents
    #the distance is the inverse of an exponentional of the sum of the distances coorected with a factor sigma
    #cardinality of the latent must be the same as the cardinality of the parents
    C = np.prod(card_parent)
    matrix = np.zeros((card_latent, C))
    M = get_index_matrix(card_parent)
    for row in range(0,card_latent):
        for column  in range(0, C):
            correction = 1
            if M[1][column] == 2:
                x = (row - M[0][column])*(row - M[0][column])/sigma/sigma
                correction = math.exp(-x)
            if M[1][column] == 1:
                correction = 1
            if M[1][column] == 0:
                x = (row - M[0][column])*(row - M[0][column])/sigma/sigma
                correction = 1 - math.exp(-x)
            matrix[row][column] = correction
    #Normalize ditribution
    for column in range(0, C):
        factor = 0
        for row in range(0, card_latent):
            factor += matrix[row][column]
        for row in range(0, card_latent):
            matrix[row][column] /= factor
    return matrix

def create_latent_distribution(card_latent, card_parent, gamma):
    #CREATES : aCPD a distribution depending on the "distance" of the latent variable index to the indexes of the parents
    #the distance is the inverse of an exponentional of the sum of the distances coorected with a factor (set to 1 for the moment)
    #cardinality of the latent must be the same as the cardinality of the parents
    C = np.prod(card_parent)
    matrix = np.zeros((card_latent, C))
    M = get_index_matrix(card_parent)
    for row in range(0,card_latent):
        for column  in range(0, C):
            distance = 0
            for M_row in range(0, len(card_parent)):
                distance  += abs(row - M[M_row][column])
            matrix[row][column] = math.exp(-gamma * distance)
    #Normalize ditribution
    for column in range(0, C):
        factor = 0
        for row in range(0, card_latent):
            factor += matrix[row][column]
        for row in range(0, card_latent):
            matrix[row][column] /= factor
    return matrix

def create_leaf_distribution(card_leaf, card_latent, modus, sigma):
    if modus == "azimuth":
        C = np.prod(card_latent)
        matrix = np.zeros((card_leaf, C))
        M = get_index_matrix(card_latent)
        for column in range(0, C):
            delta_reward = M[1][column]
            for row in range(0, card_leaf):
                delta_index = abs(M[0][column] - M[2][column])
                if delta_reward == 2:
                    mu = delta_index
                    y = (row - mu) * (row - mu) / sigma / sigma
                    matrix[row][column] = math.exp(-y)
                if delta_reward == 0:
                    mu = card_leaf - delta_index
                    y = (row - mu) * (row - mu) / sigma / sigma
                    matrix[row][column] = math.exp(-y)
                if delta_reward == 1:
                    mu = card_leaf/2
                    y = (row - mu) * (row - mu) / sigma / sigma
                    matrix[row][column] = math.exp(-y)

    if modus == "reward":
        C = np.prod(card_latent)
        matrix = np.zeros((card_leaf, C))
        M = get_index_matrix(card_latent)
        for column in range(0, C):
            delta_index = abs(M[0][column] - M[2][column])
            delta_reward =  M[1][column]
            y_low = 10 + 90 / card_latent[0] * delta_index
            y_high= 100 - 90 / card_latent[0] * delta_index
            for row in range(0, card_leaf):
                matrix[row][column] = 45
                if delta_reward == 0 and row == 0 :
                    matrix[row][column] =  y_low
                if delta_reward == 2 and row == 0:
                    matrix[row][column] = y_low
                if delta_reward == 0 and row == 2 :
                    matrix[row][column] = y_high
                if delta_reward == 2 and row == 2:
                    matrix[row][column] = y_high

    #Normalize distribution
    for column in range(0, C):
        factor = 0
        for row in range(0, card_leaf):
            factor += matrix[row][column]
        for row in range(0, card_leaf):
            matrix[row][column] /= factor

    return matrix


def parent_cpd(var, cardinality, mu, sigma):
    table = np.zeros(cardinality)
    factor = 0
    for x in range(0, cardinality):
        y = (x-mu)*(x-mu)/sigma/sigma
        prob = math.exp(-y)
        factor += prob
        table[x] = prob
    #normalize
    for t in range(0, len(table)):
        table[t] /= factor
    #print(table)
    return TabularCPD(variable=var, variable_card= cardinality, values=[table])


def latent_cpd(var, card_latent, card_parent, evid, modus, gamma):
    table =[]
    if(modus == 'fixed'):
        table = create_latent_distribution(card_latent, card_parent, gamma)
    if(modus == 'action'):
        table = create_action_distribution(card_latent, card_parent, gamma)
    if(modus == 'random'):
        cardinality = 1
        for  n in range(0, len(card_parent)):
            cardinality = cardinality*card_parent[n]
            n = n+1
            table = np.random.rand(card_latent, cardinality)
        for c in range(0,len(table[0])):
            factor = 0
            for r in range(0,len(table)):
                factor += table[r][c]
            for r in range(0, len(table)):
                table[r][c] /= factor
    evidence = []
    car_par = []
    for n in range(0,len(evid)):
        evidence.append(evid[n])
        car_par.append(card_parent[n])
    return TabularCPD(variable=var, variable_card= card_latent, values=table,
                      evidence=evidence,
                      evidence_card=car_par)



def leaf_cpd(var, card_leaf, card_latent, evid, modus, sigma):
    table =[]
    if(modus == 'azimuth'):
        table = create_leaf_distribution(card_leaf, card_latent, modus, sigma)
    if(modus == 'reward'):
        table = create_leaf_distribution(card_leaf, card_latent, modus, sigma)
    if(modus == 'random'):
        cardinality = 1
        for  n in range(0, len(card_latent)):
            cardinality = cardinality*card_latent[n]
            n = n+1
            table = np.random.rand(card_leaf, cardinality)
        for c in range(0,len(table[0])):
            factor = 0
            for r in range(0,len(table)):
                factor += table[r][c]
            for r in range(0, len(table)):
                table[r][c] /= factor
    evidence = []
    car_par = []
    for n in range(0,len(evid)):
        evidence.append(evid[n])
        car_par.append(card_latent[n])
    return TabularCPD(variable=var, variable_card= card_leaf, values=table,
                      evidence=evidence,
                      evidence_card=car_par)


class PeepoModel:
    RADIUS = 100

    def __init__(self, peepo_actor, Poopie, wall):
        self.peepo_actor = peepo_actor
        self.Poopies = Poopie
        self.wall = wall
        self.target = self.Poopies.get_poopies_obstacles()
        self.sectors = peepo_actor.sector
        self.R_previous = peepo_actor.R_previous
        self.R_now = peepo_actor.R_now
        self.models = self.create_networks()
        self.motor_output = {pg.K_LEFT: False, pg.K_RIGHT: False}
        self.target_sector = 3
        self.distance_now = 10000
        self.distance_previous = self.distance_now
        self.angle = peepo_actor.angle
        self.Reward = 0



    def create_networks(self):
        gamma = 1 # this controls how steep the discrimination will be between the classes (gamma << 1 low discrimination, gamma >> 1 : high discrimination
        sigma = 1 # this controls how steep the squezing of the action will be
        network = BayesianModel()
        ParentNodes = []
        ParentNodes.append("Azimuth_Belief")
        ParentNodes.append("Azimuth_Predicted")
        ParentNodes.append("Reward_Belief")
        ParentNodes.append("Reward_Predicted")
        count = 0
        while count < len(ParentNodes):
            network.add_node(ParentNodes[count])
            count = count+1

        LatentNodes = []
        LatentNodes.append("Delta_Azimuth")
        LatentNodes.append("Delta_Reward")
        LatentNodes.append("Action")
        count = 0
        while count < len(LatentNodes):
            network.add_node(LatentNodes[count])
            count = count + 1

        LeafNodes = []
        LeafNodes.append("Azimuth_next_cycle")
        LeafNodes.append("Reward_next_cycle")
        count = 0
        while count < len(LeafNodes):
            network.add_node(LeafNodes[count])
            count = count + 1

        network.add_edge(ParentNodes[0], LatentNodes[0])
        network.add_edge(ParentNodes[1], LatentNodes[0])
        network.add_edge(ParentNodes[2], LatentNodes[1])
        network.add_edge(ParentNodes[3], LatentNodes[1])
        network.add_edge(LatentNodes[0], LatentNodes[2])
        network.add_edge(LatentNodes[1], LatentNodes[2])

        network.add_edge(LatentNodes[0], LeafNodes[0])
        network.add_edge(LatentNodes[0], LeafNodes[1])
        network.add_edge(LatentNodes[1], LeafNodes[0])
        network.add_edge(LatentNodes[1], LeafNodes[1])
        network.add_edge(LatentNodes[2], LeafNodes[0])
        network.add_edge(LatentNodes[2], LeafNodes[1])

        cardinality_azimuth = 7
        cardinality_reward  = 3
        cardinality_action  = cardinality_azimuth#3
        CPD_Parents = []
        CPD_Parents.append(parent_cpd(ParentNodes[0],cardinality_azimuth, int(cardinality_azimuth/2), sigma/2))
        CPD_Parents.append(parent_cpd(ParentNodes[1],cardinality_azimuth, int(cardinality_azimuth/2), sigma/2))
        CPD_Parents.append(parent_cpd(ParentNodes[2],cardinality_reward,  int(cardinality_reward/2), sigma/2))
        CPD_Parents.append(parent_cpd(ParentNodes[3],cardinality_reward,  int(cardinality_reward/2), sigma/2))
        for n in range(0, len(CPD_Parents)):
            network.add_cpds(CPD_Parents[n])
        count = 0
        CPD_Latents = []
        CPD_Latents.append(latent_cpd(LatentNodes[0],cardinality_azimuth,[cardinality_azimuth,cardinality_azimuth],[ParentNodes[0],ParentNodes[1]], 'fixed', gamma))
        CPD_Latents.append(latent_cpd(LatentNodes[1],cardinality_reward,[cardinality_reward,cardinality_reward],[ParentNodes[2],ParentNodes[3]],'fixed', gamma))
        CPD_Latents.append(latent_cpd(LatentNodes[2],cardinality_action,[cardinality_azimuth, cardinality_reward], [LatentNodes[0], LatentNodes[1]], 'action', sigma))
        for n in range(0,len(CPD_Latents)):
            network.add_cpds(CPD_Latents[n])
        CPD_Leafs = []
        CPD_Leafs.append(leaf_cpd(LeafNodes[0],cardinality_azimuth,[cardinality_azimuth,cardinality_reward,cardinality_action ,],[LatentNodes[0], LatentNodes[1], LatentNodes[2]], 'azimuth', gamma))
        CPD_Leafs.append(leaf_cpd(LeafNodes[1],cardinality_reward ,[cardinality_azimuth,cardinality_reward,cardinality_action,],[LatentNodes[0], LatentNodes[1], LatentNodes[2]], 'reward', gamma))

        for n in range(0,len(CPD_Leafs)):
            network.add_cpds(CPD_Leafs[n])
        #draw_network(network)
        network.check_model()
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
        return {'main': GenerativeModel(SensoryInputVirtualPeepo(self), network)}

    def process(self):
        self.calculate_environment()
        for key in self.models:
            self.models[key].process()

    def calculate_environment(self):
        peepo_vec = vec(self.peepo_actor.rect.center)
        print("peepo_vec = ", peepo_vec)

        #first check if no collision with the wall occurred
        if peepo_vec[0] <= self.wall[0]:
            if math.cos(self.angle) == 0:
                self.angle = 0.009*math.pi/2
                self.angle = math.atan(math.sin(self.angle)/math.cos(-self.angle))
        if peepo_vec[1] <= self.wall[1]:
            if math.cos(self.angle) == 0:
                self.angle = 0.009 * math.pi / 2
                self.angle = math.atan(math.sin(-self.angle)/math.cos(self.angle))
        if peepo_vec[0] >= self.wall[2]:
            if math.cos(self.angle) == 0:
                self.angle = 0.009*math.pi/2
                self.angle = math.atan(math.sin(self.angle)/math.cos(-self.angle))
        if peepo_vec[1] >= self.wall[3]:
            if math.cos(self.angle) == 0:
                self.angle = 0.009 * math.pi / 2
                self.angle = math.atan(math.sin(-self.angle)/math.cos(self.angle))

        #Calculate distance and sector of the target
        for target in self.target:
            print("Target = ", self.Poopies.pos_x, self.Poopies.pos_y)
            #distance (is in fact the square of the distance but this doesn't matter here
            self.distance_now = (self.Poopies.pos_x - peepo_vec[0])*(self.Poopies.pos_x - peepo_vec[0])  + (self.Poopies.pos_y - peepo_vec[1])*(self.Poopies.pos_y - peepo_vec[1])
            print( "Distance now = ", math.sqrt(self.distance_now), " and previous distance = ",  math.sqrt(self.distance_previous))
            if self.distance_now - self.distance_previous < 0:
                self.Reward = 2
            if self.distance_now - self.distance_previous > 0:
                self.Reward = 0
            if self.distance_now - self.distance_previous == 0:
                self.Reward = 1
            self.distance_previous = self.distance_now


                #calculate in which quadrants the target is
            relative_angle_target = math.atan((self.Poopies.pos_y - peepo_vec[1])/(self.Poopies.pos_x - peepo_vec[0])) - self.angle
            for sec in range(0, len(self.sectors)):
                if relative_angle_target < self.sectors[1] or relative_angle_target < self.sectors[0]:
                    self.target_sector = 0
                if relative_angle_target > self.sectors[1]:
                    self.target_sector = 1
                    if relative_angle_target > self.sectors[2]:
                        self.target_sector = 2
                        if relative_angle_target > self.sectors[3]:
                            self.target_sector = 3
                            if relative_angle_target > self.sectors[4]:
                                self.target_sector = 4
                                if relative_angle_target > self.sectors[5]:
                                    self.target_sector = 5
                                    if relative_angle_target > self.sectors[6]:
                                        self.target_sector = 6
                                        if relative_angle_target > self.sectors[7]:
                                            self.target_sector = 6
            print("Target is sector ", self.target_sector)
            print("Reward ", self.Reward)


class SensoryInputVirtualPeepo(SensoryInput):
    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def action(self, node, prediction_error, prediction):
        # if prediction = [0.9, 0.1] (= moving) then move else stop
        if np.argmax(prediction) > 0:  # predicted stopping
            if 'left' in node:
                self.peepo.motor_output[pg.K_RIGHT] = False
            if 'right' in node:
                self.peepo.motor_output[pg.K_LEFT] = False
        else:  # predicted moving
            if 'left' in node:
                self.peepo.motor_output[pg.K_RIGHT] = True
            if 'right' in node:
                self.peepo.motor_output[pg.K_LEFT] = True

    def value(self, name):
        if 'Azimuth' in name:
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if 'next_cycle' in name:
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
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if 'next_cycle' in name:
                return np.array([0.1,0.8,0.1]) if self.peepo.target_sector == 3 else np.array([0.6,0.4,0.1])

        elif 'motor' in name:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            if 'left' in name:
                return np.array([0.9, 0.1]) if self.peepo.motor_output[pg.K_RIGHT] else np.array([0.1, 0.9])
            if 'right' in name:
                return np.array([0.9, 0.1]) if self.peepo.motor_output[pg.K_LEFT] else np.array([0.1, 0.9])
