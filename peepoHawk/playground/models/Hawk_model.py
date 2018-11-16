#version 15/11/2018


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
def normalize_angle(angle):
    if angle >= 2 * math.pi:
        angle -= 2 * math.pi
    if angle <= -2 * math.pi:
        angle += 2 * math.pi
    return angle

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

def create_latent_reward_distribution():
    #CREATES : aCPD a distribution depending on the "distance" of the latent variable index to the indexes of the parents
    #the distance is the inverse of an exponentional of the sum of the distances coorected with a factor (set to 1 for the moment)
    #cardinality of the latent must be the same as the cardinality of the parents
    card_parent = [3,3]
    card_latent = 3
    C = np.prod(card_parent)
    matrix = [[0.1,0.2,0.8,0.1,0.1,0.4,0.1,0.1,0.8],
              [0.1,0.4,0.1,0.1,0.1,0.4,0.1,0.1,0.1],
              [0.8,0.4,0.1,0.8,0.8,0.2,0.8,0.8,0.1]]
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

def updated_cpd(cardinality, index):
    v = np.zeros(cardinality)
    for i in range (0, cardinality):
        v[i] =0.1/(cardinality-1)
    v[index] = 0.9
    return v

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

def reward_cpd(var, card_latent, card_parent, evid, modus, gamma):
    table = create_latent_reward_distribution()
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
        sigma = 1 # this controls how steep the squezing of the action will be

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

        LeafNodes = []
        LeafNodes.append("Azimuth_next_cycle")
        LeafNodes.append("Reward_next_cycle")
        count = 0
        while count < len(LeafNodes):
            self.network.add_node(LeafNodes[count])
            count = count + 1

        self.network.add_edge(ParentNodes[0], LatentNodes[0])
        self.network.add_edge(ParentNodes[1], LatentNodes[0])
        self.network.add_edge(ParentNodes[2], LatentNodes[1])
        self.network.add_edge(ParentNodes[3], LatentNodes[1])
        self.network.add_edge(LatentNodes[0], LatentNodes[2])
        self.network.add_edge(LatentNodes[1], LatentNodes[2])

        self.network.add_edge(LatentNodes[0], LeafNodes[0])
        self.network.add_edge(LatentNodes[0], LeafNodes[1])
        self.network.add_edge(LatentNodes[1], LeafNodes[0])
        self.network.add_edge(LatentNodes[1], LeafNodes[1])
        self.network.add_edge(LatentNodes[2], LeafNodes[0])
        self.network.add_edge(LatentNodes[2], LeafNodes[1])

        cardinality_azimuth = 7
        cardinality_reward  = 3
        cardinality_action  = cardinality_azimuth#3
        CPD_Parents = []
        CPD_Parents.append(parent_cpd(ParentNodes[0],cardinality_azimuth, int(cardinality_azimuth/2), sigma/2))
        CPD_Parents.append(parent_cpd(ParentNodes[1],cardinality_azimuth, int(cardinality_azimuth/2), sigma/2))
        CPD_Parents.append(parent_cpd(ParentNodes[2],cardinality_reward,  int(cardinality_reward/2), sigma/2))
        CPD_Parents.append(parent_cpd(ParentNodes[3],cardinality_reward,  int(cardinality_reward/2), sigma/2))
        for n in range(0, len(CPD_Parents)):
            self.network.add_cpds(CPD_Parents[n])
        count = 0
        CPD_Latents = []
        CPD_Latents.append(latent_cpd(LatentNodes[0],cardinality_azimuth,[cardinality_azimuth,cardinality_azimuth],[ParentNodes[0],ParentNodes[1]], 'fixed', gamma))
        CPD_Latents.append(reward_cpd(LatentNodes[1],cardinality_reward,[cardinality_reward,cardinality_reward],[ParentNodes[2],ParentNodes[3]],'fixed', gamma))
        CPD_Latents.append(latent_cpd(LatentNodes[2],cardinality_action,[cardinality_azimuth, cardinality_reward], [LatentNodes[0], LatentNodes[1]], 'action', sigma))
        for n in range(0,len(CPD_Latents)):
            self.network.add_cpds(CPD_Latents[n])
        CPD_Leafs = []
        CPD_Leafs.append(leaf_cpd(LeafNodes[0],cardinality_azimuth,[cardinality_azimuth,cardinality_reward,cardinality_action ,],[LatentNodes[0], LatentNodes[1], LatentNodes[2]], 'azimuth', gamma))
        CPD_Leafs.append(leaf_cpd(LeafNodes[1],cardinality_reward ,[cardinality_azimuth,cardinality_reward,cardinality_action,],[LatentNodes[0], LatentNodes[1], LatentNodes[2]], 'reward', gamma))

        for n in range(0,len(CPD_Leafs)):
            self.network.add_cpds(CPD_Leafs[n])
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

    def process(self):
        self.calculate_environment()
        for key in self.models:
            er, self.next_azimuth, self.next_reward = self.models[key].process()
            #print("Next azimuth ----> ", self.next_azimuth)
            #print("Next reward  ----> ", self.next_reward)

            azimuth = self.network.get_cpds('Action').values
            azimuth = azimuth.reshape((azimuth.shape[0],-1),order = 'F')
            pos_azimuth = np.unravel_index(np.argmax(azimuth), np.array(azimuth).shape)[0]
            #print("Adapted sector = ",pos_azimuth)
            self.peepo_actor.angle = normalize_angle(self.peepo_actor.angle + self.peepo_actor.sector[pos_azimuth])
            print("new angle = ", self.peepo_actor.angle*180/math.pi," degrees")
            self.peepo_actor.update_sectors()
            self.network.get_cpds('Azimuth_Belief').values = updated_cpd(7, self.observed_azimuth)
            self.network.get_cpds('Reward_Belief').values = updated_cpd(3, self.observed_reward)
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
                #calculate in which quadrants the target is
            absolute_angle_target = math.atan((self.Poopies.pos_y - peepo_vec[1])/(self.Poopies.pos_x - peepo_vec[0]))
            #relative_angle_target = math.atan(self.Poopies.pos_y/self.Poopies.pos_x)  - math.atan(peepo_vec[1]/peepo_vec[0]) - self.peepo_actor.angle
            if absolute_angle_target < self.peepo_actor.sector[1] or absolute_angle_target <= self.peepo_actor.sector[0]:
                self.target_sector = 0
                #print("absolute angle : ", absolute_angle_target, " and choosen sector is 0 or 1 with sector angle :" , self.peepo_actor.sector[1])
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
            print("Target is sector ", self.target_sector)
            print("Reward ", self.Reward)
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

