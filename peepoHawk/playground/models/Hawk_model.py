import math
import random
import numpy as np
from numpy import array
import pygame as pg
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepoHawk.playground.util.vision import collision, end_line
from peepoHawk.predictive_processing.v3.generative_model import GenerativeModel
from peepoHawk.predictive_processing.v3.sensory_input import SensoryInput

from peepoHawk.visualize.graph import draw_network

vec = pg.math.Vector2


def get_index_matrix(cardinality):#creates a matrix of the contingency table used  in create latent distribution with a fixed distibution
    #cardinality = [2,2,3]# size vector
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
    #CREATES : a matrix with a distribution depending on the "distance" of the latent variable index to the indexes of the parents
    #the distance is the inverse of an exponentional of the sum of the distances coorected with a factor (set to 1 for the moment)
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
    #CREATES : a matrix with a distribution depending on the "distance" of the latent variable index to the indexes of the parents
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


def parent_cpd(var, cardinality):
    table =[]
    factor = 0
    for t in range(0, cardinality):
        prob = random.randint(0, 100)
        factor += prob
        table.append(prob)
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


class PeepoModel:
    RADIUS = 100

    def __init__(self, peepo_actor, target):
        self.peepo_actor = peepo_actor
        self.target = target
        self.models = self.create_networks()
        self.motor_output = {pg.K_LEFT: False,
                             pg.K_RIGHT: False}
        self.obstacle_input = {'1': False,
                               '2': False,
                               '3': False,
                               '4': False,
                               '5': False,
                               '6': False}



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
        LeafNodes.append("Azimuth_observed")
        LeafNodes.append("Reward_observed")
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
        network.add_edge(LatentNodes[2], LeafNodes[0])
        network.add_edge(LatentNodes[2], LeafNodes[1])
        cardinality_azimuth = 5
        cardinality_reward  = 3
        cardinality_action  = 5
        CPD_Parents = []
        CPD_Parents.append(parent_cpd(ParentNodes[0],cardinality_azimuth))
        CPD_Parents.append(parent_cpd(ParentNodes[1],cardinality_azimuth))
        CPD_Parents.append(parent_cpd(ParentNodes[2],cardinality_reward))
        CPD_Parents.append(parent_cpd(ParentNodes[3],cardinality_action))
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
        CPD_Leafs.append(latent_cpd(LeafNodes[0],cardinality_azimuth,[cardinality_azimuth],[LatentNodes[2]], 'fixed', gamma))
        CPD_Leafs.append(latent_cpd(LeafNodes[1],cardinality_azimuth,[cardinality_azimuth],[LatentNodes[2]], 'fixed', gamma))
        for n in range(0,len(CPD_Leafs)):
            network.add_cpds(CPD_Leafs[n])
        draw_network(network)
        network.check_model()
        for n in range(0,len(CPD_Parents)):
            print("Parents :")
            print(CPD_Parents[n])
        for n in range(0,len(CPD_Latents)):
            print("Latents :")
            print(CPD_Latents[n])
        for n in range(0,len(CPD_Leafs)):
            print("Leafs :")
            print(CPD_Leafs[n])
        count = 0
        while count == 0:
            count = 0
        return {'main': GenerativeModel(SensoryInputVirtualPeepo(self), network)}

    def process(self):
        self.calculate_target()
        for key in self.models:
            self.models[key].process()

    def calculate_target(self):
        for key in self.obstacle_input:
            self.obstacle_input[key] = False

        for target in self.target:
            peepo_vec = vec(self.peepo_actor.rect.center)
            azimuth, reward, collided = observation(target.rect, peepo_vec, self.peepo_actor.edge_left,
                                 self.peepo_actor.edge_right, PeepoModel.RADIUS)
            if collided:
                if 'wall' in target.id:
                    edge = end_line(PeepoModel.RADIUS, self.peepo_actor.rotation, self.peepo_actor.rect.center)
                    if 'left' in actor.id:
                        wall_vec = vec((5, self.peepo_actor.rect.y))
                        deg = math.degrees(
                            math.atan2(wall_vec.y - edge.y, wall_vec.x - edge.x)) + self.peepo_actor.rotation
                        if deg < 0:
                            self.obstacle_input['6'] = True
                        else:
                            self.obstacle_input['1'] = True
                    elif 'right' in actor.id:
                        wall_vec = vec((1598, self.peepo_actor.rect.y))
                        deg = math.degrees(
                            math.atan2(wall_vec.y - edge.y, wall_vec.x - edge.x)) + self.peepo_actor.rotation
                        if deg < 0:
                            self.obstacle_input['1'] = True
                        else:
                            self.obstacle_input['6'] = True
                    elif 'up' in actor.id:
                        wall_vec = vec((5, self.peepo_actor.rect.y))
                        deg = math.degrees(
                            math.atan2(wall_vec.y - edge.y, wall_vec.x - edge.x)) + self.peepo_actor.rotation
                        if deg < 90:
                            self.obstacle_input['6'] = True
                        else:
                            self.obstacle_input['1'] = True
                    else:
                        wall_vec = vec((5, self.peepo_actor.rect.y))
                        deg = math.degrees(
                            math.atan2(wall_vec.y - edge.y, wall_vec.x - edge.x)) + self.peepo_actor.rotation
                        if deg < -90:
                            self.obstacle_input['6'] = True
                        else:
                            self.obstacle_input['1'] = True

                else:
                    edge1 = end_line(PeepoModel.RADIUS, self.peepo_actor.rotation - 30, self.peepo_actor.rect.center)
                    edge2 = end_line(PeepoModel.RADIUS, self.peepo_actor.rotation - 20, self.peepo_actor.rect.center)
                    edge3 = end_line(PeepoModel.RADIUS, self.peepo_actor.rotation - 10, self.peepo_actor.rect.center)
                    edge4 = end_line(PeepoModel.RADIUS, self.peepo_actor.rotation, self.peepo_actor.rect.center)
                    edge5 = end_line(PeepoModel.RADIUS, self.peepo_actor.rotation + 10, self.peepo_actor.rect.center)
                    edge6 = end_line(PeepoModel.RADIUS, self.peepo_actor.rotation + 20, self.peepo_actor.rect.center)
                    edge7 = end_line(PeepoModel.RADIUS, self.peepo_actor.rotation + 30, self.peepo_actor.rect.center)

                    self.obstacle_input['1'] = collision(actor.rect, peepo_vec, edge1, edge2, PeepoModel.RADIUS)
                    self.obstacle_input['2'] = collision(actor.rect, peepo_vec, edge2, edge3, PeepoModel.RADIUS)
                    self.obstacle_input['3'] = collision(actor.rect, peepo_vec, edge3, edge4, PeepoModel.RADIUS)
                    self.obstacle_input['4'] = collision(actor.rect, peepo_vec, edge4, edge5, PeepoModel.RADIUS)
                    self.obstacle_input['5'] = collision(actor.rect, peepo_vec, edge5, edge6, PeepoModel.RADIUS)
                    self.obstacle_input['6'] = collision(actor.rect, peepo_vec, edge6, edge7, PeepoModel.RADIUS)


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
        if 'vision' in name:
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if '1' in name:
                return np.array([0.9, 0.1]) if self.peepo.obstacle_input['1'] else np.array([0.1, 0.9])
            if '2' in name:
                return np.array([0.9, 0.1]) if self.peepo.obstacle_input['2'] else np.array([0.1, 0.9])
            if '3' in name:
                return np.array([0.9, 0.1]) if self.peepo.obstacle_input['3'] else np.array([0.1, 0.9])
            if '4' in name:
                return np.array([0.9, 0.1]) if self.peepo.obstacle_input['4'] else np.array([0.1, 0.9])
            if '5' in name:
                return np.array([0.9, 0.1]) if self.peepo.obstacle_input['5'] else np.array([0.1, 0.9])
            if '6' in name:
                return np.array([0.9, 0.1]) if self.peepo.obstacle_input['6'] else np.array([0.1, 0.9])
        elif 'motor' in name:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            if 'left' in name:
                return np.array([0.9, 0.1]) if self.peepo.motor_output[pg.K_RIGHT] else np.array([0.1, 0.9])
            if 'right' in name:
                return np.array([0.9, 0.1]) if self.peepo.motor_output[pg.K_LEFT] else np.array([0.1, 0.9])
