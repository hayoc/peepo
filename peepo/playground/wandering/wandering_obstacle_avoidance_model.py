import math

import numpy as np
import pygame as pg
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.playground.util.vision import collision, end_line
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput

vec = pg.math.Vector2


def wandering_hypo_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.1, 0.9]])


def obstacle_hypo_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.1, 0.9]])


def single_hypo_cpd(var, evi):
    return TabularCPD(variable=var, variable_card=2, values=[[0.9, 0.1],
                                                             [0.1, 0.9]],
                      evidence=[evi],
                      evidence_card=[2])


def double_hypo_cpd(var, evi_1, evi_2):
    return TabularCPD(variable=var, variable_card=2, values=[[0.9, 0.9, 0.9, 0.1],
                                                             [0.1, 0.1, 0.1, 0.9]],
                      evidence=[evi_1, evi_2],
                      evidence_card=[2, 2])


class PeepoModel:
    RADIUS = 100

    def __init__(self, peepo_actor, actors):
        self.peepo_actor = peepo_actor
        self.actors = actors
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
        # network = BayesianModel([('brol_left', 'motor_left'),
        #                          ('brol_right', 'motor_right')])
        #
        # cpd_1 = single_hypo_cpd('motor_left', 'brol_left')
        # cpd_2 = single_hypo_cpd('motor_right', 'brol_right')
        # cpd_5 = obstacle_hypo_cpd('brol_left')
        # cpd_6 = obstacle_hypo_cpd('brol_right')
        # network.add_cpds(cpd_1, cpd_2, cpd_5, cpd_6)

        ################################

        network = BayesianModel([('wandering_left', 'motor_left'), ('wandering_right', 'motor_right'),
                                 ('obstacle_left', 'motor_left'), ('obstacle_right', 'motor_right'),
                                 ('obstacle_left', 'vision_1'), ('obstacle_left', 'vision_2'),
                                 ('obstacle_left', 'vision_3'),
                                 ('obstacle_right', 'vision_4'), ('obstacle_right', 'vision_5'),
                                 ('obstacle_right', 'vision_6')])

        cpd_1 = double_hypo_cpd('motor_left', 'wandering_left', 'obstacle_left')
        cpd_2 = double_hypo_cpd('motor_right', 'wandering_right', 'obstacle_right')

        cpd_3 = wandering_hypo_cpd('wandering_left')
        cpd_4 = wandering_hypo_cpd('wandering_right')
        cpd_5 = obstacle_hypo_cpd('obstacle_left')
        cpd_6 = obstacle_hypo_cpd('obstacle_right')

        cpd_7 = single_hypo_cpd('vision_1', 'obstacle_left')
        cpd_8 = single_hypo_cpd('vision_2', 'obstacle_left')
        cpd_9 = single_hypo_cpd('vision_3', 'obstacle_left')
        cpd_10 = single_hypo_cpd('vision_4', 'obstacle_right')
        cpd_11 = single_hypo_cpd('vision_5', 'obstacle_right')
        cpd_12 = single_hypo_cpd('vision_6', 'obstacle_right')

        network.add_cpds(cpd_1, cpd_2, cpd_3, cpd_4, cpd_5, cpd_6, cpd_7,
                         cpd_8, cpd_9, cpd_10, cpd_11, cpd_12)
        network.check_model()

        return {'main': GenerativeModel(SensoryInputVirtualPeepo(self), network)}

    def process(self):
        self.calculate_obstacles()
        for key in self.models:
            self.models[key].process()

    def calculate_obstacles(self):
        for key in self.obstacle_input:
            self.obstacle_input[key] = False

        for actor in self.actors:
            peepo_vec = vec(self.peepo_actor.rect.center)
            collided = collision(actor.rect, peepo_vec, self.peepo_actor.edge_left,
                                 self.peepo_actor.edge_right, PeepoModel.RADIUS)
            if collided:
                if 'wall' in actor.id:
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

    def action(self, node, prediction):
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
