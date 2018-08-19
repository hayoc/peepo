import math

import pygame as pg
import numpy as np

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.playground.vision import collision
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput

vec = pg.math.Vector2


def wandering_hypo_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.1, 0.9]])


def obstacle_hypo_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.1, 0.9]])


def single_hypo_cpd(var, evi):
    return TabularCPD(variable=var, variable_card=2, values=[[0.9, 0.9],
                                                             [0.1, 0.1]],
                      evidence=[evi],
                      evidence_card=[2])


def double_hypo_cpd(var, evi_1, evi_2):
    return TabularCPD(variable=var, variable_card=2, values=[[0.9, 0.9, 0.9, 0.1],
                                                             [0.1, 0.1, 0.1, 0.9]],
                      evidence=[evi_1, evi_2],
                      evidence_card=[2, 2])

class PeepoModel:
    DISTANCE = 75
    SIZE = (40, 40)
    RADIUS = 100

    def __init__(self, peepo_actor, actors):
        self.peepo_actor = peepo_actor
        self.actors = actors
        self.models = {'main': None}
        self.create_networks()
        self.motor_output = {pg.K_LEFT: False,
                             pg.K_RIGHT: False}
        self.obstacle_input = {'left': False,
                               'right': False,
                               'up': False,
                               'down': False}

    def create_networks(self):
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

        self.models['main'] = GenerativeModel(SensoryInputVirtualPeepo(self), network)

    def process(self):
        self.calculate_obstacles()
        for key in self.models:
            self.models[key].process()

    def calculate_obstacles(self):
        collided1, collided2, collided3, collided4, collided5, collided6 = False
        for actor in self.actors:
            collided = collision(actor.rect, vec(self.peepo_actor.rect.center), self.peepo_actor.edge_left,
                                 self.peepo_actor.edge_right, PeepoModel.RADIUS)
            if collided:
                pass



        for actor in self.actors:
            self.obstacle_input['left'] = actor.rect.x < self.rect.x and math.hypot(actor.rect.x - self.rect.x,
                                                                                    actor.rect.y - self.rect.y) < float(
                PeepoModel.DISTANCE)
            self.obstacle_input['right'] = actor.rect.x > self.rect.x and math.hypot(actor.rect.x - self.rect.x,
                                                                                     actor.rect.y - self.rect.y) < float(
                PeepoModel.DISTANCE)
            self.obstacle_input['up'] = actor.rect.y < self.rect.y and math.hypot(actor.rect.x - self.rect.x,
                                                                                  actor.rect.y - self.rect.y) < float(
                PeepoModel.DISTANCE)
            self.obstacle_input['down'] = actor.rect.y > self.rect.y and math.hypot(actor.rect.x - self.rect.x,
                                                                                    actor.rect.y - self.rect.y) < float(
                PeepoModel.DISTANCE)


class SensoryInputVirtualPeepo(SensoryInput):
    def __init__(self, peepo):
        super().__init__()
        self.peepo = peepo

    def action(self, node, prediction_error, prediction):
        # if prediction = [0.1, 0.9] (= moving) then move else stop
        if np.argmax(prediction) > 0:  # predicted moving
            if 'left' in node:
                self.peepo.motor_output[pg.K_RIGHT] = True
            if 'right' in node:
                self.peepo.motor_output[pg.K_LEFT] = True
        else:  # predicted stopped
            if 'left' in node:
                self.peepo.motor_output[pg.K_RIGHT] = False
            if 'right' in node:
                self.peepo.motor_output[pg.K_LEFT] = False

    def value(self, name):
        if 'vision' in name:
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if '1' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['1'] else np.array([0.9, 0.1]))
            if '2' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['2'] else np.array([0.9, 0.1]))
            if '3' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['3'] else np.array([0.9, 0.1]))
            if '4' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['4'] else np.array([0.9, 0.1]))
            if '5' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['5'] else np.array([0.9, 0.1]))
            if '6' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['6'] else np.array([0.9, 0.1]))
        elif 'motor' in name:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            if 'left' in name:
                return np.array([0.1, 0.9]) if self.peepo.motor_output[pg.K_RIGHT] else np.array([0.9, 0.1])
            if 'right' in name:
                return np.array([0.1, 0.9]) if self.peepo.motor_output[pg.K_LEFT] else np.array([0.9, 0.1])
