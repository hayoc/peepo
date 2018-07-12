import math

import pygame as pg
import numpy as np

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput


def create_network():
    network = BayesianModel([('hypo-left', 'obstacle-left'), ('hypo-left', 'motor-left'),
                             ('hypo-right', 'obstacle-right'), ('hypo-right', 'motor-right'),
                             ('hypo-up', 'obstacle-up'), ('hypo-up', 'motor-up'),
                             ('hypo-down', 'obstacle-down'), ('hypo-down', 'motor-down')])

    cpd_left = hypo_cpd('hypo-left')
    cpd_right = hypo_cpd('hypo-right')
    cpd_up = hypo_cpd('hypo-up')
    cpd_down = hypo_cpd('hypo-down')

    cpd_left_obs = child_cpd('obstacle-left', 'hypo-left')
    cpd_right_obs = child_cpd('obstacle-right', 'hypo-right')
    cpd_up_obs = child_cpd('obstacle-up', 'hypo-up')
    cpd_down_obs = child_cpd('obstacle-down', 'hypo-down')

    cpd_left_motor = child_cpd('motor-left', 'hypo-left')
    cpd_right_motor = child_cpd('motor-right', 'hypo-right')
    cpd_up_motor = child_cpd('motor-up', 'hypo-up')
    cpd_down_motor = child_cpd('motor-down', 'hypo-down')

    network.add_cpds(cpd_left, cpd_right, cpd_up, cpd_down, cpd_left_obs, cpd_right_obs, cpd_up_obs, cpd_down_obs,
                     cpd_left_motor, cpd_right_motor, cpd_up_motor, cpd_down_motor)
    network.check_model()
    return network


def hypo_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.7, 0.3]])


def child_cpd(var, evi):
    return TabularCPD(variable=var, variable_card=2, values=[[0.9, 0.1],
                                                             [0.1, 0.9]],
                      evidence=[evi],
                      evidence_card=[2])


class PeepoModel:
    DISTANCE = 75
    SIZE = (40, 40)

    def __init__(self, actors):
        self.rect = pg.Rect((0, 0), PeepoModel.SIZE)
        self.actors = actors
        self.model = GenerativeModel(SensoryInputVirtualPeepo(self), create_network())
        self.motor_output = {pg.K_LEFT: False,
                             pg.K_RIGHT: False,
                             pg.K_UP: False,
                             pg.K_DOWN: False}
        self.obstacle_input = {'left': False,
                               'right': False,
                               'up': False,
                               'down': False}

    def process(self):
        self.calculate_obstacles()
        self.model.process()

    def calculate_obstacles(self):
        for actor in self.actors:
            self.obstacle_input['left'] = actor.rect.x > self.rect.x and math.hypot(actor.rect.x - self.rect.x, actor.rect.y - self.rect.y) < float(PeepoModel.DISTANCE)
            self.obstacle_input['right'] = actor.rect.x < self.rect.x and math.hypot(actor.rect.x - self.rect.x, actor.rect.y - self.rect.y) < float(PeepoModel.DISTANCE)
            self.obstacle_input['up'] = actor.rect.y > self.rect.y and math.hypot(actor.rect.x - self.rect.x, actor.rect.y - self.rect.y) < float(PeepoModel.DISTANCE)
            self.obstacle_input['down'] = actor.rect.y < self.rect.y and math.hypot(actor.rect.x - self.rect.x, actor.rect.y - self.rect.y) < float(PeepoModel.DISTANCE)


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
            if 'up' in node:
                self.peepo.motor_output[pg.K_DOWN] = True
            if 'down' in node:
                self.peepo.motor_output[pg.K_UP] = True
        else:  # predicted stopped
            if 'left' in node:
                self.peepo.motor_output[pg.K_RIGHT] = False
            if 'right' in node:
                self.peepo.motor_output[pg.K_LEFT] = False
            if 'up' in node:
                self.peepo.motor_output[pg.K_DOWN] = False
            if 'down' in node:
                self.peepo.motor_output[pg.K_UP] = False

    def value(self, name):
        if 'obstacle' in name:
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            if 'left' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['left'] else np.array([0.9, 0.1]))
            if 'right' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['right'] else np.array([0.9, 0.1]))
            if 'up' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['up'] else np.array([0.9, 0.1]))
            if 'down' in name:
                return np.array([0.1, 0.9] if self.peepo.obstacle_input['down'] else np.array([0.9, 0.1]))
        else:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            if 'left' in name:
                return np.array([0.1, 0.9]) if self.peepo.motor_output[pg.K_RIGHT] else np.array([0.9, 0.1])
            if 'right' in name:
                return np.array([0.1, 0.9]) if self.peepo.motor_output[pg.K_LEFT] else np.array([0.9, 0.1])
            if 'up' in name:
                return np.array([0.1, 0.9]) if self.peepo.motor_output[pg.K_DOWN] else np.array([0.9, 0.1])
            if 'down' in name:
                return np.array([0.1, 0.9]) if self.peepo.motor_output[pg.K_UP] else np.array([0.9, 0.1])
