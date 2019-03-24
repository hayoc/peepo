import math
import random

import numpy as np
import pygame as pg
from pomegranate.distributions.ConditionalProbabilityTable import ConditionalProbabilityTable
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution

from peepo.playground.wandering.vision import collision, end_line
from peepo.pp.generative_model import GenerativeModel
from peepo.pp.peepo import Peepo
from peepo.pp.peepo_network import PeepoNetwork

vec = pg.math.Vector2


def single_hypo_cpd(evi):
    return ConditionalProbabilityTable(
        [[1, 1, 0.9],
         [1, 0, 0.1],
         [0, 1, 0.1],
         [0, 0, 0.9]], [evi])


def double_hypo_cpd(evi_1, evi_2):
    return ConditionalProbabilityTable(
        [[1, 1, 1, 0.9],
         [1, 1, 0, 0.1],
         [1, 0, 1, 0.9],
         [1, 0, 0, 0.1],
         [0, 1, 1, 0.9],
         [0, 1, 0, 0.1],
         [0, 0, 1, 0.1],
         [0, 0, 0, 0.9]], [evi_1, evi_2])


def change_distribution(old, new_values):
    result = []
    for x, old_value in enumerate(old):
        result.append((old_value[0], new_values[x]))
    return dict(result)


class WanderingPeepo(Peepo):
    RADIUS = 100

    WAN_LEFT = 'RON_wandering_left'
    WAN_RIGHT = 'RON_wandering_right'
    OBS_LEFT = 'RON_obstacle_left'
    OBS_RIGHT = 'RON_obstacle_right'
    MOT_LEFT = 'LEN_motor_left'
    MOT_RIGHT = 'LEN_motor_right'
    VIS_1 = 'LEN_vision_1'
    VIS_2 = 'LEN_vision_2'
    VIS_3 = 'LEN_vision_3'
    VIS_4 = 'LEN_vision_4'
    VIS_5 = 'LEN_vision_5'
    VIS_6 = 'LEN_vision_6'

    def __init__(self, peepo_actor, actors):
        super().__init__(self.create_model())

        self.peepo_actor = peepo_actor
        self.actors = actors

        self.motor_output = {pg.K_LEFT: False,
                             pg.K_RIGHT: False}
        self.obstacle_input = {'1': False,
                               '2': False,
                               '3': False,
                               '4': False,
                               '5': False,
                               '6': False}
        self.wander_left_chance = 0
        self.wander_right_chance = 0
        self.wandering_left = False
        self.wandering_right = False

        self.genmodel = GenerativeModel(self)

    def action(self, node, prediction):
        # prediction [0.9, 0.1] = STOP
        # prediction [0.1, 0.9] = START
        if np.argmax(prediction) == 0:
            if 'left' in node:
                self.motor_output[pg.K_LEFT] = False
            if 'right' in node:
                self.motor_output[pg.K_RIGHT] = False
        else:
            if 'left' in node:
                self.motor_output[pg.K_LEFT] = True
            if 'right' in node:
                self.motor_output[pg.K_RIGHT] = True

    def observation(self, name):
        if 'vision' in name:
            # prediction [0.9, 0.1] = NO OBSTACLE
            # prediction [0.1, 0.9] = OBSTACLE
            if '1' in name:
                return np.array([0.1, 0.9]) if self.obstacle_input['1'] else np.array([0.9, 0.1])
            if '2' in name:
                return np.array([0.1, 0.9]) if self.obstacle_input['2'] else np.array([0.9, 0.1])
            if '3' in name:
                return np.array([0.1, 0.9]) if self.obstacle_input['3'] else np.array([0.9, 0.1])
            if '4' in name:
                return np.array([0.1, 0.9]) if self.obstacle_input['4'] else np.array([0.9, 0.1])
            if '5' in name:
                return np.array([0.1, 0.9]) if self.obstacle_input['5'] else np.array([0.9, 0.1])
            if '6' in name:
                return np.array([0.1, 0.9]) if self.obstacle_input['6'] else np.array([0.9, 0.1])
        elif 'motor' in name:
            # prediction [0.9, 0.1] = STOPPED
            # prediction [0.1, 0.9] = MOVING
            if 'left' in name:
                return np.array([0.1, 0.9]) if self.motor_output[pg.K_LEFT] else np.array([0.9, 0.1])
            if 'right' in name:
                return np.array([0.1, 0.9]) if self.motor_output[pg.K_RIGHT] else np.array([0.9, 0.1])

    def update(self):
        network = self.genmodel.bayesian_network

        # [0.9, 0.1] = OFF
        # [0.1, 0.9] = ON
        if self.wandering_left:
            state = network.states[self.genmodel.get_node_index(self.WAN_LEFT)]
            state.distribution = DiscreteDistribution(change_distribution(state.distribution.items(), [0.9, 0.1]))
            self.wander_left_chance = 0
            self.wandering_left = False
        else:
            self.wander_left_chance += 0.1
            if random.randint(0, 100) <= self.wander_left_chance:
                state = network.states[self.genmodel.get_node_index(self.WAN_LEFT)]
                state.distribution = DiscreteDistribution(change_distribution(state.distribution.items(), [0.1, 0.9]))
                self.wandering_left = True

        if self.wandering_right:
            state = network.states[self.genmodel.get_node_index(self.WAN_RIGHT)]
            state.distribution = DiscreteDistribution(change_distribution(state.distribution.items(), [0.9, 0.1]))
            self.wander_right_chance = 0
            self.wandering_right = False
        else:
            self.wander_right_chance += 0.1
            if random.randint(0, 100) <= self.wander_right_chance:
                state = network.states[self.genmodel.get_node_index(self.WAN_RIGHT)]
                state.distribution = DiscreteDistribution(change_distribution(state.distribution.items(), [0.1, 0.9]))
                self.wandering_right = True

    def create_model(self):
        pp_network = PeepoNetwork(
            ron_nodes=[
                {'name': self.WAN_LEFT, 'card': 2},
                {'name': self.WAN_RIGHT, 'card': 2},
                {'name': self.OBS_LEFT, 'card': 2},
                {'name': self.OBS_RIGHT, 'card': 2}
            ],
            ext_nodes=[
                {'name': self.VIS_1, 'card': 2},
                {'name': self.VIS_2, 'card': 2},
                {'name': self.VIS_3, 'card': 2},
                {'name': self.VIS_4, 'card': 2},
                {'name': self.VIS_5, 'card': 2},
                {'name': self.VIS_6, 'card': 2}
            ],
            pro_nodes=[
                {'name': self.MOT_LEFT, 'card': 2},
                {'name': self.MOT_RIGHT, 'card': 2}
            ],
            edges=[
                (self.WAN_LEFT, self.MOT_LEFT),
                (self.WAN_RIGHT, self.MOT_RIGHT),
                (self.OBS_LEFT, self.MOT_RIGHT),
                (self.OBS_RIGHT, self.MOT_LEFT),
                (self.OBS_LEFT, self.VIS_1),
                (self.OBS_LEFT, self.VIS_2),
                (self.OBS_LEFT, self.VIS_3),
                (self.OBS_RIGHT, self.VIS_4),
                (self.OBS_RIGHT, self.VIS_5),
                (self.OBS_RIGHT, self.VIS_6),
            ],
            cpds={
                self.WAN_LEFT: [0.9, 0.1],
                self.WAN_RIGHT: [0.9, 0.1],
                self.OBS_LEFT: [0.9, 0.1],
                self.OBS_RIGHT: [0.9, 0.1],
                self.VIS_1: [[0.9, 0.1],
                             [0.1, 0.9]],
                self.VIS_2: [[0.9, 0.1],
                             [0.1, 0.9]],
                self.VIS_3: [[0.9, 0.1],
                             [0.1, 0.9]],
                self.VIS_4: [[0.9, 0.1],
                             [0.1, 0.9]],
                self.VIS_5: [[0.9, 0.1],
                             [0.1, 0.9]],
                self.VIS_6: [[0.9, 0.1],
                             [0.1, 0.9]],
                self.MOT_LEFT: [[0.9, 0.1, 0.1, 0.1],
                                [0.1, 0.9, 0.9, 0.9]],
                self.MOT_RIGHT: [[0.9, 0.1, 0.1, 0.1],
                                 [0.1, 0.9, 0.9, 0.9]],
            })
        pp_network.assemble()

        return pp_network

    def process(self):
        self.calculate_obstacles()
        self.genmodel.process()

    def calculate_obstacles(self):
        for key in self.obstacle_input:
            self.obstacle_input[key] = False

        for actor in self.actors:
            peepo_vec = vec(self.peepo_actor.rect.center)
            collided = collision(actor.rect, peepo_vec, self.peepo_actor.edge_left,
                                 self.peepo_actor.edge_right, WanderingPeepo.RADIUS)
            if collided:
                if 'wall' in actor.id:
                    edge = end_line(WanderingPeepo.RADIUS, self.peepo_actor.rotation, self.peepo_actor.rect.center)
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
                    edge1 = end_line(WanderingPeepo.RADIUS, self.peepo_actor.rotation - 30,
                                     self.peepo_actor.rect.center)
                    edge2 = end_line(WanderingPeepo.RADIUS, self.peepo_actor.rotation - 20,
                                     self.peepo_actor.rect.center)
                    edge3 = end_line(WanderingPeepo.RADIUS, self.peepo_actor.rotation - 10,
                                     self.peepo_actor.rect.center)
                    edge4 = end_line(WanderingPeepo.RADIUS, self.peepo_actor.rotation, self.peepo_actor.rect.center)
                    edge5 = end_line(WanderingPeepo.RADIUS, self.peepo_actor.rotation + 10,
                                     self.peepo_actor.rect.center)
                    edge6 = end_line(WanderingPeepo.RADIUS, self.peepo_actor.rotation + 20,
                                     self.peepo_actor.rect.center)
                    edge7 = end_line(WanderingPeepo.RADIUS, self.peepo_actor.rotation + 30,
                                     self.peepo_actor.rect.center)

                    self.obstacle_input['1'] = collision(actor.rect, peepo_vec, edge1, edge2, WanderingPeepo.RADIUS)
                    self.obstacle_input['2'] = collision(actor.rect, peepo_vec, edge2, edge3, WanderingPeepo.RADIUS)
                    self.obstacle_input['3'] = collision(actor.rect, peepo_vec, edge3, edge4, WanderingPeepo.RADIUS)
                    self.obstacle_input['4'] = collision(actor.rect, peepo_vec, edge4, edge5, WanderingPeepo.RADIUS)
                    self.obstacle_input['5'] = collision(actor.rect, peepo_vec, edge5, edge6, WanderingPeepo.RADIUS)
                    self.obstacle_input['6'] = collision(actor.rect, peepo_vec, edge6, edge7, WanderingPeepo.RADIUS)
