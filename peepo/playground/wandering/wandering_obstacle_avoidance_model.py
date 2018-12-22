import math

import pygame as pg
from pomegranate.BayesianNetwork import BayesianNetwork
from pomegranate.base import State
from pomegranate.distributions.ConditionalProbabilityTable import ConditionalProbabilityTable
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution

from peepo.playground.util.vision import collision, end_line
from peepo.playground.wandering.wandering_obstacle_avoidance_peepo import SensoryInputVirtualPeepo
from peepo.predictive_processing.v3.generative_model import GenerativeModel

vec = pg.math.Vector2


def single_hypo_cpd(evi):
    return ConditionalProbabilityTable(
        [['1', '1', 0.9],
         ['1', '0', 0.1],
         ['0', '1', 0.1],
         ['0', '0', 0.9]], [evi])


def double_hypo_cpd(evi_1, evi_2):
    return ConditionalProbabilityTable(
        [['1', '1', '1', 0.9],
         ['1', '1', '0', 0.1],
         ['1', '0', '1', 0.9],
         ['1', '0', '0', 0.1],
         ['0', '1', '1', 0.9],
         ['0', '1', '0', 0.1],
         ['0', '0', '1', 0.1],
         ['0', '0', '0', 0.9]], [evi_1, evi_2])


class PeepoModel:
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
        self.peepo_actor = peepo_actor
        self.actors = actors
        self.genmodel = self.create_generative_model()
        self.motor_output = {pg.K_LEFT: False,
                             pg.K_RIGHT: False}
        self.obstacle_input = {'1': False,
                               '2': False,
                               '3': False,
                               '4': False,
                               '5': False,
                               '6': False}

    def create_generative_model(self):
        d1 = DiscreteDistribution({'0': 0.9, '1': 0.1})
        d2 = DiscreteDistribution({'0': 0.9, '1': 0.1})
        d3 = DiscreteDistribution({'0': 0.9, '1': 0.1})
        d4 = DiscreteDistribution({'0': 0.9, '1': 0.1})
        d5 = double_hypo_cpd(d1, d4)
        d6 = double_hypo_cpd(d2, d3)
        d7 = single_hypo_cpd(d3)
        d8 = single_hypo_cpd(d3)
        d9 = single_hypo_cpd(d3)
        d10 = single_hypo_cpd(d4)
        d11 = single_hypo_cpd(d4)
        d12 = single_hypo_cpd(d4)

        s1 = State(d1, name=self.WAN_LEFT)
        s2 = State(d2, name=self.WAN_RIGHT)
        s3 = State(d3, name=self.OBS_LEFT)
        s4 = State(d4, name=self.OBS_RIGHT)
        s5 = State(d5, name=self.MOT_LEFT)
        s6 = State(d6, name=self.MOT_RIGHT)
        s7 = State(d7, name=self.VIS_1)
        s8 = State(d8, name=self.VIS_2)
        s9 = State(d9, name=self.VIS_3)
        s10 = State(d10, name=self.VIS_4)
        s11 = State(d11, name=self.VIS_5)
        s12 = State(d12, name=self.VIS_6)

        model = BayesianNetwork()
        model.add_states(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12)
        model.add_edge(s1, s5)
        model.add_edge(s2, s6)
        model.add_edge(s3, s6)
        model.add_edge(s4, s5)
        model.add_edge(s3, s7)
        model.add_edge(s3, s8)
        model.add_edge(s3, s9)
        model.add_edge(s4, s10)
        model.add_edge(s4, s11)
        model.add_edge(s4, s12)
        model.bake()

        return GenerativeModel(model, SensoryInputVirtualPeepo(self))

    def process(self):
        self.calculate_obstacles()
        self.genmodel.process()

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
