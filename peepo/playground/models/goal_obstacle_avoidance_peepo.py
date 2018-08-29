import math

import numpy as np


class Peepo():

    def __init__(self, actor, goal):
        self.actor = actor
        self.goal = goal

    def update(self, model):
        network = model.models['main'].model

        dx = self.actor.rect.x - self.goal[0]
        dy = self.actor.rect.y - self.goal[1]
        rads = math.atan2(-dy, dx)
        rads %= 2 * math.pi
        degs = -(math.degrees(rads) + 180) % 360

        origin = math.radians(self.actor.rotation)
        target = math.radians(degs)

        # TODO: check whether obstacle is active, if it is, don't update navigation module
        # TODO: turn off hypos every time we recalculate
        # TODO: we should not be changing hypos manually - instead sensory inputs can be given which change the
        # TODO: hypos by generating prediction errors
        network.get_cpds('goal_right').values = np.array([0.1, 0.9])
        network.get_cpds('goal_left').values = np.array([0.1, 0.9])
        if origin < target and target - origin > 0.2:
            network.get_cpds('goal_left').values = np.array([0.9, 0.1])
        elif origin > target and origin - target > 0.2:
            network.get_cpds('goal_right').values = np.array([0.9, 0.1])
