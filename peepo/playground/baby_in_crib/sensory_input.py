import numpy as np

from peepo.predictive_processing.v3.sensory_input import SensoryInput


class SensoryInputCribBaby(SensoryInput):

    def __init__(self, baby, crib):
        super().__init__()
        self.baby = baby
        self.crib = crib

    def action(self, node, prediction_error, prediction):
        if 'motor' in node:
            for limb in self.baby.limbs:
                self.baby.limbs[limb] = False
            self.baby.limbs[node] = True
            self.crib.mobile = self.crib.ribbon == node

    def value(self, name):
        if 'boredom' in name:
            # Always expects interesting view, e.g. moving mobile
            return np.array([0.1, 0.9])
        if 'mobile' in name:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING MOBILE
            return np.array([0.1, 0.9]) if self.crib.mobile else np.array([0.9, 0.1])
        if 'motor' in name:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING LIMB
            return np.array([0.1, 0.9]) if self.baby.limbs[name] else np.array([0.9, 0.1])
        raise ValueError('Unexpected type of node')
