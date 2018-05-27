from peepo.predictive_processing.v3.sensory_input import SensoryInput
import numpy as np


class SensoryInputCoin(SensoryInput):

    def __init__(self, coin_set):
        super().__init__()
        self.coin_set = coin_set
        self.index = 0

    def action(self, node, prediction_error, prediction):
        pass

    def value(self, name):
        # 0 = heads, 1 = tails
        val = np.array([1, 0]) if self.coin_set[self.index] == 0 else np.array([0, 1])
        self.index += 1
        return val
