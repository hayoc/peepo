import numpy as np

from peepo.pp.v3.sensory_input import SensoryInput


class SensoryInputBot(SensoryInput):

    def __init__(self, bot):
        super().__init__()
        self.bot = bot

    def action(self, node, prediction):
        # if prediction = [0.1, 0.9] (= moving) then move else stop
        if np.argmax(prediction) > 0:  # predicted moving
            self.bot.backward()
        else:  # predicted stopped
            self.bot.stop()

    def value(self, name):
        if name == 'infrared':
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            return np.array([0.1, 0.9] if self.bot.vision() < 60 else np.array([0.9, 0.1]))
        else:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            return np.array([0.1, 0.9]) if self.bot.is_driving_backward() else np.array([0.9, 0.1])
