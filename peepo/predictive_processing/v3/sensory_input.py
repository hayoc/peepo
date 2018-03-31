import numpy as np
import logging


class SensoryInput:

    def __init__(self, bot):
        self.bot = bot

    def action(self, node, prediction_error, prediction):
        logging.debug('issuing bot-action')
        # if prediction = [0.1, 0.9] (= moving) then move else stop
        if np.argmax(prediction) is 0:  # predicted stopped
            self.bot.stop()
        else:  # predicted moving
            self.bot.backward()

    def value(self, name):
        if name == 'infrared':
            # [0.1, 0.9] = OBSTACLE - [0.9, 0.1] = NO OBSTACLE
            return np.array([0.1, 0.9] if self.bot.vision() < 60 else np.array([0.9, 0.1]))
        else:
            # [0.1, 0.9] = MOVING - [0.9, 0.1] = NO MOVING
            return np.array([0.1, 0.9]) if self.bot.is_driving_backward() else np.array([0.9, 0.1])
