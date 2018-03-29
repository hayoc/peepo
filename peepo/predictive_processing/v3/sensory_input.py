import numpy as np
import logging


class SensoryInput:

    def __init__(self, bot):
        self.bot = bot

    def action(self, node, prediction_error, prediction):
        logging.debug('issuing bot-action')
        if np.argmax(prediction) is 0:
            self.bot.backward()
        else:
            self.bot.stop()

    def value(self, name):
        if name == 'infrared':
            return np.array([0.1, 0.9] if self.bot.vision() < 60 else np.array([0.9, 0.1]))
        else:
            return np.array([0.1, 0.9]) if self.bot.is_driving_backward() else np.array([0.9, 0.1])
