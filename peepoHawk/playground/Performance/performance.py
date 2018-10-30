import math
import os
import random
import sys

class Metrics(object):
    """ This class represents metrics to evaluate the performance of the models"""


    def __init__(self, epoch,max_epochs):
        self.max_epoch = max_epochs
        self.epoch = epoch
        self.number_of_successes = 0
        self.epochs_to_fulfill_task = 0



