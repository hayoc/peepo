import numpy as np
import logging

from peepo.predictive_processing.discrete.region import Region

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

th = 0.5
lm = np.matrix([[0.8, 0.1],
                [0.2, 0.9]])

hyp = np.array([0.0, 1.0])

act = np.array([0.9, 0.1])

region = Region(lm, hyp=hyp, th=0.1, name='dangsaf')

if region.error(region.predict(), act):
    region.update(act)


class Hierarchy:

    def __init__(self):
        pass

    def start(self):
        pass
