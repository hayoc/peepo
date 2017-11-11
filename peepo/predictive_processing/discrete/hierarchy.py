import numpy as np
import logging

from peepo.predictive_processing.discrete.region import Region

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

th = 0.5
lm = np.matrix([[0.8, 0.1],
                [0.2, 0.9]])

hyp = np.array([0.47, 0.52])

act = np.array([0.9, 0.1])

region = Region(lm, hyp, th)

if region.compare(region.predict(), act):
    region.update(act)