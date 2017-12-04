import numpy as np
import logging

from peepo.predictive_processing.discrete.hierarchy import Hierarchy
from peepo.predictive_processing.discrete.level import Level
from peepo.predictive_processing.discrete.region import Region

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

"""
==================================================================================
==================================================================================
================================== EXAMPLE =======================================
==================================================================================
==================================================================================

Notes

Add the possibility of having multiple hypothesis regions, e.g. savannah-forest affect
danger-safety, but also foggy-clear. We can combine them by taking the averages of each
node in each region, e.g. 9-1 and 8-2 -> 8.5-1.5
"""

levels = [
    Level(0, ['A']),
    Level(1, ['B', 'C'])
]

graph = {
    'A': ['B', 'C'],
    'B': [],
    'C': []
}

act = {
    'B': np.array([0.1, 0.9]),
    'C': np.array([0.4, 0.6])
}

regions = {'A': Region(np.matrix([[0.3, 0.8], [0.7, 0.2]]), hyp=np.array([0.1, 0.9]), name='A'),
           'B': Region(np.matrix([[0.9, 0.1], [0.1, 0.9]]), name='B'),
           'C': Region(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='C')}

h = Hierarchy(graph, regions, act)
h.start()
