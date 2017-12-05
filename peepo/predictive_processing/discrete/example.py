import numpy as np
import logging

from peepo.predictive_processing.discrete.hierarchy import Hierarchy
from peepo.predictive_processing.discrete.level import Level
from peepo.predictive_processing.discrete.module import Module
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

A = Region(np.matrix([[0.3, 0.8], [0.7, 0.2]]), hyp=np.array([0.1, 0.9]), name='A')
B = Region(np.matrix([[0.9, 0.1], [0.1, 0.9]]), hyp=np.array([0.1, 0.9]), name='B')
C = Region(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='C')
D = Region(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='D')
E = Region(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='E')
A.children = [C]
B.children = [C]
C.children = [D, E]

levels = [
    Level(0, [A, B]),
    Level(1, [C]),
    Level(2, [D, E])
]

act = {
    'D': np.array([[0.4, 0.6]]),
    'E': np.array([[0.4, 0.6]])
}

mod = Module(levels, act)
mod.predict_flow()
