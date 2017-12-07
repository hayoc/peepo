import numpy as np
import logging

from peepo.predictive_processing.discrete.hierarchy import Hierarchy
from peepo.predictive_processing.discrete.sensory_input import SensoryInput
from peepo.predictive_processing.discrete.level import Level
from peepo.predictive_processing.discrete.module import Module
from peepo.predictive_processing.discrete.node import Node
from peepo.predictive_processing.discrete.node_exteroceptive import NodeExteroceptive
from peepo.predictive_processing.discrete.node_proprioceptive import NodeProprioceptive
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

- Add the possibility of having multiple hypothesis regions, e.g. savannah-forest affect
danger-safety, but also foggy-clear. We can combine them by taking the averages of each
node in each region, e.g. 9-1 and 8-2 -> 8.5-1.5
DONE

- Add changing precision: either through threshold or weight in updating hypos

- Fix errors via motor commands (e.g. lowest (sensory input) levels can have 2 'hypos', one perceptual, one motor)

"""

A = Node(np.matrix([[0.3, 0.8], [0.7, 0.2]]), name='A')
B = NodeExteroceptive(np.matrix([[0.9, 0.1], [0.1, 0.9]]), name='B')
C = NodeProprioceptive(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='C')
# D = Node(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='D')
# E = Node(np.matrix([[0.9, 0.4], [0.1, 0.6]]), name='E')
A.setHyp('X', np.array([0.1, 0.9]))
A.children = [B, C]
# B.children = [C]
# C.children = [D, E]

levels = [
    Level(0, [A]),
    Level(1, [B, C])
]

si = SensoryInput({
    'B': np.array([0.9, 0.1]),
    'C': np.array([0.3, 0.7])
})

mod = Module(levels, si)
mod.predict_flow()
#
# graph = {
#     'root': ['A'],
#     'A': ['B', 'C'],
#     'B': [],
#     'C': []
# }
#
# act = {
#     'B': np.array([0.1, 0.9]),
#     'C': np.array([0.4, 0.6])
# }
#
# regions = {'A': Region(np.matrix([[0.3, 0.8], [0.7, 0.2]]), hyp=np.array([0.1, 0.9]), name='A'),
#            'B': Region(np.matrix([[0.9, 0.1], [0.1, 0.9]]), name='B'),
#            'C': Region(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='C')}
#
# h = Hierarchy(graph, regions, act)
# h.start()
