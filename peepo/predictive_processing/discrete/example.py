import numpy as np
import logging

from peepo.predictive_processing.discrete.sensory_input import SensoryInput
from peepo.predictive_processing.discrete.level import Level
from peepo.predictive_processing.discrete.module import Module
from peepo.predictive_processing.discrete.node import Node
from peepo.predictive_processing.discrete.node_exteroceptive import NodeExteroceptive
from peepo.predictive_processing.discrete.node_proprioceptive import NodeProprioceptive

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

"""
==================================================================================
==================================================================================
================================== EXAMPLE =======================================
==================================================================================
==================================================================================

Notes

- Add the possibility of having multiple hypothesis nodes, e.g. savannah-forest affect
danger-safety, but also foggy-clear. We can combine them by taking the averages of each
node in each node, e.g. 9-1 and 8-2 -> 8.5-1.5
DONE

- Add changing precision: either through threshold or weight in updating hypos

- Fix errors via motor commands (e.g. lowest (sensory input) levels can have 2 'hypos', one perceptual, one motor)

"""

A = Node(np.matrix([[0.3, 0.8], [0.7, 0.2]]), name='A')
B = NodeExteroceptive(np.matrix([[0.9, 0.1], [0.1, 0.9]]), name='B')
C = NodeProprioceptive(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='C')
D = Node(np.matrix([[0.3, 0.8], [0.7, 0.2]]), th=0.01, name='D')
E = Node(np.matrix([[0.3, 0.8], [0.7, 0.2]]), th=0.2, name='E')
A.setHyp('X', np.array([0.1, 0.9]))
A.children = [B, C]
D.setHyp('Z', np.array([0.1, 0.9]))
D.children = [C]
E.setHyp('Y', np.array([0.1, 0.9]))
E.children = [B]

levels = [
    Level(0, [A, D]),
    Level(1, [B, C])
]

si = SensoryInput({
    'B': np.array([0.9, 0.1]),
    'C': np.array([0.3, 0.7])
})

mod = Module(levels, si)
mod.predict_flow()
