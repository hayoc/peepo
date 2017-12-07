from peepo.predictive_processing.discrete.node import Node
import uuid
import logging


class NodeProprioceptive(Node):
    def __init__(self, lm, children=None, th=0.1, name=uuid.uuid4()):
        super().__init__(lm, children, th, name)

    def update(self, si, key=None):
        """
        Instead of updating the Model Hypothesis, Proprioceptive Nodes
         issue motor commands to resolve prediction error, since the
         error is caused by expectations about current motor states.

        :param si: Array of (Sensory Input) Actual Values (E)
        :param key: Optional String key to find the input in SensoryInput class

        """
        if si.vals[key][0] < 0.7:
            si.vals[key][0] += 0.1
            si.vals[key][1] -= 0.1
        #super().update(si)
