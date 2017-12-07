from peepo.predictive_processing.discrete.node import Node
import uuid


class NodeExteroceptive(Node):
    def __init__(self, lm, children=None, th=0.1, name=uuid.uuid4()):
        super().__init__(lm, children, th, name)
