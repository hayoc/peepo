import numpy as np
import logging

from peepo.predictive_processing.discrete.region import Region

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

class Hierarchy:

    def __init__(self, graph, regions, inputs):
        self.graph = graph
        self.regions = regions
        self.inputs = inputs
        self.rootnames = graph.get('root')

    def start(self):
        for rootname in self.rootnames:
            root = self.regions.get(rootname)
            self.predict_flow(root, root.getHyp(), [])

    def predict_flow(self, node, hyp, ancestors):
        ancestors.append(node)
        node.setHyp(hyp)
        name = node.__getattribute__('name')
        pred = node.predict()
        children = self.graph.get(name)

        if not children:
            input = self.inputs.get(name)
            if node.error(hyp, input):
                self.error_flow(node, input, ancestors)
        else:
            for child in children:
                childnode = regions.get(child)
                self.predict_flow(childnode, pred, ancestors)

    def error_flow(self, node, input, ancestors):
        logging.info("Error flow: " + node.__getattribute__('name'))

        node.update(input)
        if ancestors:
            self.error_flow(ancestors.pop(), node.getHyp(), ancestors)



graph = {
    'root': ['A'],
    'A': ['B', 'C'],
    'B': [],
    'C': []
}

act = {
    'B': np.array([0.9, 0.1]),
    'C': np.array([0.2, 0.8])
}

regions = {'A': Region(np.matrix([[0.3, 0.8], [0.7, 0.2]]), hyp=np.array([0.9, 0.1]), name='A'),
           'B': Region(np.matrix([[0.8, 0.1], [0.2, 0.9]]), name='B'),
           'C': Region(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='C')}

h = Hierarchy(graph, regions, act)
h.start()