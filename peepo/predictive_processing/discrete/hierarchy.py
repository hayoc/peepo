import numpy as np
import logging

from peepo.predictive_processing.discrete.region import Region

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

act = np.array([0.9, 0.1])
actls = np.array([0.3, 0.7])

dangsafe = Region(np.matrix([[0.3, 0.8],
                             [0.7, 0.2]]), name='dangsafe')
redgreen = Region(np.matrix([[0.8, 0.1],
                             [0.2, 0.9]]), name='redgreen')
loudsoft = Region(np.matrix([[0.7, 0.4],
                             [0.3, 0.6]]), name='loudsoft')

savforesthyp = np.array([0.8, 0.2])
dangsafe.setHyp(savforesthyp)
dangsafehyp = dangsafe.predict()
redgreen.setHyp(dangsafehyp)
redgreenhyp = redgreen.predict()
loudsoft.setHyp(dangsafehyp)
loudsofthyp = loudsoft.predict()

if loudsoft.error(loudsofthyp, actls):
    loudsoft.update(actls)
if redgreen.error(redgreenhyp, act):
    redgreen.update(act)
if dangsafe.error(dangsafehyp, redgreen.getHyp()):
    dangsafe.update(redgreen.getHyp())

logging.info("------------------------------")
savforesthyp = dangsafe.getHyp()
dangsafe.setHyp(savforesthyp)
dangsafehyp = dangsafe.predict()
redgreen.setHyp(dangsafehyp)
redgreenhyp = redgreen.predict()
loudsoft.setHyp(dangsafehyp)
loudsofthyp = loudsoft.predict()

if loudsoft.error(loudsofthyp, actls):
    loudsoft.update(actls)
if redgreen.error(redgreenhyp, act):
    redgreen.update(act)
if dangsafe.error(dangsafehyp, redgreen.getHyp()):
    dangsafe.update(redgreen.getHyp())

logging.info("------------------------------")
savforesthyp = dangsafe.getHyp()
dangsafe.setHyp(savforesthyp)
dangsafehyp = dangsafe.predict()
redgreen.setHyp(dangsafehyp)
redgreenhyp = redgreen.predict()
loudsoft.setHyp(dangsafehyp)
loudsofthyp = loudsoft.predict()

if loudsoft.error(loudsofthyp, actls):
    loudsoft.update(actls)
if redgreen.error(redgreenhyp, act):
    redgreen.update(act)
if dangsafe.error(dangsafehyp, redgreen.getHyp()):
    dangsafe.update(redgreen.getHyp())


graph = {
    'root': ['A'],
    'A': ['B', 'C'],
    'B': [],
    'C': []
}

act = {
    'B': np.array([0.9, 0.1]),
    'C': np.array([0.3, 0.7])
}

regions = {'A': Region(np.matrix([[0.3, 0.8], [0.7, 0.2]]), hyp=np.array([0.9, 0.1]), name='A'),
           'B': Region(np.matrix([[0.8, 0.1], [0.2, 0.9]]), name='B'),
           'C': Region(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='C')}


class Hierarchy:

    def __init__(self, graph, regions, inputs):
        self.graph = graph
        self.regions = regions
        self.inputs = inputs
        self.rootname = graph.get('root')

    def start(self):
        root = self.regions.get(self.rootname)
        self.predict_flow(root, root.predict(), [])

    def predict_flow(self, node, hyp, ancestors):
        ancestors.append(node)
        node.setHyp(hyp)
        name = node.__getattribute__('name')

        children = self.graph.get(name)

        if not children:
            input = self.inputs.get(name)
            if node.error(hyp, input):
                self.error_flow(node, input, ancestors)
        else:
            for child in children:
                childnode = regions.get(child)
                self.predict_flow(childnode, childnode.predict(), ancestors)

    def error_flow(self, node, input, ancestors):
        node.update(input)
        if ancestors:
            self.error_flow(ancestors.pop(), node.getHyp(), ancestors)





