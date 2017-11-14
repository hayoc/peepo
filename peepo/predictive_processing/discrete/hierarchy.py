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
    'A': ['B', 'C'],
    'B': [],
    'C': []
}

regions = {'A': Region(np.matrix([[0.3, 0.8], [0.7, 0.2]]), name='A'),
           'B': Region(np.matrix([[0.8, 0.1], [0.2, 0.9]]), name='B'),
           'C': Region(np.matrix([[0.7, 0.4], [0.3, 0.6]]), name='C')}

class Hierarchy:

    def __init__(self, graph, regions, par, parhyp):
        self.graph = graph
        self.regions = regions
        self.parent = par
        self.regions.get(par).setHyp(parhyp)

    def start(self):
        topregion = regions.get(self.parent)
        tophyp = topregion.predict()

        children = self.graph.get(self.parent)
        for child in children:
            child.setHyp(tophyp)



