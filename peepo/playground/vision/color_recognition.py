import logging
import re

import numpy as np
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput
from peepo.predictive_processing.v3.utils import read_from_file, fully_connected_network, get_topologies
from peepo.visualize.graph import draw_network

CASE = 'color_recognition'
NUM_COLORS = 16
DATA = np.array([
    # OBSERVED NODES
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    # HYPOTHESIS NODES
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
])


def start():
    pp_net = read_from_file('color_recognition')
    pp_net = fully_connected_network(pp_net)
    pp_net.train_data = DATA

    topologies = get_topologies(pp_net, max_removal=1)
    amount = len(topologies)

    scores = []
    best_error = {
        'error': 100000,
        'network': None
    }

    for x, topology in enumerate(topologies):

        pp_net.set_edges(topology['edges'])
        pp_net.set_cpds({})
        pp_net.assemble()

        total_error = 0
        for color in range(0, NUM_COLORS):
            color_array = DATA[:, color]

            sensory_input = SensoryInputColorRecognition(expected=color_array)
            gen_model = GenerativeModel(pp_net, sensory_input)
            hack_hypothesis(gen_model.bayesian_network, color_array)

            total_error += gen_model.process(structure_learning=True)
        total_error /= NUM_COLORS
        scores.append([topology['entropy'], total_error])
        logging.info('Loop %d out of %d | score: %s', x, amount, str(total_error))

        if total_error <= best_error['error']:
            best_error['error'] = total_error
            best_error['network'] = pp_net

    draw_network(best_error['network'])
    logging.info('Lowest error: %s', str(best_error))


def hack_hypothesis(pm_net, color_array):
    for x in range(0, 4):
        pixel = color_array[x]
        pm_net.states[x].distribution = DiscreteDistribution(
            {'0.0': 0.99, '1.0': 0.01} if pixel == 0 else {'0.0': 0.01, '1.0': 0.99})


class SensoryInputColorRecognition(SensoryInput):

    def __init__(self, expected):
        super().__init__()
        self.expected = expected

    def action(self, node, prediction):
        pass

    def value(self, name):
        expected_pixel = self.expected[int(re.search(r'\d+', name).group())]
        return np.array([0.99, 0.01]) if expected_pixel == 0 else np.array([0.01, 0.99])


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
start()
