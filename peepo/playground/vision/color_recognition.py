import itertools

import numpy as np
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput
from peepo.predictive_processing.v3.utils import read_from_file, fully_connected_network, get_topologies

CASE = 'color_recognition'
NUM_COLORS = 16
DATA = {'BENS_0': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'BENS_1': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        'BENS_2': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        'BENS_3': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        'WORLD_0': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'WORLD_1': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        'WORLD_2': [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]}


class SensoryInputColorRecognition(SensoryInput):

    def __init__(self, expected):
        super().__init__()
        self.expected = expected

    def action(self, node, prediction):
        pass

    def value(self, name):
        expected_pixel = self.expected[int(name) - 4]
        return np.array([0.99, 0.01]) if expected_pixel == 0 else np.array([0.01, 0.99])


def get_index_matrix(cardinality):
    blocks = []
    for i in range(0, len(cardinality)):
        subblock = []
        [subblock.append(int(s)) for s in range(0, cardinality[i])]
        blocks.append(subblock)
    return np.transpose(np.asarray(list(itertools.product(*blocks))))


def get_color_cpd(cardinality):
    hi = 1
    lo = 0
    C = np.prod(cardinality)
    matrix = np.full((3, C), 1. / 3.)
    matrix[0] = [hi, lo, lo, hi, lo, lo, hi, lo, hi, lo, lo, hi, lo, lo, hi, lo]
    matrix[1] = [lo, hi, lo, lo, hi, lo, lo, hi, lo, hi, lo, lo, hi, lo, lo, hi]
    matrix[2] = [lo, lo, hi, lo, lo, hi, lo, lo, lo, lo, hi, lo, lo, hi, lo, lo]
    return matrix


def create_fixed_parent(cardinality, state=0, modus='status'):
    hi = 0.99
    lo = 0.01 / (cardinality - 1)
    ar = np.full(cardinality, lo)
    if modus == 'status':
        ar[state] = hi
    # normalize
    som = 0
    for i in range(0, cardinality):
        som += ar[i]
    for i in range(0, cardinality):
        ar[i] /= som
    return ar


pp_net = read_from_file('color_recognition')
pp_net = fully_connected_network(pp_net)
pp_net.train_data = DATA

scores = []
best_error = {
    'error': 100000,
    'topology': {}
}

index_matrix = get_index_matrix([2, 2, 2, 2])
color_cpd = get_color_cpd([2, 2, 2, 2])
topologies = get_topologies(pp_net)

for x, topology in enumerate(topologies):
    print(str(x) + ' out of ' + str(len(topologies)))
    pp_net.edges = topology['edges']
    pp_net.assemble()

    pm_net = pp_net.to_pomegranate()

    total_error = 0
    for color in range(0, NUM_COLORS):
        for i, pixel in enumerate(index_matrix[:, color]):
            pm_net.states[i].distribution = DiscreteDistribution(
                {'0.0': 0.99, '1.0': 0.01} if pixel == 0 else {'0.0': 0.01, '1.0': 0.99})

        sensory_input = SensoryInputColorRecognition(expected=color_cpd[:, color])
        gen_model = GenerativeModel(pm_net, sensory_input)

        total_prediction_error_size = 0
        for index, node in enumerate(gen_model.predict()):
            node_name = gen_model.bayesian_network.states[index].name
            if int(node_name) > 3:
                prediction = np.array([x[1] for x in sorted(node.items(), key=lambda tup: tup[0])])
                observation = gen_model.sensory_input.value(node_name)
                prediction_error_size = gen_model.error_size(prediction, observation)
                total_prediction_error_size += prediction_error_size
        total_error += total_prediction_error_size
    total_error /= NUM_COLORS
    scores.append([topology['entropy'], total_error])

    if total_error <= best_error['error']:
        best_error['error'] = total_error
        best_error['topology'] = topology

print(scores)
print(best_error)
