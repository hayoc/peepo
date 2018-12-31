import itertools
import json
import os

import numpy as np

from config import ROOT_DIR
from peepo.predictive_processing.v3.peepo_network import PeepoNetwork


def get_topologies(peepo_network, max_removal=None):
    max_edges = fully_connected_network(peepo_network).get_edges()
    max_removal = max_removal or len(max_edges)

    topologies = []
    for x in range(0, max_removal + 1):
        for cmb in itertools.combinations(max_edges, x):
            edges = list(max_edges)

            for edge_to_remove in cmb:
                edges.remove(edge_to_remove)

            topologies.append({
                'edges': edges,
                'entropy': len(edges)
            })

    return topologies


def fully_connected_network(peepo_network):
    lans = peepo_network.get_lan_nodes()
    if len(lans) == 0:
        for root in peepo_network.get_root_nodes():
            for leaf in peepo_network.get_leaf_nodes():
                peepo_network.add_edge((root, leaf))
    else:
        for root in peepo_network.get_root_nodes():
            for lan in peepo_network.get_lan_nodes():
                peepo_network.add_edge((root, lan))
        for lan in peepo_network.get_lan_nodes():
            for leaf in peepo_network.get_leaf_nodes():
                peepo_network.add_edge((lan, leaf))

    return peepo_network


def get_index_matrix(cardinality):
    """
    Returns the state combinations of the parent nodes given the cardinality of the parents nodes

    :param cardinality: list with the cardinalities of the parent
    :returns: an array with the combination of all possible states
    :type cardinality: list
    :rtype : np.array

    Example
    -------
    >>> cardinality = [2, 3, 2]
    >>> print(get_index_matrix(cardinality))
    [[0 0 0 0 0 0 1 1 1 1 1 1],
     [0 0 1 1 2 2 0 0 1 1 2 2],
     [0 1 0 1 0 1 0 1 0 1 0 1 ]]
    """
    blocks = []
    for i in range(0, len(cardinality)):
        blocks.append([s for s in range(0, cardinality[i])])
    return np.transpose(np.asarray(list(itertools.product(*blocks))))


def create_fixed_parent(cardinality, state=0, modus='status'):
    hi = 0.99
    lo = 0.01 / (cardinality - 1)
    ar = np.full(cardinality, lo)
    if (modus == 'status'):
        ar[state] = hi
    # normalize
    som = 0
    for i in range(0, cardinality):
        som += ar[i]
    for i in range(0, cardinality):
        ar[i] /= som
    return ar


def write_to_file(name, peepo_network):
    directory = ROOT_DIR + '/resources/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + '/' + str(name) + '.json', 'w') as outfile:
        json.dump(peepo_network.to_json(), outfile, default=str)


def read_from_file(name):
    with open(ROOT_DIR + '/resources/' + str(name) + '.json') as json_data:
        return PeepoNetwork().from_json(json.load(json_data))