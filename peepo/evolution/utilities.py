import itertools
import math

import numpy as np


def get_topologies(peepo_network, simple_first=False, max_topologies=None, max_removal=None):
    max_edges = fully_connected_network(peepo_network).get_edges()
    max_removal = max_removal if max_removal and max_removal < len(max_edges) else len(max_edges)

    topologies = []

    if simple_first:
        for x in range(0, max_removal + 1):
            for cmb in itertools.combinations(max_edges, x):
                topologies.append({
                    'edges': list(cmb),
                    'entropy': len(cmb)
                })

                if max_topologies and len(topologies) >= max_topologies:
                    return topologies
    else:
        for x in range(len(max_edges), len(max_edges) - max_removal, -1):
            for cmb in itertools.combinations(max_edges, x):
                topologies.append({
                    'edges': list(cmb),
                    'entropy': len(cmb)
                })

                if max_topologies and len(topologies) >= max_topologies:
                    return topologies

    return topologies


def fully_connected_network(peepo_network):
    for root in peepo_network.get_root_nodes():
        for leaf in peepo_network.get_leaf_nodes():
            peepo_network.add_edge((root, leaf))
    return peepo_network


def ga_child_cpd(card_parents, omega):
    """
    Used in the framework of Genetic Algorithm to initialize a childs cpd and to alter it later (mutation)
    The word 'child" used here is in the context of Bayesian network and is not an offspring of a GA


    :param card_parents: array containing the cardinality of the childs' parents
    :param omega: an array of dim cardinality of the child (card_child) containing the cpd's  generating parameters
    :return: a matrix of shape (card_child, numpy.prod(card_parents) containing the conditional probability distribution

    example :

        my_card_parents = [2, 3, 2]
        max_omega = 2 * math.pi * np.prod(my_card_parents)

        zero generation : (my_card_child = 3)
        my_omega = [2.0,0.1,0.9] -> only for this example. As a rule one will use
            my_omega = np.random.rand(my_card_child) * max_omega to initialize (0th generation)
            a child's cpd and
            my_omega += (0.5 - np.random.rand(my_card_child))*epsilon with espilon small (e.g. 0.05)
            when mutating the childs' cpd

        my_pdf = ga_child_cpd( my_card_parents, my_omega)

        mutation :
        epsilon = 0.05
        my_omega += (0.5 - np.random.rand(len(my_omega))*epsilon
        my_pdf_ = ga_child_cpd( my_card_parents, my_omega)

        --->
            Zero generation cpd
            [[0.07 0.25 0.53 0.18 0.14 0.37 0.18 0.09 0.54 0.46 0.06 0.45]
             [0.56 0.62 0.47 0.67 0.49 0.28 0.35 0.48 0.35 0.54 0.69 0.24]
             [0.37 0.13 0.00 0.15 0.37 0.35 0.47 0.44 0.11 0.00 0.25 0.31]]


            Mutated generation cpd
            [[0.08 0.23 0.54 0.21 0.13 0.37 0.20 0.06 0.55 0.55 0.03 0.47]
             [0.55 0.62 0.46 0.65 0.50 0.27 0.32 0.45 0.32 0.45 0.71 0.21]
             [0.37 0.14 0.00 0.13 0.38 0.36 0.47 0.49 0.13 0.00 0.26 0.33]]


            Delta zero generation - mutated cpd
            [[-0.01 0.01 -0.01 -0.03 0.02 -0.01 -0.03 0.03 -0.01 -0.08 0.04 -0.02]
             [0.01 -0.00 0.01 0.02 -0.01 0.01 0.03 0.03 0.03 0.09 -0.03 0.03]
             [-0.00 -0.01 -0.00 0.01 -0.01 -0.00 -0.00 -0.05 -0.03 -0.00 -0.01 -0.01]]
    """

    phase_shift = omega[0]
    n_comb = int(np.prod(card_parents))
    pdf = []
    for ang in omega:
        pdf_row = []
        for col in range(n_comb):
            pdf_row.append(math.sin(ang * (col + 1) + phase_shift) + 1.2)
        pdf.append(pdf_row)
    return normalize_distribution(np.asarray(pdf))


def ga_parent_cpd(omega):
    phase_shift = omega[0]

    pdf = []
    for idx, ang in enumerate(omega):
        pdf.append(math.sin(ang * (idx + 1) + phase_shift) + 1.2)

    return normalize_distribution(np.asarray(pdf))


def normalize_distribution(matrix):
    """
    Normalizes the columns of a matrix (i.e. sum matrix[:,i] = 1

    :param matrix: numpy 2 dimensional array
    :return: numpy 2 dimensional normalized array
    """
    factor = np.sum(matrix, axis=0)
    return matrix / factor


def get_adjency_map(network):
    """
    Returns a (pseudo)-adjency list of the topology
    i.e. a len(leaf_nodes)xlen(root_nodes) matrix with
    0 and 1's

    :param network: peepo_network
    :return: a matrix of shape (len(leaf_nodes),len(root_nodes)) containing the adjency matrix
    """
    roots = network.get_root_nodes()
    leaves = network.get_leaf_nodes()
    map = np.zeros((len(leaves), len(roots)))
    for i, root in enumerate(roots):
        for k, leaf in enumerate(leaves):
            for x in network.get_edges():
                if x[1] == leaf and x[0] == root:
                    map[k][i] = 1
    return map


def adjency_to_edges(network, i_map):
    """
    Returns array containing the tuples of the edges from a (pseudo)-adjency list of the topology
    i.e. a len(leaf_nodes)xlen(root_nodes) matrix with
    0 and 1's

    :param network: peepo_network
    :param i_map: a matrix of shape (len(leaf_nodes),len(root_nodes)) containing the adjency matrixa
    :return: array containing the tuples of the edges
    """
    edges = []
    for col, root in enumerate(network.get_root_nodes()):
        for row, leaf in enumerate(network.get_leaf_nodes()):
            if i_map[row][col] == 1:
                edges.append((root, leaf))
    return edges
