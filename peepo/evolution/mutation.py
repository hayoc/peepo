import logging
import math
import random
import uuid

import numpy as np

from peepo.evolution.utilities import ga_parent_cpd, ga_child_cpd


def add_node(network):
    """
    Random mutation of adding a single node and an edge.
    :param network: PeepoNetwork to mutate
    :return: mutated PeepoNetwork
    """
    new_parent_node = str(uuid.uuid4())[:8]
    child_node = random.choice(network.get_leaf_nodes())
    logging.info('Adding new node with edge to: %s to %s', new_parent_node, child_node)

    network.add_belief_node(new_parent_node, 2)
    network.add_edge((new_parent_node, child_node))

    parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(child_node)]

    network.add_omega(new_parent_node, np.random.rand(2) * (2 * math.pi))
    network.add_cpd(new_parent_node, ga_parent_cpd(network.omega_map[new_parent_node]))
    network.add_cpd(child_node, ga_child_cpd(parents_card, network.omega_map[child_node]))

    return network


def add_edge(network):
    """
    Random mutation of adding a single edge between two random chosen nodes.
    :param network: PeepoNetwork to mutate
    :return: mutated PeepoNetwork
    """
    node_pool = network.get_root_nodes()
    if not node_pool:
        logging.debug('Model contains no valid nodes to add edge from... Adding a new node')
        return add_node(network)

    parent_node = random.choice(node_pool)
    child_pool = np.setdiff1d(network.get_leaf_nodes(), network.get_outgoing_edges(parent_node), assume_unique=True)
    if not child_pool.size:
        return remove_edge(network)
    child_node = random.choice(child_pool)
    logging.info('Adding edge from %s to %s', parent_node, child_node)

    network.add_edge((parent_node, child_node))

    parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(child_node)]

    network.add_cpd(child_node, ga_child_cpd(parents_card, network.omega_map[child_node]))

    return network


def remove_node(network):
    """
    Random mutation of removing a single node chosen randomly.
    :param network: PeepoNetwork to mutate
    :return: mutated PeepoNetwork
    """
    node_pool = network.get_root_nodes()
    if not node_pool:
        logging.debug('Model contains no valid nodes to remove... Adding a new node')
        return add_node(network)

    to_remove = random.choice(node_pool)
    affected_children = network.get_outgoing_edges(to_remove)

    network.remove_belief_node(to_remove)
    logging.info('Removed node %s', to_remove)

    for child_node in affected_children:
        parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(child_node)]

        network.add_cpd(child_node, ga_child_cpd(parents_card, network.omega_map[child_node]))

    return network


def remove_edge(network):
    """
    Random mutation of removing a single edge between two random chosen nodes.
    :param network: PeepoNetwork to mutate
    :return: mutated PeepoNetwork
    """
    node_pool = network.get_root_nodes()
    if not node_pool:
        logging.debug('Model contains no valid nodes to remove an edge from... Adding a new node')
        return add_node(network)

    parent_node = random.choice(node_pool)
    outgoing_edges = network.get_outgoing_edges(parent_node)
    if len(outgoing_edges) <= 1:
        logging.debug('Model contains no valid edges to remove... Adding a new edge instead')
        return add_edge(network)

    child_node = random.choice(outgoing_edges)

    network.remove_edge((parent_node, child_node))
    logging.info('Removed edge (%s, %s)', parent_node, child_node)

    parents_card = [network.cardinality_map[x] for x in network.get_incoming_edges(child_node)]

    network.add_cpd(child_node, ga_child_cpd(parents_card, network.omega_map[child_node]))

    return network