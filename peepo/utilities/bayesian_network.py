import json
import logging
import math
import os
import random

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from config import ROOT_DIR


def bayesian_network_to_json(bayesian_network, folder_id, bn_id):
    cpds = []
    for cpd in bayesian_network.get_cpds():
        cpds.append({
            'variable': cpd.variable,
            'variable_card': cpd.variable_card,
            'values': cpd.get_values().tolist(),
            'evidence': cpd.get_evidence(),
            'evidence_card': cpd.cardinality[1:].tolist()
        })

    to_json = {
        'nodes': bayesian_network.nodes(),
        'edges': bayesian_network.edges(),
        'cpds': cpds
    }

    directory = ROOT_DIR + '/resources/bn/' + str(folder_id)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + '/' + str(bn_id) + '.json', 'w') as outfile:
        json.dump(to_json, outfile)


def json_to_bayesian_network(folder_id, bn_id):
    with open(ROOT_DIR + '/resources/bn/' + str(folder_id) + '/' + str(bn_id) + '.json') as json_data:
        json_object = json.load(json_data)

        bayesian_network = BayesianModel()
        bayesian_network.add_nodes_from(json_object['nodes'])
        bayesian_network.add_edges_from(json_object['edges'])

        for cpd_data in json_object['cpds']:
            cpd = TabularCPD(variable=cpd_data['variable'], variable_card=cpd_data['variable_card'],
                             values=np.array(cpd_data['values']), evidence=cpd_data['evidence'],
                             evidence_card=np.array(cpd_data['evidence_card']))
            bayesian_network.add_cpds(cpd)
        bayesian_network.check_model()
        return bayesian_network


def add_node(model):
    model = model.copy()
    new_node_name = str(len(model))
    node_to_add_to = random.choice(model.nodes())
    logging.info('Adding new node with edge to: %s to %s', new_node_name, node_to_add_to)

    model.add_node(new_node_name)
    model.add_edge(new_node_name, node_to_add_to)

    new_node_cpd = TabularCPD(variable=new_node_name, variable_card=2, values=[random_distribution(2)])
    model.add_cpds(new_node_cpd)

    old_cpd = model.get_cpds(node_to_add_to)
    variable_card = old_cpd.variable_card
    evidence = old_cpd.get_evidence()
    evidence.append(new_node_name)
    evidence_card = list(old_cpd.cardinality[1:])
    old_evidence_card = list(evidence_card)
    evidence_card.append(2)

    values = expand_cpd(reshape_cpd(old_cpd.values, variable_card, old_evidence_card), 2)

    cpd_for_changed_node = TabularCPD(variable=node_to_add_to,
                                      variable_card=variable_card,
                                      values=values,
                                      evidence=evidence,
                                      evidence_card=evidence_card)

    check_for_nan(node_to_add_to, values)

    model.remove_cpds(old_cpd)
    model.add_cpds(cpd_for_changed_node)

    model.check_model()

    return model


def add_edge(model):
    model = model.copy()
    parent_node = random.choice(model.nodes())
    child_node = random.choice(filter_for_edge(model, parent_node))
    logging.info('Adding edge from %s to %s', parent_node, child_node)

    try:
        model.add_edge(parent_node, child_node)
    except ValueError:
        return add_edge(model)
    old_cpd = model.get_cpds(child_node)
    variable_card = old_cpd.variable_card
    evidence = old_cpd.get_evidence()
    evidence.append(parent_node)
    evidence_card = list(old_cpd.cardinality[1:])
    old_evidence_card = list(evidence_card)
    evidence_card.append(model.get_cpds(child_node).variable_card)

    values = expand_cpd(reshape_cpd(old_cpd.values, variable_card, old_evidence_card), 2)

    new_cpd = TabularCPD(variable=child_node,
                         variable_card=old_cpd.variable_card,
                         values=values,
                         evidence=evidence,
                         evidence_card=evidence_card)

    check_for_nan(child_node, values)

    model.remove_cpds(old_cpd)
    model.add_cpds(new_cpd)

    model.check_model()

    return model


def remove_node(model):
    model = model.copy()
    options = filter_observed_nodes([x for x in model.nodes() if x not in model.get_leaves()])
    if not options:
        logging.warning('Model contains no valid nodes to remove... Choosing new mutation')
        return random_mutation(model)

    node_to_remove = random.choice(options)
    model.remove_node(node_to_remove)

    logging.info('Removed %s', node_to_remove)
    model.check_model()

    return model


OBSERVABLES = ['motor', 'vision', 'obs']


def filter_observed_nodes(nodes):
    result = []
    for node in nodes:
        if not any(obs in node for obs in OBSERVABLES):
            result.append(node)
    return result


def filter_non_observed_nodes(nodes):
    result = []
    for node in nodes:
        if any(obs in node for obs in OBSERVABLES):
            result.append(node)
    return result


def remove_edge(model):
    model = model.copy()
    options = model.edges()
    if not options:
        logging.warning('Model contains no valid edges to remove... Choosing new mutation')
        return random_mutation(model)

    edge_to_remove = random.choice(options)
    logging.info('Removing edge %s', str(edge_to_remove))
    parent_node = edge_to_remove[0]
    child_node = edge_to_remove[1]

    model.remove_edge(parent_node, child_node)

    old_cpd = model.get_cpds(child_node)

    evidence = old_cpd.get_evidence()
    evidence.remove(parent_node)

    if len(evidence) is not 0:
        values = old_cpd.get_values()[:, 0::2]  # TODO: Make it work for non-binary CPDs
        evidence_card = old_cpd.cardinality[len(old_cpd.get_evidence()):]  # TODO: Idem

        new_cpd = TabularCPD(variable=child_node,
                             variable_card=old_cpd.variable_card,
                             values=values,
                             evidence=evidence,
                             evidence_card=evidence_card)
    else:
        values = [old_cpd.get_values()[:, 0]]

        new_cpd = TabularCPD(variable=child_node,
                             variable_card=old_cpd.variable_card,
                             values=values)

    check_for_nan(child_node, values)

    model.remove_cpds(old_cpd)
    model.add_cpds(new_cpd)

    model.check_model()

    return model


def change_parameters(model):
    model = model.copy()
    node = random.choice(model.nodes())
    logging.info('Changing parameters of %s', node)

    cpd = model.get_cpds(node)
    values = reshape_cpd(cpd.values, cpd.variable_card, list(cpd.cardinality[1:]))

    for idx_col, col in enumerate(values.T):
        for idx_row, row in enumerate(col):
            to_add = abs(math.log(row, 100))
            to_subtract = to_add / (len(col) - 1)

            values[idx_row, idx_col] = values[idx_row, idx_col] + to_add
            for idx_row_copy, row_copy in enumerate(col):
                if idx_row_copy is not idx_row:
                    values[idx_row_copy, idx_col] = values[idx_row_copy, idx_col] - to_subtract

    check_for_nan(node, values)
    cpd.values = values
    model.check_model()

    return model


def filter_for_edge(model, parent_node):
    filtered1 = [x for x in model.nodes() if x not in [parent_node]]

    filtered2 = []
    for x in filtered1:
        remove = False
        for edge in model.edges(parent_node):
            if x in edge:
                remove = True

        if not remove:
            filtered2.append(x)
    return filtered2


def random_distribution(size):
    dist = np.random.uniform(0.3, 0.7, size=size)
    return dist / dist.sum()


def get_two_dim(array):
    if len(array.shape) > 1:
        return array
    return array.reshape(array.shape[0], -1)


def reshape_cpd(cpd, variable_card, evidence_card):
    if len(cpd.shape) == 1:
        return cpd.reshape([variable_card, 1])
    return cpd.reshape([variable_card, np.prod(evidence_card)])


def expand_cpd(cpd, evidence_card):
    cpd = np.repeat(cpd, evidence_card, axis=1)
    for x in range(0, cpd.shape[1]):
        perturbation = random.uniform(-0.1, 0.1)
        cpd[0, x] = cpd[0, x] + perturbation  # TODO: Now it only works when variable has 2 values... fix this
        cpd[1, x] = cpd[1, x] - perturbation
    if len(cpd.shape) > 2:
        print()
    return cpd


def check_for_nan(node, array):
    if np.isnan(array).any():
        raise ValueError('New CPD has NaNs for node: ' + node)


MUTATIONS = [add_node, add_edge, change_parameters, remove_node, remove_edge]


def random_mutation(model):
    return random.choice(MUTATIONS)(model)

# network = createone()
# for _ in range(0, 1000):
#     pop = random_mutation(network)
