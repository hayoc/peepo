import json

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from config import ROOT_DIR


def bayesian_network_to_json(bayesian_network, bn_id):
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

    with open(ROOT_DIR + '/resources/bn/bn' + str(bn_id) + '.json', 'w') as outfile:
        json.dump(to_json, outfile)


def json_to_bayesian_network(bn_id):
    with open(ROOT_DIR + '/resources/bn/bn' + str(bn_id) + '.json') as json_data:
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
