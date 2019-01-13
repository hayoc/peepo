import datetime
import datetime as dt
import itertools
import json
import math
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from pomegranate.BayesianNetwork import BayesianNetwork
from pomegranate.base import State
from pomegranate.distributions.ConditionalProbabilityTable import ConditionalProbabilityTable
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution

from config import ROOT_DIR
from peepo.predictive_processing.v3.utils import get_index_matrix

'''jan 2019: changes by Bufo:
    1) adapted get_topologies as the previous version did allow orphan leaf nodes 
    This genenrated problems in Pomegranate.
        Q: keep it that way or adapt so that Pomegranete can accept orphan leaf nodes?
    2) added a method disassemble() in class PeepoNetwork in order to facilitate the methods in GA
        Q: OK or do it another way
    '''


def get_topologies(peepo_network, max_removal=None):
    max_edges = fully_connected_network(peepo_network).get_edges()
    max_removal = max_removal or len(max_edges)
    n_root_nodes = len(peepo_network.get_root_nodes())
    leaf_nodes = peepo_network.get_leaf_nodes()

    topologies = []
    for x in range(0, max_removal + 1):
        for cmb in itertools.combinations(max_edges, x):
            edges = list(max_edges)
            append = True
            for ln in leaf_nodes:
                deleted_root_nodes = []
                [deleted_root_nodes.append(x[0]) for x in cmb if x[1] == ln]
                if len(deleted_root_nodes) >= n_root_nodes:
                    append = False
            for edge_to_remove in cmb:
                edges.remove(edge_to_remove)
            if append:
                topologies.append({
                    'edges': edges,
                    'entropy': len(edges)
                })
    return topologies


def fully_connected_network(peepo_network):
    for root in peepo_network.get_root_nodes():
        for leaf in peepo_network.get_leaf_nodes():
            peepo_network.add_edge((root, leaf))
    return peepo_network


def write_to_file(name, peepo_network):
    directory = ROOT_DIR + '/resources/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + '/' + str(name) + '.json', 'w') as outfile:
        json.dump(peepo_network.to_json(), outfile, default=str)


def read_from_file(name):
    with open(ROOT_DIR + '/resources/' + str(name) + '.json') as json_data:
        return PeepoNetwork().from_json(json.load(json_data)).assemble()


def converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()

class PeepoNetwork:
    """
    A Peepo Model implemented as a Bayesian Network model.

        A Peepo Model has a naming convention which needs to be followed in order to
        be able to be used in the Predictive Processing flow defined in
            <peepo.predictive_processing.v3.generative_model.GenerativeModel>
        The model is a wrapper around the Bayesian Network defined in
            <pomegranate.BayesianNetwork>

        Parameters
        ----------
        identification : str, optional
            The name of the network. Default is None
        description : str, optional
            The description of the network. Default is None
        train_data : np.array, optional
            The data to fit the network to, given a structure, to generate the CPDs. Default is None
        frozen : bool, optional
            Whether the network can be modified. Default is False
        bel_nodes : list, optional
            The belief nodes of the network. Default is None
        mem_nodes : list, optional
            The memory nodes of the network. Default is None
        lan_nodes : list, optional
            The latent nodes of the network. Default is None
        ext_nodes : list, optional
            The exteroceptive nodes of the network. Default is None
        int_nodes : list, optional
            The interoceptive nodes of the network. Default is None
        pro_nodes : list, optional
            The proprioceptive nodes of the network. Default is None
        edges : list, optional
            The edges of the network. Default is None
        edges : dict, optional
            The cpds of the network. Default is None

        Example
        -------
        >>> from peepo.predictive_processing.v3.peepo_network import PeepoNetwork
        >>> pp_network = PeepoNetwork(bel_nodes=[{'name': 'belief_1', 'card': 2}],
        >>>                           ext_nodes=[{'name': 'exteroceptive_1', 'card': 2}]),
        >>>                           edges=[('belief_1', 'exteroceptive_1')],
        >>>                           cpds={'belief_1': [0.7, 0.3], 'exteroceptive_1': [[0.9, 0.1], [0.1, 0.9]]}
        >>> pp_network.assemble()
        >>> print(pp_network.to_json())
        {
            'header': {
                'identification': '',
                'description': '',
                'date': '',
                'frozen': False,
                'train_from': ''
            },
            'nodes': {
                'RON': {
                    'BEL': [{
                        'name': 'belief_1',
                        'card': 2
                    }],
                    'MEM': []
                },
                'LAN': [],
                'LEN': {
                    'EXT': [{
                        'name': 'exteroceptive_1',
                        'card': 2
                    }],
                    'INT': [],
                    'PRO': []
                }
            },
            'edges': [
                ('belief_1', 'exteroceptive_1')
            ],
            'cpds': {
                'belief_1': [0.7, 0.3]
                'exteroceptive_1': [[0.9, 0.1],
                                    [0.1, 0.9]]
            }
        }
    """

    def __init__(self,
                 identification=None,
                 description=None,
                 train_from=None,
                 train_data=None,
                 frozen=False,
                 bel_nodes=None,
                 mem_nodes=None,
                 lan_nodes=None,
                 ext_nodes=None,
                 int_nodes=None,
                 pro_nodes=None,
                 edges=None,
                 cpds=None):
        self.identification = identification or ''
        self.description = description or ''
        self.train_from = train_from or ''
        self.train_data = train_data or []
        self.frozen = frozen
        self.date = dt.datetime.now()
        self.bel_nodes = bel_nodes or []
        self.mem_nodes = mem_nodes or []
        self.lan_nodes = lan_nodes or []
        self.ext_nodes = ext_nodes or []
        self.int_nodes = int_nodes or []
        self.pro_nodes = pro_nodes or []
        self.edges = edges or []
        self.cpds = cpds or {}
        self.network = self.make_network()
        self.cardinality_map = self.make_cardinality_map()
        self.omega_map = self.make_omega_map()

    def assemble(self):
        self.network = self.make_network()
        self.cardinality_map = self.make_cardinality_map()
        self.omega_map = self.make_omega_map()
        return self

    def disassemble(self):
        self.edges = []
        self.cpds = {}
        self.omega_map = {}
        return self

    def to_pomegranate(self):
        if self.cpds:
            distributions = OrderedDict()

            for root in itertools.chain(self.bel_nodes, self.mem_nodes):
                cpd = DiscreteDistribution(dict(enumerate(self.cpds[root['name']])))
                distributions.update({root['name']: cpd})

            for child_node in itertools.chain(self.lan_nodes, self.ext_nodes, self.int_nodes, self.pro_nodes):
                parents = [parent for parent, child in self.edges if child == child_node['name']]
                parent_cpds = [distributions[key] for key in parents]

                cardinalities = [self.cardinality_map[key] for key in parents]
                cardinalities.append(self.cardinality_map[child_node['name']])

                states = get_index_matrix(cardinalities)
                original_cpd = np.array(self.cpds[child_node['name']])
                probabilities = []

                for col in range(0, original_cpd.shape[1]):
                    for row in range(0, original_cpd.shape[0]):
                        probabilities.append(original_cpd[row, col])

                if len(probabilities) != states.shape[1]:
                    print('NOOO')
                stacjed = np.vstack([states, probabilities])
                cpd = ConditionalProbabilityTable(stacjed.T.tolist(), parent_cpds)
                distributions.update({child_node['name']: cpd})

            states = OrderedDict()
            for key, value in distributions.items():
                states.update({key: State(value, name=key)})

            pm_net = BayesianNetwork()
            for state in states.values():
                pm_net.add_state(state)
            for edge in self.edges:
                pm_net.add_edge(states[edge[0]], states[edge[1]])
            pm_net.bake()

            return pm_net

        elif self.train_data:
            structure = []
            nodes = self.get_nodes()
            for node in nodes:
                parents = []
                for edge in self.edges:
                    if node == edge[1]:
                        parents.append(nodes.index(edge[0]))
                structure.append(tuple(parents))
            pm_net = BayesianNetwork.from_structure(X=self.train_data,
                                                    structure=tuple(structure),
                                                    state_names=nodes)

            for i, state in enumerate(pm_net.states):
                state.name = nodes[i]

                if isinstance(state.distribution, DiscreteDistribution):
                    cpd = []
                    parameter = state.distribution.parameters[0]
                    for x in range(0, self.cardinality_map[state.name]):
                        if x in parameter:
                            cpd.append(parameter[x])
                        else:
                            raise ValueError('Pomegranate dropped some values during fitting. Most likely because '
                                             'it contains a zero probability. Check your training data.')
                else:
                    cardinality_values = []
                    for key, _ in state.distribution.keymap.items():
                        c = key[state.distribution.m]
                        if c in cardinality_values:
                            break
                        cardinality_values.append(c)

                    cardinality = len(cardinality_values)
                    if cardinality != self.cardinality_map[state.name]:
                        raise ValueError('Pomegranate cardinality does not match expected cardinality of node %s',
                                         state.name)

                    parameters = state.distribution.parameters[0]
                    param_len = len(parameters)

                    matrix = np.empty(shape=(cardinality, int(param_len / cardinality)))
                    for x in range(0, param_len, cardinality):
                        for y in range(0, cardinality):
                            row = parameters[x + y]
                            matrix[y, int(x / cardinality)] = row[len(row) - 1]

                    cpd = matrix.tolist()

                self.add_cpd(state.name, cpd)

            return pm_net

    def from_pomegranate(self, pm_net):
        pass

    def to_json(self, separators=(',', ' : '), indent=4):
        for k, v in self.cpds.items():
            if isinstance(v, np.ndarray):
                self.cpds[k] = v.tolist()
        return json.dumps(self.make_network(), separators=separators, indent=indent, default=converter)

    def from_json(self, obj):
        header = obj['header']
        self.identification = header['identification']
        self.description = header['description']
        self.train_data = header['train_from']
        self.frozen = header['frozen']
        self.date = header['date']

        nodes = obj['nodes']

        ron_nodes = nodes['RON']
        self.bel_nodes = ron_nodes['BEL']
        self.mem_nodes = ron_nodes['MEM']

        self.lan_nodes = nodes['LAN']

        len_nodes = nodes['LEN']
        self.ext_nodes = len_nodes['EXT']
        self.int_nodes = len_nodes['INT']
        self.pro_nodes = len_nodes['PRO']

        self.edges = obj['edges']
        self.cpds = obj['cpds']

        return self

    def get_nodes(self):
        nodes = [[x['name'] for x in self.bel_nodes],
                 [x['name'] for x in self.mem_nodes],
                 [x['name'] for x in self.lan_nodes],
                 [x['name'] for x in self.ext_nodes],
                 [x['name'] for x in self.int_nodes],
                 [x['name'] for x in self.pro_nodes]]
        return [item for sublist in nodes for item in sublist]

    def get_root_nodes(self):
        roots = [[node['name'] for node in self.bel_nodes],
                 [node['name'] for node in self.mem_nodes]]
        return [item for sublist in roots for item in sublist]

    def get_leaf_nodes(self):
        leaves = [[node['name'] for node in self.ext_nodes],
                  [node['name'] for node in self.int_nodes],
                  [node['name'] for node in self.pro_nodes]]
        return [item for sublist in leaves for item in sublist]

    def get_lan_nodes(self):
        lans = [[node['name'] for node in self.lan_nodes]]
        return [item for sublist in lans for item in sublist]

    def get_pro_nodes(self):
        pros = [[node['name'] for node in self.pro_nodes]]
        return [item for sublist in pros for item in sublist]

    def add_belief_node(self, node, cardinality):
        self.bel_nodes.append({'name': node, 'card': cardinality})
        self.cardinality_map.update({node: cardinality})

    def add_memory_node(self, node, cardinality):
        self.mem_nodes.append({'name': node, 'card': cardinality})
        self.cardinality_map.update({node: cardinality})

    def remove_belief_node(self, node):
        self.bel_nodes = [n for n in self.bel_nodes if n['name'] != node]
        self.remove_node(node)

    def remove_node(self, node):
        self.cpds.pop(node, None)
        self.cardinality_map.pop(node, None)
        self.omega_map.pop(node, None)
        for parent in self.get_incoming_edges(node):
            self.remove_edge((parent, node))
        for child in self.get_outgoing_edges(node):
            self.remove_edge((node, child))

    def get_edges(self):
        return self.edges

    def get_incoming_edges(self, node):
        edges = [nod[0] for nod in self.edges if (nod[1] == node)]
        return edges

    def get_outgoing_edges(self, node):
        edges = [nod[1] for nod in self.edges if (nod[0] == node)]
        return edges

    def set_edges(self, edges):
        self.edges = edges

    def add_edge(self, edge):
        self.edges.append(edge)

    def remove_edge(self, edge):
        self.edges.remove(edge)

    def get_cpds(self, node=None):
        if node:
            return self.cpds[node]
        else:
            return self.cpds

    def set_cpds(self, cpds):
        self.cpds = cpds

    def add_cpd(self, node, cpd):
        self.cpds.update({node: cpd})

    def make_cardinality_map(self):
        return {node['name']: node['card'] for node in itertools.chain(self.bel_nodes, self.mem_nodes, self.lan_nodes,
                                                                       self.ext_nodes, self.int_nodes, self.pro_nodes)}

    def get_cardinality_map(self):
        return self.cardinality_map

    def make_omega_map(self):
        omg_map = {}
        for node in itertools.chain(self.bel_nodes, self.mem_nodes, self.lan_nodes,
                                    self.ext_nodes, self.int_nodes, self.pro_nodes):
            parents_card = [self.cardinality_map[parent] for parent in self.get_incoming_edges(node['name'])]
            max_omega = 2 * math.pi * np.prod(parents_card)
            omega = np.random.rand(node['card']) * max_omega
            omg_map.update({node['name']: omega})

        return omg_map

    def get_omega_map(self):
        return self.omega_map

    def add_omega(self, node, omega):
        self.omega_map.update({node: omega})

    def make_network(self):
        return {
            'header': {
                'identification': self.identification,
                'description': self.description,
                'frozen': self.frozen,
                'train_from': self.train_from,
                'date': self.date,
            },
            'nodes': {
                'RON': {
                    'BEL': self.bel_nodes,
                    'MEM': self.mem_nodes
                },
                'LAN': self.lan_nodes,
                'LEN': {
                    'EXT': self.ext_nodes,
                    'INT': self.int_nodes,
                    'PRO': self.pro_nodes,
                }
            },
            'edges': self.edges,
            'cpds': self.cpds
        }

    def __copy__(self):
        return PeepoNetwork(identification=self.identification,
                            description=self.description,
                            train_from=self.train_from,
                            train_data=self.train_data,
                            frozen=self.frozen,
                            bel_nodes=self.bel_nodes.copy(),
                            mem_nodes=self.mem_nodes.copy(),
                            lan_nodes=self.lan_nodes.copy(),
                            ext_nodes=self.ext_nodes.copy(),
                            int_nodes=self.int_nodes.copy(),
                            pro_nodes=self.pro_nodes.copy(),
                            edges=self.edges.copy(),
                            cpds=deepcopy(self.cpds))

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return self.to_json()
