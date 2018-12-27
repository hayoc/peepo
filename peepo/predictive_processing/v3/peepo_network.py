import datetime as dt
import json

import numpy as np
from pomegranate.BayesianNetwork import BayesianNetwork
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution


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
        self.network = {}
        self.pomegranate_network = None

    def assemble(self):
        self.pomegranate_network = self.to_pomegranate()
        self.network = {
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

    def to_pomegranate(self):
        if self.cpds:
            pass  # TODO
        else:
            structure = []
            nodes = self.get_nodes()
            for node in nodes:
                parents = []
                for edge in self.edges:
                    if node == edge[1]:
                        parents.append(nodes.index(edge[0]))
                structure.append(tuple(parents))
            pm_net = BayesianNetwork.from_structure(self.train_data, tuple(structure))

            for i, state in enumerate(pm_net.states):
                state.name = nodes[i]

                if isinstance(state.distribution, DiscreteDistribution):
                    nodevalue = []
                    # TODO: Check whether there's always only one... Why a list of dicts anyway... Pomegranate sucks
                    parameter = state.distribution.parameters[0]
                    for key in sorted(parameter.keys()):
                        nodevalue.append(parameter[key])
                else:
                    parameters = state.distribution.parameters[0]
                    param_len = len(parameters)
                    node_cardinality = len(parameters[0]) - state.distribution.m

                    matrix = np.empty(shape=(node_cardinality, int(param_len / node_cardinality)))
                    for x in range(0, param_len, node_cardinality):
                        for y in range(0, node_cardinality):
                            row = parameters[node_cardinality + y]
                            matrix[y: int(x / node_cardinality)] = row[len(row) - 1]

                    nodevalue = matrix.tolist()

                self.add_cpd(state.name, nodevalue)

            return pm_net

    def from_pomegranate(self, pm_net):
        pass

    def to_json(self, separators=(',', ' : '), indent=4):
        return json.dumps(self.network, separators=separators, indent=indent)

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

    def get_edges(self):
        return self.edges

    def set_edges(self, edges):
        self.edges = edges

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_cpds(self, node=None):
        if node:
            return self.cpds[node]
        else:
            return self.cpds

    def set_cpds(self, cpds):
        self.cpds = cpds

    def add_cpd(self, node, cpd):
        self.cpds.update({node: cpd})

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return self.to_json()
