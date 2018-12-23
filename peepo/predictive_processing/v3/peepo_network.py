import datetime as dt

import pandas
from pomegranate.BayesianNetwork import BayesianNetwork


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
        train_data : str, optional
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
        >>> print(pp_network.assemble())
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

    def assemble(self):
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
            nodes = self.nodes_to_list()
            for node in nodes:
                parents = []
                for edge in self.edges:
                    if node == edge[1]:
                        parents.append(nodes.index(edge[0]))
                structure.append(tuple(parents))

            return BayesianNetwork.from_structure(self.train_data_to_matrix(), tuple(structure))

    def from_pomegranate(self, pm_net):
        pass

    def to_json(self):
        return self.network

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

        self.assemble()

        return self

    def train_data_to_matrix(self):
        return pandas.DataFrame(self.train_data).values

    def nodes_to_list(self):
        nodes = [[x['name'] for x in self.bel_nodes],
                 [x['name'] for x in self.mem_nodes],
                 [x['name'] for x in self.lan_nodes],
                 [x['name'] for x in self.ext_nodes],
                 [x['name'] for x in self.int_nodes],
                 [x['name'] for x in self.pro_nodes]]
        return [item for sublist in nodes for item in sublist]
