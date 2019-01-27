import json
import os

import networkx as nx
from networkx.readwrite import json_graph

path = os.path.dirname(os.path.realpath(__file__))


def pretty_str(cpd):
    result = ',\n'.join(str(sublist) for sublist in cpd)
    return result


def draw_network(peepo_network):
    """
    To view the results of the drawing, run this script separately: peepo/visualize/server.py
    and go to: http://localhost:8000/peepo.html

    :param peepo_network:
    :return:
    """
    G = nx.DiGraph()

    for node in peepo_network.get_nodes():
        G.add_node(node, name=node, cpd=pretty_str(peepo_network.get_cpds(node)))

    G.add_edges_from(peepo_network.get_edges())

    d = json_graph.node_link_data(G)

    # write json
    json.dump(d, open(path + '/static/peepo.json', 'w'))
