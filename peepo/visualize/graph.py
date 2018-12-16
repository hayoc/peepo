import json
import logging
import os

import networkx as nx
from networkx.readwrite import json_graph

path = os.path.dirname(os.path.realpath(__file__))


def draw_network(network):
    """
    To view the results of the drawing, run this script separately: peepo/visualize/server.py
    and go to: http://localhost:8000/peepo.html

    :param network:
    :param block:
    :return:
    """
    G = nx.DiGraph()

    for node in network.nodes():
        G.add_node(node, name=node, cpd=str(network.get_cpds(node).values))

    G.add_edges_from(network.edges())

    d = json_graph.node_link_data(G)  # node-link format to serialize

    # write json
    json.dump(d, open(path + '/static/peepo.json', 'w'))
    logging.debug('Wrote node-link JSON data to static/peepo.json')
