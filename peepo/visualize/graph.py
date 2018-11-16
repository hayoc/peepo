import json
import os
import networkx as nx
from networkx.readwrite import json_graph


path = os.path.dirname(os.path.realpath(__file__))


def draw_network(network, block=False):
    """
    To view the results of the drawing, run this script separately: peepo/visualize/server.py
    and go to: http://localhost:8000/force.html

    :param network:
    :param block:
    :return:
    """
    G = nx.DiGraph()

    for node in network.nodes():
        G.add_node(node, name=node, cpd=str(network.get_cpds(node).values))

    G.add_edges_from(network.edges())

    pos_nodes = nx.spring_layout(G)
    nx.draw(G, pos_nodes, node_color='#a0cbe2', edge_color='#a1a9ad', with_labels=True)

    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.1)

    node_attrs = nx.get_node_attributes(G, 'cpd')
    custom_node_attrs = {}
    for node, attr in node_attrs.items():
        custom_node_attrs[node] = str(attr)

    nx.draw_networkx_labels(G, pos_attrs, labels=custom_node_attrs, font_size=6)

    d = json_graph.node_link_data(G)  # node-link format to serialize

    # write json
    json.dump(d, open(path + '/force/force.json', 'w'))
    print('Wrote node-link JSON data to force/force.json')




