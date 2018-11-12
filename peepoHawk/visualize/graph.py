import matplotlib.pyplot as plt
import networkx as nx


def draw_network(network, block=False):

    plt.figure(figsize=(15, 8))
    plt.clf()
    G = nx.DiGraph()

    for node in network.nodes():
        G.add_node(node, cpd=network.get_cpds(node))

    G.add_edges_from(network.edges())

    pos_nodes = nx.circular_layout(G)
    nx.draw(G, pos_nodes, node_color='#a0cbe2', edge_color='#a1a9ad', with_labels=True)
    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.1)

    node_attrs = nx.get_node_attributes(G, 'cpd')
    custom_node_attrs = {}
    for node, attr in node_attrs.items():
        custom_node_attrs[node] = str(attr)

    #nx.draw_networkx_labels(G, pos_attrs, labels=custom_node_attrs, font_size=6)
    #nx.draw_networkx_labels(G, pos_attrs, font_size=6)
    plt.pause(1)

    plt.show(block=block)
