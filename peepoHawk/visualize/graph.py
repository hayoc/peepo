
import os
#20/11/2018
import networkx as nx
import matplotlib.pyplot as plt
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
    plt.figure(figsize=(10, 5))
    G = nx.DiGraph()

    for node in network.nodes():
        card = network.get_cardinality(node)
        G.add_node(node, name=node, cpd=str("dim = ")+str(card))

    G.add_edges_from(network.edges())
    pos = nx.circular_layout(G, scale=2)

    nx.draw(G, pos, node_size=1200, node_color='lightblue',
            linewidths=0.25, font_size=10, font_weight='bold', with_labels=True)

    plt.show()

    '''d = json_graph.node_link_data(G)  # node-link format to serialize

    # write json
    json.dump(d, open(path + '/static/peepo.json', 'w'))
    print('Wrote node-link JSON data to static/peepo.json')'''

'''from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

def draw_network(network):
    # Colors.
    p_color = {"ec": "#46a546"}
    s_color = {"ec": "#f89406"}

    pgm = daft.PGM([3.6, 3.5], origin=[0.7, 0])

    n = daft.Node("phi", r"$\phi$", 1, 3, plot_params=s_color)
    n.va = "baseline"
    pgm.add_node(n)
    pgm.add_node(daft.Node("speckle_coeff", r"$z_i$", 2, 3, plot_params=s_color))
    pgm.add_node(daft.Node("speckle_img", r"$x_i$", 2, 2, plot_params=s_color))

    pgm.add_node(daft.Node("spec", r"$s$", 4, 3, plot_params=p_color))
    pgm.add_node(daft.Node("shape", r"$g$", 4, 2, plot_params=p_color))
    pgm.add_node(daft.Node("planet_pos", r"$\mu_i$", 3, 3, plot_params=p_color))
    pgm.add_node(daft.Node("planet_img", r"$p_i$", 3, 2, plot_params=p_color))

    pgm.add_node(daft.Node("pixels", r"$y_i ^j$", 2.5, 1, observed=True))

    # Edges.
    pgm.add_edge("phi", "speckle_coeff")
    pgm.add_edge("speckle_coeff", "speckle_img")
    pgm.add_edge("speckle_img", "pixels")

    pgm.add_edge("spec", "planet_img")
    pgm.add_edge("shape", "planet_img")
    pgm.add_edge("planet_pos", "planet_img")
    pgm.add_edge("planet_img", "pixels")

    # And a plate.
    pgm.add_plate(daft.Plate([1.5, 0.2, 2, 3.2], label=r"exposure $i$",
                             shift=-0.1))
    pgm.add_plate(daft.Plate([2, 0.5, 1, 1], label=r"pixel $j$",
                             shift=-0.1))

    # Render and save.
    pgm.render()
    pgm.figure.savefig("exoplanets.pdf")
    pgm.figure.savefig("exoplanets.png", dpi=150)'''

