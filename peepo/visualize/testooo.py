from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import networkx as nx
import matplotlib.pyplot as plt


def create_model():
    network = BayesianModel([('A', 'B'), ('A', 'C')])
    cpd_a = TabularCPD(variable='A',
                       variable_card=2,
                       values=[[0.9, 0.1]])

    cpd_b = TabularCPD(variable='B',
                       variable_card=2,
                       values=[[0.99, 0.01],
                               [0.01, 0.99]],
                       evidence=['A'],
                       evidence_card=[2])

    cpd_c = TabularCPD(variable='C',
                       variable_card=2,
                       values=[[0.99, 0.01],
                               [0.01, 0.99]],
                       evidence=['A'],
                       evidence_card=[2])

    network.add_cpds(cpd_a, cpd_b, cpd_c)
    network.check_model()

    return network


model = create_model()
G = nx.DiGraph()

for node in model.nodes():
    G.add_node(node, cpd=model.get_cpds(node))

G.add_edges_from(model.edges())


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
plt.show()






