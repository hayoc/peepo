import logging
import math
import os
import random
import sys
import copy
import networkx as nx
import json
import numpy as np
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from peepo.utilities.utilities import Utilities
from peepo.utilities.lattices import Lattices
from peepo.playground.simple_color_recognition.CeePeeDees import CPD
from peepo.predictive_processing.v3.sensory_input import SensoryInput
import matplotlib.pyplot as plt
from scipy.stats import entropy
from pomegranate import *
import seaborn, time


class MyClass(object):
    def __init__(self, case):
        self.case = case
        self.results = []
        self.networx_test = nx.DiGraph()
        self.networx_fixed = nx.DiGraph()
        self.pommy_test = BayesianNetwork()
        self.networx = nx.DiGraph()
        self.pommy = BayesianNetwork()
        self.best_error =   math.inf
        self.best_topology = [0, 0, nx.DiGraph, 0]  # [error, entropy, networkx DiGraph, loop]
        self.dictionary = []
        self.header = {}
        self.nodes_0 = []
        self.edges_0 = {}
        self.nodes = []
        self.edges = {}
        self.cpds = {}
        self.colors_dictionary = {}
        self.colors_table = []
        self.colors_cpd = []
        self.nummber_of_colors = 0
        self._util = Utilities(case)
        self._lat = Lattices(self._util)
        self.expected_result = [0, 0, 0]
        self.loop = 0
        self.summary = {}
        self.pom_nodes = {}
        self.summary_test = {}
        self.pom_nodes_test = {}
        self.pixel_states = {'RON_BEN_0': 0, 'RON_BEN_1': 0, 'RON_BEN_2': 0, 'RON_BEN_3': 0}
        self.all_pixel_states = {}

    def get_my_colors(self):
        evidence = []
        cardinality = []
        for i, node in enumerate(self.nodes):
            if 'BEN' in node[0] or 'MEN' in node[0]:
                evidence.append(node[0])
                cardinality.append(node[1]['cardinality'])
        self.colors_dictionary, self.colors_table, self.colors_cpd = self.color_cpd('LEN_WORLD', 3, evidence,
                                                                                    cardinality)
        self.number_of_colors = self.colors_table.shape[1]

    def color_cpd(self, var, card_var, evidence, cardinality):
        self.all_pixel_states = {}
        table = CPD.get_index_matrix(cardinality)
        for i in range(0, len(table[1])):
            self.all_pixel_states.update({i: []})
            a_dic = {}
            for j in range(0, len(evidence)):
                a_dic.update({evidence[j]: table[j][i]})
            self.all_pixel_states[i] = a_dic
        # print('*********************************************self.all_pixel_states')
        # print(self.all_pixel_states)
        colors = {}
        hi = 2  # 0.999
        lo = 0
        C = np.prod(cardinality)
        average = 1
        matrix = np.full((3, C), average)
        if 'RON_BEN_1' in evidence and not 'RON_BEN_2' in evidence and 'RON_BEN_3' in evidence and 'RON_BEN_0' in evidence:
            matrix[0] = [average, lo, hi, average, average, lo, hi, average]
            matrix[1] = [average, lo, lo, average, average, lo, lo, average]
            matrix[2] = [average, hi, lo, average, average, hi, lo, average]
        if 'RON_BEN_1' in evidence and not 'RON_BEN_2' in evidence and 'RON_BEN_3' in evidence and not 'RON_BEN_0' in evidence:
            matrix[0] = [average, lo, hi, average]
            matrix[1] = [average, lo, lo, average]
            matrix[2] = [average, hi, lo, average]

        if not 'RON_BEN_1' in evidence and 'RON_BEN_2' in evidence and 'RON_BEN_3' in evidence and 'RON_BEN_0' in evidence:
            matrix[0] = [average, lo, hi, average, average, lo, hi, average]
            matrix[1] = [average, lo, lo, average, average, lo, lo, average]
            matrix[2] = [average, hi, lo, average, average, hi, lo, average]
        if not 'RON_BEN_1' in evidence and 'RON_BEN_2' in evidence and 'RON_BEN_3' in evidence and not 'RON_BEN_0' in evidence:
            matrix[0] = [average, lo, hi, average]
            matrix[1] = [average, lo, lo, average]
            matrix[2] = [average, hi, lo, average]

        if 'RON_BEN_1' in evidence and 'RON_BEN_2' in evidence and not 'RON_BEN_3' in evidence and 'RON_BEN_0' in evidence:
            matrix[0] = [average, lo, hi, average, average, lo, hi, average]
            matrix[1] = [average, lo, lo, average, average, lo, lo, average]
            matrix[2] = [average, hi, lo, average, average, hi, lo, average]
        if 'RON_BEN_1' in evidence and 'RON_BEN_2' in evidence and not 'RON_BEN_3' in evidence and not 'RON_BEN_0' in evidence:
            matrix[0] = [average, lo, hi, average]
            matrix[1] = [average, lo, lo, average]
            matrix[2] = [average, hi, lo, average]

        if 'RON_BEN_1' in evidence and 'RON_BEN_2' in evidence and 'RON_BEN_3' in evidence and not 'RON_BEN_0' in evidence:
            matrix[0] = [lo, lo, lo, lo, hi, lo, hi, lo]
            matrix[1] = [hi, lo, hi, lo, lo, hi, lo, hi]
            matrix[2] = [lo, hi, lo, hi, lo, lo, lo, lo]
        if 'RON_BEN_0' in evidence and 'RON_BEN_1' in evidence and 'RON_BEN_2' in evidence and 'RON_BEN_3' in evidence:
            matrix[0] = [lo, lo, lo, lo, hi, lo, hi, lo, lo, lo, lo, lo, hi, lo, hi, lo]
            matrix[1] = [hi, lo, hi, lo, lo, hi, lo, hi, hi, lo, hi, lo, lo, hi, lo, hi]
            matrix[2] = [lo, hi, lo, hi, lo, lo, lo, lo, lo, hi, lo, hi, lo, lo, lo, lo]
        for i, node in enumerate(evidence):
            colors.update({node: table[i]})
        return colors, table.astype(int), matrix.astype(int)

    def create_learning_data(self):
        self.get_my_colors()
        learning_data = []
        #learning_labels = [x for x in self.networx.nodes() ]
        ben_nodes = [x for x in self.nodes if "RON_BEN" in x[0]]
        world_nodes = [x for x in self.nodes if "LEN_WORLD" in x[0]]

        for i, node in enumerate(ben_nodes):
            learning_data.append(self.colors_table[i].tolist())
        for i, node in enumerate(world_nodes):
            shape = self.colors_cpd.shape
            reshaped_cpd = self.colors_cpd.reshape(shape[0], int(np.prod(shape) / shape[0]))
            for hue in range(0, 3):
                if str(hue) in node[0]:
                    learning_data.append(reshaped_cpd[hue, :].tolist())
        learning_data = np.asarray(learning_data).transpose()
        learning_labels = []
        for i in range(0,len(learning_data)):
            learning_labels.append(True)
        for i in range(0,len(learning_data)):
            learning_labels[i] = np.asarray(learning_labels[i]).tolist()
        # print('learningng labels')
        # print(learning_labels)
        # print('learning data')
        # print(learning_data.tolist())
        learning_lables = np.asarray(['1'])
        return learning_data.tolist(), learning_labels

    def add_edges(self, topology):
        self.networx.remove_edges_from(self.edges)
        self.edges = []
        shape = np.asarray(topology).shape
        ''' let's first remove all void nodes  ----> not necssary -----> delete the code ??'''
        nodes_to_remove = []
        # rows = np.sum(topology, axis = 1)
        # for row in range(0, len(rows)):
        #     if rows[row] == 0:
        #         nodes_to_remove.append('WORLD_' + str(row))
        columns = np.sum(topology, axis=0)
        for column in range(0, len(columns)):
            if columns[column] == 0:
                nodes_to_remove.append('RON_BEN_' + str(column))
        self.networx.remove_nodes_from(nodes_to_remove)
        self.nodes = self.networx.nodes(data=True)
        for column in range(0, shape[1]):
            for row in range(0, shape[0]):
                if topology[row][column] == 1:
                    parent = 'RON_BEN_' + str(column)
                    child = 'LEN_WORLD_' + str(row)
                    self.networx.add_edge(parent, child)
        self.edges = self.networx.edges()

    def add_dummy_cpds(self):
        for i, node in enumerate(self.nodes):
            cardinality = node[1]['cardinality']
            if ('BEN' in node[0]) or ('MEN' in node[0]):
                self.nodes[i][1]['cpd'] = CPD.create_fixed_parent(cardinality, modus='uniform')
            else:
                incoming_nodes = self.networx.in_edges(node[0])
                if len(incoming_nodes) == 0:
                    self.nodes[i][1]['cpd'] = CPD.create_random_child(cardinality, modus='orphan')
                    continue
                card_parent = []
                for m, n in enumerate(incoming_nodes):
                    par = self.networx.node[n[0]]['cardinality']
                    card_parent.append(par)
                self.nodes[i][1]['cpd'] = CPD.create_random_child(cardinality, card_parent)

    def update_network(self):
        ''' update the data associated with the nodes'''
        for i, node in enumerate(self.nodes):
            self.nodes[i][1]['childs'] = []
            self.nodes[i][1]['parents'] = []
            self.nodes[i][1]['parents_cardinality'] = []
            for k, edge in enumerate(self.networx.edges()):
                if node[0] == edge[0]:
                    self.nodes[i][1]['childs'].append(edge[1])
            cardinality = node[1]['cardinality']
            if ('BEN' in node[0]) or ('MEM' in node[0]):
                self.nodes[i][1]['parents'] = []
            else:
                incoming_nodes = self.networx.in_edges(node[0])
                if len(incoming_nodes) == 0:
                    self.nodes[i][1]['parents'] = []
                    continue
                card_parent = []
                paren = []
                for m, n in enumerate(incoming_nodes):
                    if n[1] == node[0]:
                        par = self.networx.node[n[0]]['cardinality']
                        paren.append(n[0])
                        card_parent.append(par)
                    self.nodes[i][1]['parents_cardinality'] = card_parent

                    self.nodes[i][1]['parents'] = paren
        for node, out_degree in self.networx.out_degree_iter():
            if out_degree == 0:
                self.networx.node[node]['childs'] = []

        self.nodes = self.networx.nodes(data=True)

    def do_it(self):
        '''EXPLANATIONS'''
        self.networx_fixed, self.summary, self.dictionary, self.header = self._util.get_network()
        self.networx = self.networx_fixed.copy()
        self.networx_test = self.networx_fixed.copy()
        self.nodes = self.networx.nodes(data=True)
        print('Dictionary : ', self.dictionary)


        ''' -------------- Constructing all possible topologies,
                              --> option : restrain the number with the treshold :
                                        0 -> all possible topologies, 100 -> only the fully connnected topology'''
        possible_topologies = self._lat.get_possible_topologies(
            treshold=50)  # setting the entropy at a 50% -> only topologies with an entropy >= 0.5 will be considered
        print("Possible topologies : ", len(possible_topologies))
        entropy = 0
        count = 0  # TEMPORARY
        ''' -------------- walking through all toplogies'''
        for topology in possible_topologies:
            entropy = topology[1]
            if entropy == 0:
                continue  # safeguard
            print('Loop *-> ', self.loop + 1, ' of ', len(possible_topologies))
            topo = topology[0]
            self.networx = self.networx_fixed.copy()
            ''' ----------- for each topology we construct the edges and update dummy cpd (necessary as the shape of the LENs cpd's can change
                            depending on the number of incoming nodes'''
            self.add_edges(topo)
            # print('edges : ',self.edges)
            self.add_dummy_cpds()
            ''' update the data associated with the nodes'''
            self.update_network()
            learning_data, learning_labels = self.create_learning_data()
            ''' ----------- convert DiGraph topomegrante'''
            self.pommy, self.pom_nodes, self.summary = self._util.translate_digraph_to_pomegranate(self.networx)
            self.pommy.bake()
            '''------------ askpomegranate to guess the best cpd's of the LANs and LENs
                             -> provide pomegranate with the learning data'''
            self.pommy.fit(learning_data)
            self.update_network()

            '''-------------- Testing the constructed topology'''
            # score = self.pommy.score(learning_data, learning_labels)
            # print('score = ', score)
            MDL_score = math.log(len(learning_data))*self.pommy.state_count()/2 - np.sum(self.pommy.log_probability(learning_data,1))
            # print('score = ', MDL_score)
            self.results.append([entropy, MDL_score])
            if MDL_score <= self.best_error:
                self.best_error = MDL_score
                self.best_topology[0] = MDL_score
                self.best_topology[1] = entropy
                self.best_topology[2] = self.networx
                self.best_topology[3] = self.loop
            self.loop += 1
            '''following  4 lines to remove : just use to check whether the algorithms are correct regarding the edges building'''
            count += 1
            # print('edges : ', self.edges)
            # #
            # if count > 400:
            #     break
        self.draw()
        self.draw_xy()
        return self.results

    def draw_xy(self):
        x = []
        y = []
        s = []
        color = []
        best_x = 0
        best_y = 0
        for i in range(0, len(self.results)):
            x.append(self.results[i][0])
            y.append(self.results[i][1])
            if i == self.best_topology[3]:
                best_x = self.results[i][0]
                best_y = self.results[i][1]
                s.append(60)
                color.append("r")
            else:
                s.append(20)
                color.append("b")
        plt.scatter(x, y, s=s, c=color, alpha=0.5)
        plt.xlabel("Complexity of topology")
        plt.ylabel("Average error over all colors")
        plt.show()

    def draw(self):
        '''TO REMOVE LATER'''
        plt.figure(figsize=(10, 5))
        pos = nx.circular_layout(self.best_topology[2], scale=2)
        # node_labels = nx.get_node_attributes(self.networx, 'cpd')
        nx.draw(self.best_topology[2], pos, node_size=1200, node_color='lightblue',
                linewidths=0.25, font_size=10, font_weight='bold', with_labels=True)
        plt.text(1, 1, 'Topology nr. : ' + str(self.best_topology[3]))
        plt.show()


def main():
    case = 'simple_color_recognition'
    mycase = MyClass(case)
    results = mycase.do_it()
    # print(results)


####################################################################################
############################### BEGIN HERE #########################################
####################################################################################

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()