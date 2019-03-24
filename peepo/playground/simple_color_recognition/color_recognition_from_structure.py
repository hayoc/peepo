#28/12/2018
import logging
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pomegranate import *

from peepo.playground.simple_color_recognition.CeePeeDees import CPD
from peepo.utilities.lattices import Lattices
from peepo.utilities.utilities import Utilities


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
        hi = 1
        lo = 0
        C = np.prod(cardinality)
        matrix = np.full((3, C), 1. / 3.)
        matrix[0] = [hi, lo, lo, hi, lo, lo, hi, lo, hi, lo, lo, hi, lo, lo, hi, lo]
        matrix[1] = [lo, hi, lo, lo, hi, lo, lo, hi, lo, hi, lo, lo, hi, lo, lo, hi]
        matrix[2] = [lo, lo, hi, lo, lo, hi, lo, lo, lo, lo, hi, lo, lo, hi, lo, lo]
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
        for column in range(0, shape[1]):
            for row in range(0, shape[0]):
                if topology[row][column] == 1:
                    parent = 'RON_BEN_' + str(column)
                    child  = 'LEN_WORLD_' + str(row)
                    self.networx.add_edge(parent, child)
        self.edges = self.networx.edges()



    def do_it(self):
        '''EXPLANATIONS'''
        self.networx_fixed, self.summary, self.dictionary, self.header = self._util.get_network()
        self.networx = self.networx_fixed.copy()
        self.networx_test = self.networx_fixed.copy()
        self.nodes = self.networx.nodes(data=True)
        state_names = self.networx.nodes()
        print('Dictionary : ', self.dictionary)
        number_of_RONs = 0
        for node in self.nodes:
            if 'RON' in node[0]:
                number_of_RONs += 1


        ''' -------------- Constructing all possible topologies,
                              --> option : restrain the number with the treshold :
                                        0 -> all possible topologies, 100 -> only the fully connnected topology'''
        possible_topologies = self._lat.get_possible_topologies(
            treshold=50)  # setting the entropy at a 50% -> only topologies with an entropy >= 0.5 will be considered
        print("Possible topologies : ", len(possible_topologies))
        entropy = 0
        count = 0  # TEMPORARY
        learning_data, learning_labels = self.create_learning_data()
        test_data = []
        for a in learning_data:
            test_data.append(a)
        ''' -------------- walking through all toplogies'''
        for topology in possible_topologies:
            entropy = topology[1]
            if entropy == 0:
                continue  # safeguard
            print('Loop *-> ', self.loop + 1, ' of ', len(possible_topologies))
            # if self.loop > 2000:
            #     self.loop += 1
            #     count += 1
            #     continue
            topo = topology[0]
            self.networx = self.networx_fixed.copy()
            ''' ----------- for each topology we construct the edges and update dummy cpd (necessary as the shape of the LENs cpd's can change
                            depending on the number of incoming nodes'''
            self.add_edges(topo)
            # print('edges : ',self.edges)
            my_dic = {}
            states = self.networx.nodes()
            [my_dic.update({state:{'id': nr,'parents':[]}}) for nr, state in enumerate(states)]
            [my_dic[node[1]]['parents'].append(node[0]) for node in self.edges]
            structure = []
            for n, node in enumerate(states):
                parents = np.asarray(my_dic[node]['parents'])
                for n, par in enumerate(parents):
                    parents[n] =  int(my_dic[par]['id'])
                my_tuple = (parents.astype(int))
                if n >= number_of_RONs and len(my_tuple) < number_of_RONs:
                    my_tuple.append('')
                structure.append(tuple(my_tuple))
            structure = tuple(structure)
            print('my structure = ', structure)
            self.pommy = BayesianNetwork.from_structure(learning_data, structure, state_names = state_names)
            self.pommy.bake()
            # scores = []
            # for i in range(0, len(test_data)):
            #     state = np.asarray(test_data[i])
            #     score = self.pommy.score(learning_data, state)
            #     scores.append(score)
            #
            # mean_score = np.mean(np.asarray(scores))
            '''-------------- Testing the constructed topology'''
            MDL_score = 0*math.log(len(learning_data)) * self.pommy.state_count() / 2 - np.mean(self.pommy.log_probability(learning_data))
            # print('MDL score : ', MDL_score)
            # MDL_score = mean_score
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