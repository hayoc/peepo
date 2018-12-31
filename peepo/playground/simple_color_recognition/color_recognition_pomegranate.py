#28/12/2018
import logging
import math
import os
import random
import sys
import copy
import networkx as nx
import json
import numpy as np
from peepo.predictive_processing.v3.peepo_network import PeepoNetwork
from peepo.predictive_processing.v3.utils import *
import matplotlib.pyplot as plt
from pomegranate import *

from config import ROOT_DIR

class MyClass(object):
    def __init__(self, case):
        self.case = case
        self.results = []
        self.networx = nx.DiGraph()
        self.best_error =   math.inf
        self.best_topology = [0, 0, nx.DiGraph, 0]  # [error, entropy, networkx DiGraph, loop]
        self.nodes = []
        self.edges = {}
        self.colors_table = []
        self.colors_cpd = []

    def get_my_colors(self,root_nodes):
        cardinality = []
        evidence = []
        for i, node in enumerate(root_nodes):
            evidence.append(node['name'])
            cardinality.append(node['card'])
        self.colors_table, self.colors_cpd = self.color_cpd('', 3, evidence,cardinality)
        self.number_of_colors = self.colors_table.shape[1]

    def color_cpd(self, var, card_var, evidence, cardinality):
        table = get_index_matrix(cardinality)
        hi = 1
        lo = 0
        C = np.prod(cardinality)
        matrix = np.full((3, C), 1. / 3.)
        matrix[0] = [hi, lo, lo, hi, lo, lo, hi, lo, hi, lo, lo, hi, lo, lo, hi, lo]
        matrix[1] = [lo, hi, lo, lo, hi, lo, lo, hi, lo, hi, lo, lo, hi, lo, lo, hi]
        matrix[2] = [lo, lo, hi, lo, lo, hi, lo, lo, lo, lo, hi, lo, lo, hi, lo, lo]
        return table.astype(int), matrix.astype(int)

    def create_learning_data(self, network):
        ben_nodes = network.bel_nodes
        self.get_my_colors(ben_nodes)
        learning_data = []
        for i, node in enumerate(ben_nodes):
            learning_data.append(self.colors_table[i].tolist())
        shape = self.colors_cpd.shape
        reshaped_cpd = self.colors_cpd.reshape(shape[0], int(np.prod(shape) / shape[0]))
        for hue in range(0, 3):
            learning_data.append(reshaped_cpd[hue, :].tolist())
        learning_data = np.asarray(learning_data).transpose()
        return learning_data.tolist()


    def do_it(self):
        '''EXPLANATIONS'''
        with open(ROOT_DIR + '/resources/' + self.case + '.json') as json_data:
            json_object = json.load(json_data)
            print(json_object)
        peepo = PeepoNetwork()
        network = peepo.from_json(json_object)
        network.train_data = self.create_learning_data(network)
        possible_topologies = get_topologies(network, 20)
        print("Possible topologies : ", len(possible_topologies))
        ''' -------------- walking through all toplogies'''
        loop = 0
        for topology in possible_topologies:
            entropy = topology['entropy']
            if entropy == 0:
                continue  # safeguard
            print('Loop *-> ', loop + 1, ' of ', len(possible_topologies))
            print(topology)
            # if loop > 200:
            #     loop += 1
            #     continue
            network.edges = topology['edges']
            network.cpds = {}
            network.assemble()
            # scores = []
            # for i in range(0, len(test_data)):
            #     state = np.asarray(test_data[i])
            #     score = self.pommy.score(learning_data, state)
            #     scores.append(score)
            #
            # mean_score = np.mean(np.asarray(scores))
            '''-------------- Testing the constructed topology by using an MDL typ score'''

            MDL_score = math.log(len(network.train_data )) * network.pomegranate_network.state_count() / 2 \
                        - np.mean(network.pomegranate_network.log_probability(network.train_data ))
            print('MDL score : ', MDL_score)
            # MDL_score = mean_score
            self.results.append([entropy, MDL_score])
            if MDL_score <= self.best_error:
                self.best_error = MDL_score
                self.best_topology[0] = MDL_score
                self.best_topology[1] = entropy
                nwx = nx.DiGraph()
                nwx.add_edges_from(network.edges)
                self.best_topology[2] = nwx
                self.best_topology[3] = loop
            loop += 1
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
    case = 'color_recognition'
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