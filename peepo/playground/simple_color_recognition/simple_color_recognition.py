import math
import os
import random
import sys
import networkx as nx
import numpy as np
import pygame as pg
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from peepo.utilities.utilities import  Utilities
from peepo.utilities.lattices import  Lattices
from peepo.playground.simple_color_recognition.CeePeeDees import CPD
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput
import matplotlib.pyplot as plt


class MyClass(object):
    def __init__(self, case):
        self.case = case
        self.results = []
        self.networx_0 = nx.DiGraph()
        self.pgmpy_0  = BayesianModel()
        self.networx = nx.DiGraph()
        self.pgmpy = BayesianModel()
        self.dictionary = []
        self.header = {}
        self.nodes_0 = []
        self.edges_0 = {}
        self.nodes = []
        self.edges = {}
        self.cpds = {}
        self.colors_dictionary ={}
        self.colors_table =[]
        self.colors_cpd = []
        self.nummber_of_colors = 0
        self._util = Utilities(case)
        self._lat = Lattices(self._util)

    def get_my_colors(self):
        evidence = []
        cardinality = []
        for i, node in enumerate(self.nodes):
            if 'BEN' in node[0] or 'MEM' in node[0]:
                evidence.append(node[0])
                cardinality.append(node[1]['cardinality'])
        self.colors_dictionary, self.colors_table, self.colors_cpd = CPD.color_cpd('hue',3,evidence,cardinality)
        self.number_of_colors = self.colors_table.shape[1]
        print('Number of colors : ', self.number_of_colors)
        print(self.colors_table)
        print(self.colors_cpd)


    def add_edges(self, topology):
        self.networx.remove_edges_from(self.edges)
        self.edges = []
        ''' lest first remove all void nodes'''
        shape = np.asarray(topology).shape
        nodes_to_remove = []
        rows = np.sum(topology, axis = 1)
        columns = np.sum(topology,axis = 0)
        for row in range(0, len(rows)):
            if rows[row] == 0:
                nodes_to_remove.append('WORLD_' + str(row))
        for column in range(0, len(columns)):
            if columns[column] == 0:
                nodes_to_remove.append('BENS_' + str(column))
        self.networx.remove_nodes_from(nodes_to_remove)
        self.nodes = self.networx.nodes(data = True)
        for column in range(0,shape[1]):
            for row in range(0,shape[0]):
                if topology[row][column] == 1:
                    parent = 'BENS_' + str(column)
                    child  = 'WORLD_'+ str(row)
                    self.networx.add_edge(parent, child)
        self.edges = self.networx.edges()


    def add_dummy_cpds(self):
        for i, node in enumerate(self.nodes):
            cardinality = node[1]['cardinality']
            if ('BEN' in node[0]) or ('MEM' in node[0]):
                self.nodes[i][1]['cpd'] = CPD.create_fixed_parent(cardinality, 0)
            else:
                incoming_nodes = self.networx.in_edges(node[0])
                if len(incoming_nodes) == 0:
                    continue
                card_parent = []
                for  m, n in enumerate(incoming_nodes):
                    par = self.networx.node[n[0]]['cardinality']
                    card_parent.append(par)
                self.nodes[i][1]['cpd'] = CPD.create_random_child(cardinality, card_parent)





    def do_it(self):
        '''EXPLANATIONS'''
        self.networx_0, self.dictionary, self.header = self._util.get_network()
        self.nodes_0 = self.networx_0.nodes(data=True)
        self.networx = self.networx_0.copy()
        self.nodes = self.nodes_0.copy()
        self.get_my_colors()
        print(self.nodes_0)
        print(self.header)
        print(self.dictionary)
        possible_topologies  = self._lat.get_possible_topologies()
        print("Possible topologies : ", len(possible_topologies))
        entropy = 0
        count = 0#TEMPORARY
        for topology in possible_topologies:
            entropy = topology[1]
            if entropy == 0:
                continue
            topo  = topology[0]
            self.networx = self.networx_0.copy()
            edges = []
            parent = ''
            child = ''
            self.add_edges(topo)
            self.add_dummy_cpds()
            self.pgmpy = self._util.translate_digraph_to_pgmpy(self.networx)
            self.pgmpy.check_model()
            '''now looping through all the colors'''
            for colors  in range(0, self.number_of_colors):
                color = self.colors_table[:,colors]
                print('Color : ' ,color)
            '''following  4 lines to remove : just use to check whether the algorithms are correct regarding the edges building'''
            count += 1
            #print('edges : ', self.edges)
            if count > 0:
                break
        print('Check -> number of processed topologies in loop : ', count)
        # print('My colors : ')
        # print(self.colors_table)
        # print(self.colors_cpd)
        '''TO DO ----------------------------------------------------
                a) add random cpds , convert to pgmpy BN, 
                b) enbedd the skeleton loop  within the learning loop->
                    loop through all possible colors and the expected classification
                    -- > for each skeleton with the possible color as BEN, make  pgmpy guess the best cpd's 
                         with the method class 
                                   in pgmpy.estimators.BayesianEstimator.BayesianEstimator(model, data, **kwargs)[source]
                                            estimate_cpd(node, prior_type='BDeu', pseudo_counts=[], equivalent_sample_size=5)[source]
                    -- > make inference and calulate the 'error (to be determined)
                    ---> log the error as a tuple (error, 'entropy of the skeleton')
                c) create output (grapgh?)
                    
            
            
            '''



        '''  the methods have to be completed to cope with a general case i.e. BENS,MEMS,LANS, MOTORs, WORLDs
        but for the moment being we just assume there are only BEN's and WORLD's'''

        # self.networx.add_edge('BENS_1','WORLD_1')
        # self.networx.node['BENS_1']['cpd'] = [0.8,0.2]
        # self.networx.node['WORLD_2']['cpd'] = [[0.8, 0.2, 0.5,0.3],[0.2,0.8,0.5,0.7]]
        ''' if a best model has ben found, save it -> first update the Utility class object and save it'''
        # self._util.update_networkx(self.networx, self.dictionary, self.header)
        # self._util.save_network()
        # self._util.update_pgmpy(self.pgmpy, self.dictionary, self.header)
        # self._util.save_pgmpy_network()
        self.draw()
        return self.results


    def draw(self):
        '''TO REMOVE LATER'''
        plt.figure(figsize=(10, 5))
        pos = nx.circular_layout(self.networx, scale=2)
        #node_labels = nx.get_node_attributes(self.networx, 'cpd')
        nx.draw(self.networx, pos, node_size=1200, node_color='lightblue',
                linewidths=0.25,  font_size=10, font_weight='bold', with_labels=True)
        plt.show()

def main():
    case = 'simple_color_recognition'
    mycase = MyClass(case)
    results = mycase.do_it()
    print(results)

####################################################################################
############################### BEGIN HERE #########################################
####################################################################################

if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.INFO)
    main()