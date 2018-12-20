#versie 19/12/2018
import math
import os
import random
import sys
import copy
import pandas as pd
import networkx as nx
import numpy as np
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from peepo.utilities.utilities import  Utilities
from peepo.utilities.lattices import  Lattices
from peepo.playground.simple_color_recognition.CeePeeDees import CPD
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput
import matplotlib.pyplot as plt


class SensoryInputVirtualPeepo(SensoryInput):
    def __init__(self, obj):
        super().__init__()
        self.peepo = obj

    def action(self, node, prediction):
        a = 0

    def value(self, name):
        expected_result = self.peepo.expected_result
        cpds = {}
        for i in range(0,len(expected_result)):
            cpds.update({'WORLD_'+str(i): CPD.create_fixed_parent(2, state = int(expected_result[i]))})
        for i, node in enumerate(self.peepo.nodes):
            if name == node[0]:
                return cpds[name]

class MyClass(object):
    def __init__(self, case):
        self.case = case
        self.results = []
        self.networx_test = nx.DiGraph()
        self.networx_fixed = nx.DiGraph()
        self.pgmpy_test  = BayesianModel()
        self.networx = nx.DiGraph()
        self.pgmpy = BayesianModel()
        self.best_error = math.inf
        self.best_topology = [0,0,nx.DiGraph]#[error, entropy, networkx DiGraph]
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
        self.learning_data = {}
        self.nummber_of_colors = 0
        self._util = Utilities(case)
        self._lat = Lattices(self._util)
        self.expected_result = [0, 0, 0]
        self.loop = 0

    def get_my_colors(self):
        evidence = []
        cardinality = []
        for i, node in enumerate(self.nodes):
            if 'BEN' in node[0] or 'MEM' in node[0]:
                evidence.append(node[0])
                cardinality.append(node[1]['cardinality'])
        self.colors_dictionary, self.colors_table, self.colors_cpd = self.color_cpd('WORLD',3,evidence,cardinality)
        self.number_of_colors = self.colors_table.shape[1]
        # print('Number of colors : ', self.number_of_colors)
        # print(self.colors_cpd)
        #print(self.colors_cpd.values)

    def color_cpd(self,var,card_var,evidence,cardinality):
        table = CPD.get_index_matrix(cardinality)
        colors ={}
        hi = 1
        lo = 0
        C = np.prod(cardinality)
        matrix = np.full((3, C), 1. / 3.)
        matrix[0] = [hi, lo, lo, hi, lo, lo, hi, lo, hi, lo, lo, hi, lo, lo, hi, lo]
        matrix[1] = [lo, hi, lo, lo, hi, lo, lo, hi, lo, hi, lo, lo, hi, lo, lo, hi]
        matrix[2] = [lo, lo, hi, lo, lo, hi, lo, lo, lo, lo, hi, lo, lo, hi, lo, lo]
        cpd =TabularCPD(variable=var, variable_card=card_var, values=matrix,
                          evidence=evidence,
                          evidence_card=cardinality)
        for i, node in enumerate(evidence):
            colors.update({node:table[i]})
        return colors,table, cpd


    # def set_color(self, color):
    #     col = self.colors_table[:, color]
    #     for i in range(0,len(col)):
    #         node = 'BENS_'+ str(i)
    #         self.pgmpy.get_cpds(node).values = CPD.RON_cpd(node, self.pgmpy.get_cardinality(node), mu = int(col[i])).values

    def add_edges(self, topology):
        self.networx.remove_edges_from(self.edges)
        self.edges = []
        shape = np.asarray(topology).shape
        ''' let's first remove all void nodes  ----> not necssary -----> delete the code ??'''
        nodes_to_remove = []
        rows = np.sum(topology, axis = 1)
        columns = np.sum(topology,axis = 0)
        # for row in range(0, len(rows)):
        #     if rows[row] == 0:
        #         nodes_to_remove.append('WORLD_' + str(row))
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
                self.nodes[i][1]['cpd'] = CPD.create_fixed_parent(cardinality, modus = 'uniform')
            else:
                incoming_nodes = self.networx.in_edges(node[0])
                if len(incoming_nodes) == 0:
                    self.nodes[i][1]['cpd'] = CPD.create_random_child(cardinality, modus = 'orphan')
                    continue
                card_parent = []
                for  m, n in enumerate(incoming_nodes):
                    par = self.networx.node[n[0]]['cardinality']
                    card_parent.append(par)
                self.nodes[i][1]['cpd'] = CPD.create_random_child(cardinality, card_parent)


    def create_learning_data(self):
        self.get_my_colors()
        self.learning_data = {}
        for i, node in enumerate(self.nodes):
            if "BEN" in node[0]:
                self.learning_data.update({node[0]:self.colors_table[i].tolist()})
            if "WORLD" in node[0]:
                shape = self.colors_cpd.values.shape
                reshaped_cpd = self.colors_cpd.values.reshape(shape[0], int(np.prod(shape)/shape[0]))
                for hue in range(0,3):
                    if str(hue) in node[0]:
                        self.learning_data.update({node[0]:reshaped_cpd[hue,:].tolist()})
        # print('Learning data')
        # print(self.learning_data)


    def do_inference(self, models):
        error = 0
        for key in models:
            error += models[key].process()
        return error





    def test_topology(self, entropy):
        self.networx_test = copy.deepcopy(self.networx)
        self.pgmpy_test = BayesianModel()
        self.pgmpy_test   = copy.deepcopy(self.pgmpy)
        model = {'main': GenerativeModel(SensoryInputVirtualPeepo(self), self.pgmpy_test)}
        self.expected_result = [0,0,0]
        ''' ------ going through all possible "colors'''
        error = 0
        for color in range(0, self.number_of_colors):
            states = self.colors_table[:,color]
            shape = self.colors_cpd.values.shape
            reshaped_cpd = self.colors_cpd.values.reshape(shape[0], int(np.prod(shape) / shape[0]))
            self.expected_result = reshaped_cpd[:,int(color)]
            for i, pixel in enumerate(states):
                if 'BENS_'+str(i) not in self.networx_test.nodes():
                   continue
                cardinality = self.pgmpy_test.get_cardinality('BENS_'+str(i))
                self.pgmpy_test.get_cpds('BENS_' + str(i)).values = CPD.create_fixed_parent(cardinality, state = int(pixel))
            error += self.do_inference(model)
        error /= self.number_of_colors
        self.results.append([entropy, error])
        if error <= self.best_error:
            self.best_error = error
            self.best_topology[0] = error
            self.best_topology[1] = entropy
            self.best_topology[2] = self.networx_test


    def estimate_parameters(self):
        data = pd.DataFrame(data=self.learning_data)
        estimator = BayesianEstimator(self.pgmpy, data)
        for i, node in enumerate(self.nodes):
            if 'LAN' in node[0] or 'MOTOR' in node[0] or 'WORLD' in node[0]:
                self.pgmpy.get_cpds(node[0]).values = estimator.estimate_cpd(node[0], prior_type='dirichlet', pseudo_counts=[2, 3]).values




    def do_it(self):
        '''EXPLANATIONS'''
        self.networx_fixed , self.dictionary, self.header = self._util.get_network()
        self.networx = copy.deepcopy(self.networx_fixed)
        self.networx_test= copy.deepcopy(self.networx_fixed)
        self.nodes = self.networx.nodes(data=True)
        self.create_learning_data()
        print(self.learning_data)
        print('Dictionary : ', self.dictionary)

        ''' -------------- Constructing all possible topologies, 
                              --> option : restrain the number with the treshold : 
                                        0 -> all possible topologies, 100 -> only the fully connnected topology'''
        possible_topologies  = self._lat.get_possible_topologies(treshold = 50)#setting the entropy at a 50% -> only topologies with an entropy >= 0.5 will be considered
        print("Possible topologies : ", len(possible_topologies))
        entropy = 0
        count = 0#TEMPORARY
        ''' -------------- walking through all toplogies'''
        for topology in possible_topologies:
            entropy = topology[1]
            if entropy == 0:
                continue#safeguard
            topo  = topology[0]
            self.networx = copy.deepcopy(self.networx_fixed)
            edges = []
            parent = ''
            child = ''

            ''' ----------- for each topology we construct the edges and update dummy cpd (necessary as the shape of the LENs cpd's can change
                            depending on the number of incoming nodes'''
            self.add_edges(topo)

            self.loop += 1
            self.add_dummy_cpds()
            print('Loop **************************************************************-> ', self.loop)

            ''' ----------- convert DiGraph to pgmpy and check'''
            self.pgmpy = self._util.translate_digraph_to_pgmpy(self.networx)

            '''------------ ask pgmpy to guess the best cpd's of the LANs and LENs 
                             -> provide pgmpy with the learning data'''
            self.estimate_parameters()

            self.pgmpy.check_model()

            '''-------------- Testing the constructed topology'''
            self.test_topology(entropy)

            '''following  4 lines to remove : just use to check whether the algorithms are correct regarding the edges building'''
            count += 1
            #print('edges : ', self.edges)
            if count > 20:
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
        self.draw_xy()
        return self.results

    def draw_x_y(self):
        '''TO REMOVE LATER'''
        plt.figure(figsize=(10, 5))
        pos = nx.circular_layout(self.best_topology[2], scale=2)
        # node_labels = nx.get_node_attributes(self.networx, 'cpd')
        nx.draw(self.best_topology[2], pos, node_size=1200, node_color='lightblue',
                linewidths=0.25, font_size=10, font_weight='bold', with_labels=True)
        plt.show()

    def draw(self):
        '''TO REMOVE LATER'''
        plt.figure(figsize=(10, 5))
        pos = nx.circular_layout(self.best_topology[2], scale=2)
        #node_labels = nx.get_node_attributes(self.networx, 'cpd')
        nx.draw(self.best_topology[2], pos, node_size=1200, node_color='lightblue',
                linewidths=0.25,  font_size=10, font_weight='bold', with_labels=True)
        plt.show()

def main():
    case = 'simple_color_recognition'
    mycase = MyClass(case)
    results = mycase.do_it()
    #print(results)

####################################################################################
############################### BEGIN HERE #########################################
####################################################################################

if __name__ == "__main__":
    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.INFO)
    main()