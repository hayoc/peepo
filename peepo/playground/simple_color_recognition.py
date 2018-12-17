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
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput
import matplotlib.pyplot as plt


class MyClass(object):
    def __init__(self, case):
        self.case = case
        self.results = []
        self.networx = nx.DiGraph()
        self.pgmpy  = BayesianModel()
        self.dictionary = []
        self.header = {}
        self.nodes = []
        self.edges = {}
        self.cpds = {}
        self.possible_edges = []
        self._util = Utilities(case)
        self._lat = Lattices(self._util)

    def do_it(self):
        '''EXPLANATIONS'''
        self.networx, self.dictionary, self.header = self._util.get_network()
        self.nodes = self.networx.nodes(data=True)
        print(self.nodes)
        print(self.header)
        print(self.dictionary)
        possible_structures  = self._lat.get_possible_paths()
        entropy = 0
        count = 0#TEMPORARY
        for skeleton in possible_structures:
            entropy = skeleton[1]
            skelet  = skeleton[0]
            edges = []
            parent = ''
            child = ''
            for column, ben  in enumerate(skelet[1]):
                if ben == 1:
                    parent = 'BENS_'+str(column)
                    print('parent : ', parent)
                if len(parent) > 0:
                    for row, world in enumerate(skelet[0]):
                        child = 'WORLD_'+str(row)
                        self.networx.add_edge(parent,child)
            self.edges= self.networx.edges()
            '''following  4 lines to remove : just use to check whether the algorithms are correct regarding the edges building'''
            count += 1
            print('edges : ', self.edges)
            if count > 100:
                break
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