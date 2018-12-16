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
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput
import matplotlib.pyplot as plt


class MyClass(object):
    def __init__(self, case):
        self.case = case
        self.results = []
        self.networx = nx.DiGraph()
        self.dictionary = []
        self.header = {}
        self.nodes = []
        self.edges = {}
        self.cpds = {}
        self.possible_edges = []

    def get_possible_edges(self):
        pass

    def do_it(self):
        '''EXPLANATIONS'''
        self.networx, self.dictionary, self.header = Utilities.get_network(self.case)
        self.nodes = self.networx.nodes(data=True)
        print(self.nodes)
        print(self.header)
        print(self.dictionary)
        # self.networx.add_edge('BENS_1','WORLD_1')
        # self.networx.node['BENS_1']['cpd'] = [0.8,0.2]
        # self.networx.node['WORLD_2']['cpd'] = [[0.8, 0.2, 0.5,0.3],[0.2,0.8,0.5,0.7]]
        Utilities.save_network(self.case,self.header,self.dictionary, self.networx)
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