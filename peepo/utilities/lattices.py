import os
import networkx as nx
import numpy as np
import scipy as scp
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import re
import itertools
from peepo.playground.simple_color_recognition.CeePeeDees import CPD


class Lattices(object):

    def __init__(self,utility_pointer):
        self._util = utility_pointer



    def get_possible_states(self,number_of_nodes):
        if number_of_nodes == 0:
            return False
        card = np.full(number_of_nodes,2)
        return np.transpose(CPD.get_index_matrix(card))

    def make_state_sub_matrix(self,unit_state, number_of_atoms):
        matrix = unit_state
        matrix = matrix.tolist()
        for r in range(0,number_of_atoms-1):
            for i in itertools.product(matrix,unit_state.tolist()):
                a = i.tolist()
                matrix.append(a)
        return matrix

    def make_state_base(self,unit_state, number_of_atoms):
        list = []
        matrix = []
        for el in range(0,number_of_atoms):
            list.append(unit_state.tolist())
        for m in itertools.product(*list):
            ''' CHECK whether all LENS has at least 1 incoming parent'''
            sum_columns = np.sum(m,axis =1)
            accept = True
            for c in range(0,len(sum_columns)):
                if sum_columns[c] == 0:
                    accept = False
                    break
            if accept:
                matrix.append(m)
        return matrix

    def calculate_entropy(self, b_w_matrix, treshold):
        treshold /= 100
        B_W_matrix =[]
        shape = np.asarray(b_w_matrix[0]).shape
        #nu = scp.special.lambertw(1.0 / np.prod(shape)).real  # , k=0, tol=1e-8)[source]¶
        for i, mat in enumerate(b_w_matrix):
            entropy = np.sum(mat)/np.prod(shape)
            if entropy >= treshold :
                B_W_matrix.append([mat,entropy])
        '''reorder B_W_matrix with the entropy in descending order as key'''
        B_W_matrix.sort(reverse=True,key=lambda tup: tup[1])  # sorts in place
        return B_W_matrix

    def calculate_NM_entropy(self, b_w_matrix, shape,nu):
        entropy = 0
        for row in range(0, shape[0]):
            for column in range(0,shape[1]):
                entropy += nu*b_w_matrix[row][column]*np.exp(nu*b_w_matrix[row][column])
        return entropy


    def get_possible_topologies(self, treshold = 0):
        BENS_Nodes  = self._util.get_nodes_in_family( 'BENS')
        # MEMS_Nodes  = self._util.get_nodes_in_family('MEMS')
        # LANS_Nodes  = self._util.get_nodes_in_family( 'LANS')
        # MOTOR_Nodes = self._util.get_nodes_in_family( 'MOTOR')
        WORLD_Nodes = self._util.get_nodes_in_family( 'WORLD')
        BENS_states = self.get_possible_states(len(BENS_Nodes))
        b_w_matrix  = self.make_state_base(BENS_states, len(WORLD_Nodes))
        '''
        *******************  TO DO *********************************'''

        # TO be developed further for MEMS and LANS
        # MEMS_states = self.get_possible_states(len(MEMS_Nodes))
        # LANS_states = self.get_possible_states(len(LANS_Nodes))
        #
        #
        # if not LANS_states:
        #     if not BENS_states:
        #         b_m_matrix = self.make_state_base(BENS_states, len(MOTOR_Nodes))
        #     if not BENS_states:
        #     b_w_matrix = self.make_state_base(BENS_states, len(WORLD_Nodes))
        #     if not BENS_states:
        #     m_m_matrix = self.make_state_base(MEMS_states, len(MOTOR_Nodes))
        #     if not BENS_states:
        #     m_w_matrix = self.make_state_base(MEMS_states, len(WORLD_Nodes))

        B_W_matrix = self.calculate_entropy(b_w_matrix,treshold)

        #print(b_n_matrix)
        #B_M_matrix = self.make_state_sub_matrix(BENS_states,len(WORLD_Nodes))
        # print('B_W_matrix :')
        # for i, m in enumerate(B_W_matrix):
        #     print(m[1],  ' ---> ', m[0])
        # print(len(B_M_matrix))
        #
        #
        # RONS_Nodes = BENS_Nodes + MEMS_Nodes
        # LENS_Nodes = MOTOR_Nodes + WORLD_Nodes
        # print(RONS_Nodes)
        # print(LANS_Nodes)
        # print(LENS_Nodes)
        # RONS_pool = RONS_Nodes
        # for n in range(len(RONS_Nodes) - 1):
        #     RONS_pool += RONS_Nodes
        # print('rosn pool : ', RONS_pool)
        # RONS_combinations = []
        # for combination in itertools.product(RONS_pool, RONS_pool):
        #     print(combination)
        #     RONS_combinations.append(combination)
        # print('RONS possible combinations :')
        # print(RONS_combinations)
        #
        # number_of_levels = 1
        # if len(LANS_Nodes) > 0:
        #     number_of_levels += 1
        # print('number_of_levels : ', number_of_levels)
        # if number_of_levels == 1:
        #     for path in itertools.product(RONS_Nodes, LENS_Nodes):
        #         print("path : ", path)
        #         possible_paths.append(path)
        # else:
        #     for path in itertools.product(RONS_Nodes, LANS_Nodes, LENS_Nodes):
        #         print("path : ", path)
        #         possible_paths.append(path)
        return B_W_matrix