import os
import networkx as nx
import numpy as np
import scipy as scp
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
import re
import itertools


class Lattices(object):

    def __init__(self,utility_pointer):
        self._util = utility_pointer
        ''' no object creation -> opportune  ?'''

        pass

    def get_index_matrix(self,cardinality):
        ''' creates a matrix of all possible two states combinations (0,1)  of a vector of size len(cardinality) '''
        C = np.prod(cardinality)
        blocks = np.copy(cardinality)
        B = len(blocks)
        for b in range(1, B):
            index = B - 1 - b
            blocks[index] = blocks[index + 1] * cardinality[index]
        M = np.zeros((len(cardinality), C))
        # construct first the lowest row
        block = np.zeros(cardinality[len(cardinality) - 1])
        for n in range(0, len(block)):
            block[n] = n
        # fill M  with the right number of blocks
        n_blocks = int(C / blocks[B - 1])
        R = []
        for n in range(0, n_blocks):
            R.append(block)
        R = np.reshape(R, n_blocks * len(block))
        M[B - 1, :] = R
        block_mem = int(C / blocks[B - 1])
        # now the rest of the rows
        for b in range(0, B - 1):
            row = B - 2 - b
            block = np.zeros(blocks[row])
            block_mem /= cardinality[row]
            n_blocks = int(block_mem)
            # fill first the block
            index = 0
            index_ = 0
            for p in range(0, len(block)):
                block[p] = index
                index_ += 1
                if index_ > blocks[row + 1] - 1:
                    index_ = 0
                    index += 1
            # now create an R array with the right number of blocks
            R = []
            for n in range(0, n_blocks):
                R.append(block)
            R = np.reshape(R, n_blocks * len(block))
            M[row, :] = R
        return np.transpose(M)

    def get_possible_states(self,number_of_nodes):
        if number_of_nodes == 0:
            return False
        card = np.full(number_of_nodes,2)
        return self.get_index_matrix(card)

    def make_state_sub_matrix(self,unit_state, number_of_atoms):
        matrix = unit_state
        matrix = matrix.tolist()
        for r in range(0,number_of_atoms-1):
            for i in itertools.product(matrix,unit_state.tolist()):
                #print('i = ', i)
                a = i.tolist()
                print(' a = ',a)
                matrix.append(a)
        return matrix

    def make_state_base(self,unit_state, number_of_atoms):
        list = []
        matrix = []
        for el in range(0,number_of_atoms):
            list.append(unit_state.tolist())
        for m in itertools.product(*list):
            #print('matrix : ', matrix)
            matrix.append(m)
        return matrix

    def calculate_entropy(self, b_w_matrix):
        B_W_matrix =[]
        shape = np.asarray(b_w_matrix[0]).shape
        #mu = 1.0 / np.prod(shape)
        nu = scp.special.lambertw(1.0 / np.prod(shape)).real  # , k=0, tol=1e-8)[source]¶
        for i, mat in enumerate(b_w_matrix):
            entropy = self.calculate_NM_entropy(mat,shape, nu)
            B_W_matrix.append([mat, entropy])
        '''reorder B_W_matrix with the entropy as key'''
        B_W_matrix.sort(key=lambda tup: tup[1])  # sorts in place
        return B_W_matrix

    def calculate_NM_entropy(self, b_w_matrix, shape, nu):
        entropy = 0
        for row in range(0, shape[0]):
            for column in range(0,shape[1]):
                entropy += nu*b_w_matrix[row][column]*np.exp(nu*b_w_matrix[row][column])
        return entropy


    def get_possible_paths(self):
        print('in get_paths')
        BENS_Nodes  = self._util.get_nodes_in_family( 'BENS')
        # MEMS_Nodes  = self._util.get_nodes_in_family('MEMS')
        # LANS_Nodes  = self._util.get_nodes_in_family( 'LANS')
        # MOTOR_Nodes = self._util.get_nodes_in_family( 'MOTOR')
        WORLD_Nodes = self._util.get_nodes_in_family( 'WORLD')
        BENS_states = self.get_possible_states(len(BENS_Nodes))
        b_w_matrix  = self.make_state_base(BENS_states, len(WORLD_Nodes))
        '''TO be developed further for MEMS and LANS
        MEMS_states = self.get_possible_states(len(MEMS_Nodes))
        LANS_states = self.get_possible_states(len(LANS_Nodes))


        if not LANS_states:
            if not BENS_states:
                b_m_matrix = self.make_state_base(BENS_states, len(MOTOR_Nodes))
            if not BENS_states:
            b_w_matrix = self.make_state_base(BENS_states, len(WORLD_Nodes))
            if not BENS_states:
            m_m_matrix = self.make_state_base(MEMS_states, len(MOTOR_Nodes))
            if not BENS_states:
            m_w_matrix = self.make_state_base(MEMS_states, len(WORLD_Nodes))'''
        print('b_w_matrix has length : ', len(b_w_matrix))

        B_W_matrix = self.calculate_entropy(b_w_matrix)
        #print(b_n_matrix)
        #B_M_matrix = self.make_state_sub_matrix(BENS_states,len(WORLD_Nodes))
        print('B_W_matrix :')
        for i, m in enumerate(B_W_matrix):
            print(m[1], ' ---> ', m[0])
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