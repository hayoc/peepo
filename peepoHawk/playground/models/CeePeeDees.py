#28/11/2018
import math
import random
import numpy as np
from pgmpy.factors.discrete import TabularCPD

class CPD:

    def __init__(self):
        pass
    def get_index_matrix(cardinality):  # creates a matrix for the header of the contingency table (used  in create latent distribution with a fixed distibution)
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
        return M

    def create_fixed_parent(cardinality, index):
        ar = np.zeros(cardinality)
        for i in range(0,cardinality):
            ar[i] = 1/cardinality
        ar[index] = cardinality
        #normalize
        som = 0
        for i in range(0,cardinality):
            som += ar[i]
        for i in range(0,cardinality):
            ar[i] /= som
        return ar

    def create_one2one_distribution(card_leaf, card_parent):
        ar = np.identity(card_parent[0])
        if card_leaf != card_parent[0]:
            print("Passing wrong dimension in one 2 one distribution")
            ar = np.zeros(card_parent)
            return
        return ar


    def create_Raptor_distribution(card_latent, card_parent, index_jump, modus):
        # CREATES : a CPD with a distribution depending on the "distance" of the latent variable index to the indexes of the parents
        # the distance is the inverse of an exponentional of the sum of the distances corrected with a factor sigma
        # cardinality of the latent must be the same as the cardinality of the parents
        C = np.prod(card_parent)
        matrix = np.full((card_latent, C), 1./card_latent)
        if modus ==  'MEM_direction':
            M = CPD.get_index_matrix(card_parent)
            print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
            print(M)
            hi = 100
            lo = 5
            # 0 : Delta_R > Delta_L
            # 1 : Delta_L > Delta_R
            #
            for column in range(0, C):
                matrix[0][column] = lo
                matrix[1][column] = hi
                if (M[0][column]  -  M[1][column]) < (M[2][column]  -  M[3][column]):
                    matrix[0][column] = hi
                    matrix[1][column] = lo

        if modus ==  'MEM_correction':
            M = CPD.get_index_matrix(card_parent)
            hi = 100
            lo = 5
            print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
            print(M)
            matrix = []
            row = [lo,hi,lo,lo,lo,lo,hi,lo]
            matrix.append(row)
            row = [lo,lo,hi,hi,hi,hi,lo,lo]
            matrix.append(row)
            row = [hi,lo,lo,lo,lo,lo,lo,hi]
            matrix.append(row)
        # Normalize distribution
        for column in range(0, C):
            factor = 0
            for row in range(0, card_latent):
                matrix[row][column] += random.uniform(-0.005, 0.005)#this to avoid an exact equidistant probabiliy
                factor += matrix[row][column]
            for row in range(0, card_latent):
                matrix[row][column] /= factor

        if 'vision' in modus:
            transition_matrix = []
            if 'right' in modus:
                transition_matrix.append([-index_jump, +0, +index_jump])
                transition_matrix.append([-index_jump, +index_jump, +index_jump])
                transition_matrix.append([+index_jump, +0, -index_jump])
            if 'left' in modus:
                transition_matrix.append([+index_jump, +0, -index_jump])
                transition_matrix.append([+index_jump, -index_jump, -index_jump])
                transition_matrix.append([+index_jump, +0, +index_jump])
            M = CPD.get_index_matrix(card_parent)
            print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")
            print(M)
            hi = 1000
            lo = 5
            R = card_latent
            # 0 : Delta_R > Delta_L
            # 1 : Delta_L > Delta_R
            #
            for column in range(0, C):
                d_index = transition_matrix[int(M[1][column])][int(M[2][column])]
                index = int(M[0][column] + d_index)
                if index < 0:
                    index = 0
                if index >= card_latent:
                    index = int(card_latent - 1)
                matrix[index][column] = hi
        # Normalize distribution
        matrix = CPD.normalize_distribution(matrix)
        return matrix

    def normalize_distribution(matrix):
        R = np.size(matrix,0)
        C = np.size(matrix,1)
        for column in range(0, C):
            factor = 0
            for row in range(0, R):
                matrix[row][column] += random.uniform(-0.005, 0.005)#this to avoid an exact equidistant probabiliy
                factor += matrix[row][column]
            for row in range(0, R):
                matrix[row][column] /= factor
        return matrix

    def parent_cpd(var, cardinality, mu, sigma, mode):
        if mode == "random":
            table = np.random.rand( cardinality)
            # Normalize distribution
            factor = 0
            for column in range(0, cardinality):
                factor += table[column]
            for column in range(0, cardinality):
                table[column] /= factor
            return TabularCPD(variable=var, variable_card=cardinality, values=[table])
        table = CPD.create_fixed_parent(cardinality,mu)

        return TabularCPD(variable=var, variable_card=cardinality, values=[table])

    def latent_cpd(var, card_latent, card_parent, evid, modus, gamma):
        table = []
        if (modus == 'fixed'):
            table = CPD.create_latent_distribution(card_latent, card_parent, gamma)
        if (modus == 'delta_alfa')  or (modus == 'direction') or (modus == 'correction') :
                table = CPD.create_Raptor_distribution(card_latent, card_parent, gamma, modus)
        if (modus == 'random'):
            cardinality = 1
            for n in range(0, len(card_parent)):
                cardinality = cardinality * card_parent[n]
                n = n + 1
                table = np.random.rand(card_latent, cardinality)
            for c in range(0, len(table[0])):
                factor = 0
                for r in range(0, len(table)):
                    factor += table[r][c]
                for r in range(0, len(table)):
                    table[r][c] /= factor
        evidence = []
        car_par = []
        for n in range(0, len(evid)):
            evidence.append(evid[n])
            car_par.append(card_parent[n])
        return TabularCPD(variable=var, variable_card=card_latent, values=table,
                          evidence=evidence,
                          evidence_card=car_par)


    def leaf_cpd(var, card_latent, card_parent, evid, modus, gamma):
        table = []
        if (modus == 'fixed'):
            table = CPD.create_latent_distribution(card_latent, card_parent, gamma)
        if 'vision' in modus :
                table = CPD.create_Raptor_distribution(card_latent, card_parent, gamma, modus)
        if 'one_2_one' in modus :
                table = CPD.create_one2one_distribution(card_latent, card_parent)
        if (modus == 'random'):
            cardinality = 1
            for n in range(0, len(card_parent)):
                cardinality = cardinality * card_parent[n]
                n = n + 1
                table = np.random.rand(card_latent, cardinality)
            for c in range(0, len(table[0])):
                factor = 0
                for r in range(0, len(table)):
                    factor += table[r][c]
                for r in range(0, len(table)):
                    table[r][c] /= factor
        evidence = []
        car_par = []
        for n in range(0, len(evid)):
            evidence.append(evid[n])
            car_par.append(card_parent[n])
        return TabularCPD(variable=var, variable_card=card_latent, values=table,
                          evidence=evidence,
                          evidence_card=car_par)