import math
import random
import numpy as np
from pgmpy.factors.discrete import TabularCPD

class CPD:

    def __init__(self):
        pass
    def get_index_matrix(cardinality):  # creates a matrix for the header of the contingency table (used  in create latent distribution with a fixed distibution)
        C = int(np.prod(cardinality))
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

    def create_random_child(card_latent, card_parent):
        modus = 'random'
        if (modus == 'random'):
            cardinality = 1
            if len(card_parent) == 1:
                table = np.random.rand(card_latent, card_parent[0])
                # Normalize distribution
                factor = 0
                for column in range(0, card_parent[0]):
                    factor += table[column]
                for column in range(0, card_parent[0]):
                    table[column] /= factor
                return table

            for n, nod in enumerate(card_parent):
                cardinality = cardinality * card_parent[n]
                #n = n + 1
                table = np.random.rand(card_latent, cardinality)
            for c in range(0, len(table[0])):
                factor = 0
                for r in range(0, len(table)):
                    factor += table[r][c]
                for r in range(0, len(table)):
                    table[r][c] /= factor
        return table


    def create_one2one_distribution(card_leaf, card_parent):
        ar = np.identity(card_parent[0])
        if card_leaf != card_parent[0]:
            print("Passing wrong dimension in one 2 one distribution")
            ar = np.zeros(card_parent)
            return
        return ar

    def color_cpd(var,card_var,evidence,cardinality):
        table = CPD.get_index_matrix(cardinality)
        colors ={}
        n_evidence = len(cardinality)
        hi = 0.99
        lo = 0.01
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

    def RON_cpd(var, cardinality, mu, sigma, mode):
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

    def LAN_cpd(var, card_latent, card_parent, evid, modus, gamma):
        table = []
        if (modus == 'fixed'):
            table = CPD.create_latent_distribution(card_latent, card_parent, gamma)
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


    def LEN_cpd(var, card_latent, card_parent, evid, modus, gamma):
        table = []
        if (modus == 'fixed'):
            table = CPD.create_latent_distribution(card_latent, card_parent, gamma)
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