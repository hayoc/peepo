import math
import random
import numpy as np
from pgmpy.factors.discrete import TabularCPD
import itertools

class CPD:

    def __init__(self):
        pass
    def get_index_matrix(cardinality):  # creates a matrix for the header of the contingency table (used  in create latent distribution with a fixed distibution)
        """
        Returs the state tables of the parents of a node

        :param cardinality: an array with the cardinalities of the parent
        :returns: an array with the combination of all possible states
        :type cardinality: array
        :rtype : array

        example:

        cardinality =   [2,3,2]
        will return:
                        [[0 0 0 0 0 0 1 1 1 1 1 1],
                         [0 0 1 1 2 2 0 0 1 1 2 2],
                         [0 1 0 1 0 1 0 1 0 1 0 1 ]]
        """
        blocks =[]
        for i in range(0,len(cardinality)):
            subblock = []
            [subblock.append(int(s)) for s in range(0,cardinality[i])]
            blocks.append(subblock)
        return np.transpose(np.asarray(list(itertools.product(*blocks))))



    def create_fixed_parent(cardinality, state = 0, modus = 'status'):
        hi = 0.99
        lo = 0.01/(cardinality-1)
        ar = np.full(cardinality,lo)
        if(modus == 'status'):
            ar[state] = hi
        #normalize
        som = 0
        for i in range(0,cardinality):
            som += ar[i]
        for i in range(0,cardinality):
            ar[i] /= som
        return ar

    def create_random_child(card_child, card_parent = [], modus = 'random'):
        table = []

        if (modus == 'orphan'):
            hi = 1.
            lo = 1.
            table = np.full(card_child, lo)
            # normalize
            som = 0
            for i in range(0, card_child):
                som += table[i]
            for i in range(0, card_child):
                table[i] /= som
            return table

        if (modus == 'random'):
            cardinality = 1
            if len(card_parent) == 1:
                table = np.random.rand(card_child, card_parent[0])
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
                table = np.random.rand(card_child, cardinality)

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

    def RON_cpd(var, cardinality, mu = 0, sigma = 0, modus = 'fixed'):
        if modus == "random":
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