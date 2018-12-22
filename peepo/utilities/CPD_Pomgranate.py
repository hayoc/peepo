from  pomegranate import *
import random
import numpy as np

class CPD_P:

    def __init__(self):
        pass

    def get_index_matrix(
            cardinality):  # creates a matrix for the header of the contingency table (used  in create latent distribution with a fixed distibution)
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

    def create_fixed_parent(cardinality, state=0, modus='status'):
        hi = 0.99
        lo = 0.01 / (cardinality - 1)
        ar = np.full(cardinality, lo)
        if (modus == 'status'):
            ar[state] = hi
        # normalize
        som = 0
        for i in range(0, cardinality):
            som += ar[i]
        for i in range(0, cardinality):
            ar[i] /= som
        return ar

    def create_random_child(card_child, card_parent=[], modus='random'):
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
                # n = n + 1
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
        R = np.size(matrix, 0)
        C = np.size(matrix, 1)
        for column in range(0, C):
            factor = 0
            for row in range(0, R):
                matrix[row][column] += random.uniform(-0.005, 0.005)  # this to avoid an exact equidistant probabiliy
                factor += matrix[row][column]
            for row in range(0, R):
                matrix[row][column] /= factor
        return matrix

    def RON_cpd(var, cardinality, mu=0, sigma=0, modus='fixed'):
        if modus == "random":
            table = np.random.rand(cardinality)
            # Normalize distribution
            factor = 0
            for column in range(0, cardinality):
                factor += table[column]
            for column in range(0, cardinality):
                table[column] /= factor
            return TabularCPD(variable=var, variable_card=cardinality, values=[table])
        table = CPD.create_fixed_parent(cardinality, mu)
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