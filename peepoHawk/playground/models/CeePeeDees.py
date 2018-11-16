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

    def create_action_distribution(card_latent, card_parent, sigma):
        # CREATES : a CPD with a distribution depending on the "distance" of the latent variable index to the indexes of the parents
        # the distance is the inverse of an exponentional of the sum of the distances coorected with a factor sigma
        # cardinality of the latent must be the same as the cardinality of the parents
        C = np.prod(card_parent)
        matrix = np.zeros((card_latent, C))
        M = CPD.get_index_matrix(card_parent)
        for row in range(0, card_latent):
            for column in range(0, C):
                correction = 1
                if M[1][column] == 2:
                    x = (row - M[0][column]) * (row - M[0][column]) / sigma / sigma
                    correction = math.exp(-x)
                if M[1][column] == 1:
                    correction = 1
                if M[1][column] == 0:
                    x = (row - M[0][column]) * (row - M[0][column]) / sigma / sigma
                    correction = 1 - math.exp(-x)
                matrix[row][column] = correction
        # Normalize ditribution
        for column in range(0, C):
            factor = 0
            for row in range(0, card_latent):
                factor += matrix[row][column]
            for row in range(0, card_latent):
                matrix[row][column] /= factor
        return matrix

    def create_latent_reward_distribution():
        # CREATES : aCPD a distribution depending on the "distance" of the latent variable index to the indexes of the parents
        # the distance is the inverse of an exponentional of the sum of the distances coorected with a factor (set to 1 for the moment)
        # cardinality of the latent must be the same as the cardinality of the parents
        card_parent = [3, 3]
        card_latent = 3
        C = np.prod(card_parent)
        matrix = [[0.1, 0.2, 0.8, 0.1, 0.1, 0.4, 0.1, 0.1, 0.8],
                  [0.1, 0.4, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1],
                  [0.8, 0.4, 0.1, 0.8, 0.8, 0.2, 0.8, 0.8, 0.1]]
        return matrix

    def create_latent_distribution(card_latent, card_parent, gamma):
        # CREATES : aCPD a distribution depending on the "distance" of the latent variable index to the indexes of the parents
        # the distance is the inverse of an exponentional of the sum of the distances coorected with a factor (set to 1 for the moment)
        # cardinality of the latent must be the same as the cardinality of the parents
        C = np.prod(card_parent)
        matrix = np.zeros((card_latent, C))
        M = CPD.get_index_matrix(card_parent)
        for row in range(0, card_latent):
            for column in range(0, C):
                distance = 0
                for M_row in range(0, len(card_parent)):
                    distance += abs(row - M[M_row][column])
                matrix[row][column] = math.exp(-gamma * distance)
        # Normalize ditribution
        for column in range(0, C):
            factor = 0
            for row in range(0, card_latent):
                factor += matrix[row][column]
            for row in range(0, card_latent):
                matrix[row][column] /= factor
        return matrix

    def create_leaf_distribution(card_leaf, card_latent, modus, sigma):
        if modus == "azimuth":
            C = np.prod(card_latent)
            matrix = np.zeros((card_leaf, C))
            M = CPD.get_index_matrix(card_latent)
            for column in range(0, C):
                delta_reward = M[1][column]
                for row in range(0, card_leaf):
                    delta_index = abs(M[0][column] - M[2][column])
                    if delta_reward == 2:
                        mu = delta_index
                        y = (row - mu) * (row - mu) / sigma / sigma
                        matrix[row][column] = math.exp(-y)
                    if delta_reward == 0:
                        mu = card_leaf - delta_index
                        y = (row - mu) * (row - mu) / sigma / sigma
                        matrix[row][column] = math.exp(-y)
                    if delta_reward == 1:
                        mu = card_leaf / 2
                        y = (row - mu) * (row - mu) / sigma / sigma
                        matrix[row][column] = math.exp(-y)

        if modus == "reward":
            C = np.prod(card_latent)
            matrix = np.zeros((card_leaf, C))
            M = CPD.get_index_matrix(card_latent)
            for column in range(0, C):
                delta_index = abs(M[0][column] - M[2][column])
                delta_reward = M[1][column]
                y_low = 10 + 90 / card_latent[0] * delta_index
                y_high = 100 - 90 / card_latent[0] * delta_index
                for row in range(0, card_leaf):
                    matrix[row][column] = 45
                    if delta_reward == 0 and row == 0:
                        matrix[row][column] = y_low
                    if delta_reward == 2 and row == 0:
                        matrix[row][column] = y_low
                    if delta_reward == 0 and row == 2:
                        matrix[row][column] = y_high
                    if delta_reward == 2 and row == 2:
                        matrix[row][column] = y_high

        # Normalize distribution
        for column in range(0, C):
            factor = 0
            for row in range(0, card_leaf):
                factor += matrix[row][column]
            for row in range(0, card_leaf):
                matrix[row][column] /= factor

        return matrix

    def updated_cpd(cardinality, index):
        v = np.zeros(cardinality)
        for i in range(0, cardinality):
            v[i] = 0.1 / (cardinality - 1)
        v[index] = 0.9
        return v

    def parent_cpd(var, cardinality, mu, sigma):
        table = np.zeros(cardinality)
        factor = 0
        for x in range(0, cardinality):
            y = (x - mu) * (x - mu) / sigma / sigma
            prob = math.exp(-y)
            factor += prob
            table[x] = prob
        # normalize
        for t in range(0, len(table)):
            table[t] /= factor
        # print(table)
        return TabularCPD(variable=var, variable_card=cardinality, values=[table])

    def latent_cpd(var, card_latent, card_parent, evid, modus, gamma):
        table = []
        if (modus == 'fixed'):
            table = CPD.create_latent_distribution(card_latent, card_parent, gamma)
        if (modus == 'action'):
            table = CPD.create_action_distribution(card_latent, card_parent, gamma)
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

    def reward_cpd(var, card_latent, card_parent, evid, modus, gamma):
        table = CPD.create_latent_reward_distribution()
        evidence = []
        car_par = []
        for n in range(0, len(evid)):
            evidence.append(evid[n])
            car_par.append(card_parent[n])
        return TabularCPD(variable=var, variable_card=card_latent, values=table,
                          evidence=evidence,
                          evidence_card=car_par)

    def leaf_cpd(var, card_leaf, card_latent, evid, modus, sigma):
        table = []
        if (modus == 'azimuth'):
            table = CPD.create_leaf_distribution(card_leaf, card_latent, modus, sigma)
        if (modus == 'reward'):
            table = CPD.create_leaf_distribution(card_leaf, card_latent, modus, sigma)
        if (modus == 'random'):
            cardinality = 1
            for n in range(0, len(card_latent)):
                cardinality = cardinality * card_latent[n]
                n = n + 1
                table = np.random.rand(card_leaf, cardinality)
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
            car_par.append(card_latent[n])
        return TabularCPD(variable=var, variable_card=card_leaf, values=table,
                          evidence=evidence,
                          evidence_card=car_par)
