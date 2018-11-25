#   20/11/18
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
        som = 0
        for i in range(0,cardinality):
            som += ar[i]
        for i in range(0,cardinality):
            ar[i] /= som
        return ar

    def create_prediction_distribution(card_latent, card_parent, sigma):
        # CREATES : a CPD with a distribution depending on the "distance" of the latent variable index to the indexes of the parents
        # the distance is the inverse of an exponentional of the sum of the distances coorected with a factor sigma
        # cardinality of the latent must be the same as the cardinality of the parents
        C = np.prod(card_parent)
        matrix = np.zeros((card_latent, C))
        M = CPD.get_index_matrix(card_parent)
        epsilon = 0.01
        for row in range(0, card_latent):
            for column in range(0, C):
                distance =  sigma*abs(row - M[0][column]) + abs(row  - M[1][column])
                matrix[row][column] = epsilon + (2 - M[2])[column]*math.exp(-distance)
        # Normalize distribution
        for column in range(0, C):
            factor = 0
            for row in range(0, card_latent):
                factor += matrix[row][column]
            for row in range(0, card_latent):
                matrix[row][column] /= factor
        return matrix

    def create_Raptor_distribution(card_latent, card_parent, sigma, modus):
        # CREATES : a CPD with a distribution depending on the "distance" of the latent variable index to the indexes of the parents
        # the distance is the inverse of an exponentional of the sum of the distances corrected with a factor sigma
        # cardinality of the latent must be the same as the cardinality of the parents
        C = np.prod(card_parent)
        matrix = np.full((card_latent, C), 1./card_latent)
        if modus ==  'delta_alfa':
            M = CPD.get_index_matrix(card_parent)
            hi = 100
            lo = 5
            for column in range(0, C):
                '''if M[0][column] ==  M[1][column]:
                    matrix[0][column] = lo
                    matrix[1][column] = hi
                    matrix[2][column] = lo'''
                if M[0][column] - M[1][column] <= 0:
                    matrix[0][column] = hi
                    matrix[1][column] = lo

                if M[0][column] - M[1][column] > 0:
                    matrix[0][column] = lo
                    matrix[1][column] = hi



        if modus ==  'direction':
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




        if modus ==  'correction':
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
            '''for column in range(0, C):
                matrix[0][column] = lo
                matrix[1][column] = hi
                matrix[2][column] = lo
                if M[2][column]>= 0:
                    if M[0][column] == 0 and M[1][column] == 0:
                        matrix[0][column] = hi
                        matrix[1][column] = lo
                        matrix[2][column] = lo
                    if M[0][column] == 2 and M[1][column] == 2:
                        matrix[0][column] = lo
                        matrix[1][column] = lo
                        matrix[2][column] = hi
                if M[2][column] >= 0:
                    if M[0][column] == 2 and M[1][column] == 2:
                        matrix[0][column] = lo
                        matrix[1][column] = lo
                        matrix[2][column] = hi
                    if M[0][column] == 2 and M[1][column] == 2:
                        matrix[0][column] = hi
                        matrix[1][column] = lo
                        matrix[2][column] = lo'''

        # Normalize distribution
        for column in range(0, C):
            factor = 0
            for row in range(0, card_latent):
                matrix[row][column] += random.uniform(-0.005, 0.005)#this to avoid an exact equidistant probabiliy
                factor += matrix[row][column]
            for row in range(0, card_latent):
                matrix[row][column] /= factor
        return matrix

    def create_correction_distribution(card_latent, card_parent, sigma):
        # CREATES : a CPD with a distribution depending on the "distance" of the latent variable index to the indexes of the parents
        # the distance is the inverse of an exponentional of the sum of the distances coorected with a factor sigma
        # cardinality of the latent must be the same as the cardinality of the parents
        C = np.prod(card_parent)
        matrix = np.zeros((card_latent, C))
        M = CPD.get_index_matrix(card_parent)
        epsilon = 0.99
        nu = 1
        for row in range(0, card_latent):
            for column in range(0, C):
                mu = 3 + nu*(M[0][column] - M[1][column])/2
                distance =  sigma*abs(row - mu)
                f_km = epsilon*math.exp(-distance)
                mu = 3
                distance = sigma * abs(row - mu)
                g_km = epsilon * math.exp(-distance)
                matrix[row][column] = 1#(2 - M[2][column])*g_km + M[2][column]*f_km
                #print("Reward = " ,M[2][column], " and f[", row, ",", column, "] = ", f_km, " and g[", row, ",", column, "] = ", g_km, " giving : ",  matrix[row][column])
                #print("f*correction = ", M[2][column]*f_km, " and g*correction = ", (2 - M[2][column])*g_km)
        # Normalize distribution
        for column in range(0, C):
            factor = 0
            for row in range(0, card_latent):
                factor += matrix[row][column]
            for row in range(0, card_latent):
                matrix[row][column] /= factor
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
        # Normalize distribution
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
                    if delta_reward == 0:
                        mu = delta_index
                        y = (row - mu) * (row - mu) / sigma / sigma
                        matrix[row][column] = math.exp(-y)
                    if delta_reward == 2:
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

    def get_random_parent(cardinality):
        table = np.random.rand( cardinality)
        # Normalize distribution
        factor = 0
        for column in range(0, cardinality):
            factor += table[column]
        for column in range(0, cardinality):
            table[column] /= factor
        return table

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
        '''table = np.zeros(cardinality)
        factor = 0
        for x in range(0, cardinality):
            y = (x - mu) * (x - mu) / sigma / sigma
            prob = math.exp(-y)
            factor += prob
            table[x] = prob
        # normalize
        for t in range(0, len(table)):
            table[t] /= factor
        # print(table)'''
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
