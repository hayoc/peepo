import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from functools import reduce
import math
import random
''' Method to be used to haven a control how cpd are shaped and distributed when using Genetic Algorithms'''
class Random_Child(object):
    def __init__(self, omega, card_parents):
        self.omega = np.array(omega)
        self.phase_shift = 2*math.pi*omega[0]
        self.card_parents =  card_parents
        self.card_child = len(self.omega)
        self.n_comb = np.prod(self.card_parents)

    def pdf(self):
        '''Returns pdf value '''
        pdf = []
        for row, ang in enumerate(omega):
            pdf_row = []
            for col in range(self.n_comb):
                pdf_row.append(math.sin(ang*(col+1) + self.phase_shift*(row+1))+1)
            pdf.append(pdf_row)
        '''normalize'''

        return self.normalize_distribution(pdf)

    def normalize_distribution(self,matrix):
        R = np.size(matrix,0)
        C = np.size(matrix,1)
        for column in range(0, C):
            factor = 0
            for row in range(0, R):
                factor += matrix[row][column]
            for row in range(0, R):
                matrix[row][column] /= factor
        return matrix



if __name__ == '__main__':
    '''example : 3 parents and cardinality child = 4'''
    card_parents =  [2,3,2]
    max_omega = 2*math.pi/np.prod(card_parents)
    omega = [random.uniform(0, max_omega), random.uniform(0, max_omega), random.uniform(0, max_omega),random.uniform(0, max_omega)]
    pdf = Random_Child(omega, card_parents).pdf()
    print(pdf)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(0, np.prod(card_parents), 1)
    datasets = [{"x": x, "y": np.full(len(x), row), "z": pdf[row], "colour": "blue"} for row  in range(len(omega))]
    for dataset in datasets:
        ax.plot(dataset["x"], dataset["y"], dataset["z"], color=dataset["colour"])
    plt.show()
