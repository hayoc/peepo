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
        self.omega = np.array(omega,dtype = np.dtype('d'))
        self.phase_shift = np.dtype('d')
        self.phase_shift = omega[0]
        self.card_parents =  card_parents
        self.card_child = len(self.omega)
        self.n_comb = np.prod(self.card_parents)

    def pdf(self):
        '''Returns pdf value '''
        pdf = []
        for row, ang in enumerate(omega):
            pdf_row = []
            for col in range(self.n_comb):
                pdf_row.append(math.sin(ang*(col+1) + self.phase_shift)+1)
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
    card_parents = [2, 3, 2]
    card_child = 3
    max_omega = np.dtype('d')
    max_omega =  2 * math.pi * np.prod(card_parents)
    pdf = np.full((card_child, np.prod(card_parents)), 0.0)
    num_check = 1
    '''iterate num_check times to check the uniformity of the distribution
            ->  should have a uniform distribution over all combinations when num_check >>'''
    for n in range(num_check):
        omega = np.random.rand(card_child)*max_omega
        pdf += Random_Child(omega, card_parents).pdf()
    pdf = np.asarray(pdf)
    pdf /= num_check
    print(pdf)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(0, np.prod(card_parents), 1)
    datasets = [{"x": x, "y": np.full(len(x), row).astype(int), "z": pdf[row], "colour": (random.uniform(0.2,1),random.uniform(0,0.5),random.uniform(0,1),1)} for row  in range(len(omega))]
    for dataset in datasets:
        ax.plot(dataset["x"], dataset["y"], dataset["z"], color=dataset["colour"])
    ax.set_xlabel('parents state combinations')
    ax.set_ylabel('child states')
    ax.set_zlabel('phi states child');
    ax.set_zlim((0, 1))
    plt.yticks(np.arange(0, len(omega), step=1))
    plt.show()
