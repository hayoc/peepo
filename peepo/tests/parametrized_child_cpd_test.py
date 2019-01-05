#version 5/1/2019 11h00
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D


def ga_child_cpd(card_parents, omega):
    """
    Used in the framework of Genetic Algorithm to initialize a childs cpd and to alter it later (mutation)
    The word 'child" used here is in the context of Bayesian network and is not an offspring of a GA


    :param card_parents: array containing the cardinality of the childs' parents
    :param omega: an array of dim cardynality of the child (card_child) containing the cpd's  generating parameters
    :return: a matrix of shape (card_child, numpy.prod(card_parents) containing the conditional probability distribution

    example :

        my_card_parents = [2, 3, 2]
        max_omega = 2 * math.pi * np.prod(my_card_parents)

        zero generation : (my_card_child = 3)
        my_omega = [2.0,0.1,0.9] -> only for this example. As a rule one will use
            my_omega = np.random.rand(my_card_child) * max_omega to initialize (0th generation)
            a child's cpd and
            my_omega += (0.5 - np.random.rand(my_card_child))*epsilon with espilon small (e.g. 0.05)
            when mutating the childs' cpd

        my_pdf = ga_child_cpd( my_card_parents, my_omega)

        mutation :
        epsilon = 0.05
        my_omega += (0.5 - np.random.rand(len(my_omega))*epsilon
        my_pdf_ = ga_child_cpd( my_card_parents, my_omega)

        --->
            Zero generation cpd
            [[0.07 0.25 0.53 0.18 0.14 0.37 0.18 0.09 0.54 0.46 0.06 0.45]
             [0.56 0.62 0.47 0.67 0.49 0.28 0.35 0.48 0.35 0.54 0.69 0.24]
             [0.37 0.13 0.00 0.15 0.37 0.35 0.47 0.44 0.11 0.00 0.25 0.31]]


            Mutated generation cpd
            [[0.08 0.23 0.54 0.21 0.13 0.37 0.20 0.06 0.55 0.55 0.03 0.47]
             [0.55 0.62 0.46 0.65 0.50 0.27 0.32 0.45 0.32 0.45 0.71 0.21]
             [0.37 0.14 0.00 0.13 0.38 0.36 0.47 0.49 0.13 0.00 0.26 0.33]]


            Delta zero generation - mutated cpd
            [[-0.01 0.01 -0.01 -0.03 0.02 -0.01 -0.03 0.03 -0.01 -0.08 0.04 -0.02]
             [0.01 -0.00 0.01 0.02 -0.01 0.01 0.03 0.03 0.03 0.09 -0.03 0.03]
             [-0.00 -0.01 -0.00 0.01 -0.01 -0.00 -0.00 -0.05 -0.03 -0.00 -0.01 -0.01]]
    """

    phase_shift = omega[0]
    n_comb = np.prod(card_parents)  # type: int
    pdf = []
    for ang in omega:
        pdf_row = []
        for col in range(n_comb):
            pdf_row.append(math.sin(ang * (col + 1) + phase_shift) + 1.2)
        pdf.append(pdf_row)
    return normalize_distribution(np.asarray(pdf))


def normalize_distribution(matrix):
    """
    Normalizes the columns of a matrix (i.e. sum matrix[:,i] = 1

    :param matrix: numpy 2 dimensional array
    :return: numpy 2 dimensional normalized array
    """
    factor = np.sum(matrix, axis=0)
    return matrix / factor


if __name__ == '__main__':
    # example : 3 parents and cardinality child = 4
    my_card_parents = [2, 3, 2]
    my_card_child = 3
    max_omega = 2 * math.pi * np.prod(my_card_parents)
    my_omega = np.random.rand(my_card_child) * max_omega
    # my_omega = [2.0, 0.1, 0.9]
    my_pdf = ga_child_cpd( my_card_parents, my_omega)
    np.set_printoptions(formatter={'float': lambda my_pdf: "{0:0.2f}".format(my_pdf)})
    print('Zero generation cpd')
    print(my_pdf)
    epsilon = 0.05
    my_omega += (0.5 - np.random.rand(my_card_child))*epsilon
    my_pdf_ = ga_child_cpd(my_card_parents, my_omega)
    np.set_printoptions(formatter={'float': lambda my_pdf_: "{0:0.2f}".format(my_pdf_)})
    print('\n\nZero generation cpd')
    print(my_pdf_)
    print('\n\nDelta mutated cpd')
    print(my_pdf - my_pdf_)

    # ----------- PLOTTING ------------------
    fig = plt.figure()
    ax = Axes3D(fig)

    cmb = np.prod(my_card_parents)  # type: int
    x = np.arange(0, cmb, 1)
    datasets = [{"x": x, "y": np.full(len(x), row).astype(int), "z": my_pdf[row],
                 "colour": (random.uniform(0.2, 1), random.uniform(0, 0.5), random.uniform(0, 1), 1)} for row in
                range(my_card_child)]
    for dataset in datasets:
        ax.plot(dataset["x"], dataset["y"], dataset["z"], color=dataset["colour"])
    ax.set_xlabel('parents state combinations')
    ax.set_ylabel('child states')
    ax.set_zlabel('phi states child')
    ax.set_zlim((0, 1))
    plt.yticks(np.arange(0, my_card_child, step=1))
    plt.show()
