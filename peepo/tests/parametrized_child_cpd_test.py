#version 5/1/2019 10h00
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

"""
Coding comments:
- Try to avoid creating classes when you can. Object-oriented programming (= focusing on the creation of objects, each
    with their own states and properties) is a nice way of programming when needed, but only when needed. In this case
    for example we are writing utility functions, there is no need to create any objects. We want the user of these
    functions just to call the function with certain parameters X and get a result Y back.
- So try to abstract away as much as possible in your functions. You still had to calculate the omega and then loop
    through num_check as a user. You could put all that logic in the function itself, so that the user can call one
    single function very simply, e.g. pdf = mutated_cpd(my_card_child, my_card_parents, my_num_check)

-   self.phase_shift = np.dtype('d')
    self.phase_shift = omega[0]
    -> This doesn't actually do anything. You're assigning the variable to one value (i.e. the dtype) and then you're
    assigning it a new value
- I think what you were trying to do there is just set the type of the array. Like in this line:
    self.omega = np.array(omega,dtype = np.dtype('d'))
  but this is also not necessary I think. 
- This is only for the plotting part but I changed one line from
    ax = fig.gca(projection='3d')
  to 
    ax = Axes3D(fig)
  The reason I did that is because I wanted to remove the import statements you didn't use (you should do that as well,
  you can do it automatically by pressing Ctrl + Alt + O (or go to your top bar to Code > Optimize Imports). It's not
  just a matter of style but also you're importing libraries for no reason which might slow things down.



General Style Comments:
- Try to Reformat your code once in a while, you often have style issues which PyCharm doesn't like (it will display it
    with a yellow squiggly line below it (e.g. this is a green squiggly one: fhdnkfjskdf). You can reformat your code 
    automatically by pressing Ctrl + Alt + L (or go to your top bar to Code > Reformat Code)
- Comment blocks should be done with 3 double quotes, you always use 3 single quotes (i.e. " vs '). These comment
    blocks should be for descriptions of methods and classes. If you want a single line comment somewhere inside a
    method between lines of code, use # (which is much easier than typing 3 single quotes) (See some examples below)

"""


def ga_child_cpd(card_child, card_parents, omega):
    """
    Used in the framework of Genetic Algorithm to initialize a childs cpd and to alter it later (mutation)

    :param card_child: integer = childs' cardinality
    :param card_parents: array containing the cardinality of the childs' parents
    :param omega: an array of dim card_child containing the generating cpd's parameters
    :return: a matrix of shape (card_child, numpy.prod(card_parents) conting the conditional probability distribution

    example :

        my_card_parents = [2, 3, 2]
        my_card_child = 3
        max_omega = 2 * math.pi * np.prod(my_card_parents)

        my_omega = [2.0,0.1,0.9] -> only for this example. As a rule one will use
            my_omega = np.random.rand(my_card_child) * max_omega to initialize (0th generation)
            a child's cpd and
            my_omega += (0.5 - np.random.rand(my_card_child))*epsilon with espilon small (e.g. 0.01)
            when mutating the childs' cpd

        my_pdf = ga_child_cpd(my_card_child, my_card_parents, my_omega)

        print(my_pdf)
        ->
            [[0.07 0.25 0.53 0.18 0.14 0.37 0.18 0.09 0.54 0.46 0.06 0.45]
             [0.56 0.62 0.47 0.67 0.49 0.28 0.35 0.48 0.35 0.54 0.69 0.24]
             [0.37 0.13 0.00 0.15 0.37 0.35 0.47 0.44 0.11 0.00 0.25 0.31]]
    """

    phase_shift = omega[0]
    n_comb = np.prod(card_parents)  # type: int
    pdf = []
    for row, ang in enumerate(omega):
        pdf_row = []
        for col in range(n_comb):
            pdf_row.append(math.sin(ang * (col + 1) + phase_shift) + 1)
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
    my_omega = [2.0,0.1,0.9]
    my_pdf = ga_child_cpd(my_card_child, my_card_parents, my_omega)
    np.set_printoptions(formatter={'float': lambda my_pdf: "{0:0.2f}".format(my_pdf)})
    print(my_pdf)

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
