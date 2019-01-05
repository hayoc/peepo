<<<<<<< HEAD
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

        return self.normalize_distribution(np.asarray(pdf))

    def normalize_distribution(self,matrix):
        factor = np.sum(matrix, axis = 0)
        return matrix/factor



if __name__ == '__main__':
    '''example : 3 parents and cardinality child = 4'''
    card_parents = [2, 2]
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
=======
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


def mutated_cpd(card_child, card_parents, num_check=1):
    """
    Example function comment/description

    :param card_child:
    :param card_parents:
    :param num_check:
    :return:
    """
    max_omega = 2 * math.pi * np.prod(card_parents)
    total_pdf = np.full((card_child, np.prod(card_parents)), 0.0)
    for n in range(num_check):
        omega = np.random.rand(card_child) * max_omega
        phase_shift = omega[0]
        n_comb = np.prod(card_parents)  # type: int

        pdf = []
        for row, ang in enumerate(omega):
            pdf_row = []
            for col in range(n_comb):
                pdf_row.append(math.sin(ang * (col + 1) + phase_shift) + 1)
            pdf.append(pdf_row)

        # An example inline comment
        total_pdf += normalize_distribution(np.asarray(pdf))

    total_pdf = np.asarray(total_pdf)
    total_pdf /= num_check

    return total_pdf


def normalize_distribution(matrix):
    """
    Another one

    :param matrix:
    :return:
    """
    factor = np.sum(matrix, axis=0)
    return matrix / factor


if __name__ == '__main__':
    # example : 3 parents and cardinality child = 4
    my_card_parents = [2, 3, 2]
    my_card_child = 3
    my_num_check = 1

    my_pdf = mutated_cpd(my_card_child, my_card_parents, my_num_check)

    # ----------- PLOTTING ------------------
    fig = plt.figure()
    ax = Axes3D(fig)

    cmb = np.prod(my_card_parents)  # type: int
    x = np.arange(0, cmb, 1)
    datasets = [{"x": x, "y": np.full(len(x), row).astype(int), "z": my_pdf[row],
                 "colour": (random.uniform(0.2, 1), random.uniform(0, 0.5), random.uniform(0, 1), 1)} for row in
                range(my_card_child)]
>>>>>>> origin/model_update_bernard
    for dataset in datasets:
        ax.plot(dataset["x"], dataset["y"], dataset["z"], color=dataset["colour"])
    ax.set_xlabel('parents state combinations')
    ax.set_ylabel('child states')
<<<<<<< HEAD
    ax.set_zlabel('phi states child');
    ax.set_zlim((0, 1))
    plt.yticks(np.arange(0, len(omega), step=1))
    plt.show()
=======
    ax.set_zlabel('phi states child')
    ax.set_zlim((0, 1))
    plt.yticks(np.arange(0, my_card_child, step=1))
    plt.show()

# import numpy as np
# from mpl_toolkits.mplot3d.axes3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import matplotlib.tri as tri
# from functools import reduce
# import math
# import random
# ''' Method to be used to haven a control how cpd are shaped and distributed when using Genetic Algorithms'''
# class Random_Child(object):
#     def __init__(self, omega, card_parents):
#         self.omega = np.array(omega,dtype = np.dtype('d'))
#         self.phase_shift = np.dtype('d')
#         self.phase_shift = omega[0]
#         self.card_parents =  card_parents
#         self.card_child = len(self.omega)
#         self.n_comb = np.prod(self.card_parents)
#
#     def pdf(self):
#         '''Returns pdf value '''
#         pdf = []
#         for row, ang in enumerate(omega):
#             pdf_row = []
#             for col in range(self.n_comb):
#                 pdf_row.append(math.sin(ang*(col+1) + self.phase_shift)+1)
#             pdf.append(pdf_row)
#         '''normalize'''
#
#         return self.normalize_distribution(pdf)
#
#     def normalize_distribution(self,matrix):
#         R = np.size(matrix,0)
#         C = np.size(matrix,1)
#         for column in range(0, C):
#             factor = 0
#             for row in range(0, R):
#                 factor += matrix[row][column]
#             for row in range(0, R):
#                 matrix[row][column] /= factor
#         return matrix
#
#
#
# if __name__ == '__main__':
#     '''example : 3 parents and cardinality child = 4'''
#     card_parents = [2, 3, 2]
#     card_child = 3
#     max_omega = np.dtype('d')
#     max_omega =  2 * math.pi * np.prod(card_parents)
#     pdf = np.full((card_child, np.prod(card_parents)), 0.0)
#     num_check = 1
#     '''iterate num_check times to check the uniformity of the distribution
#             ->  should have a uniform distribution over all combinations when num_check >>'''
#     for n in range(num_check):
#         omega = np.random.rand(card_child)*max_omega
#         pdf += Random_Child(omega, card_parents).pdf()
#     pdf = np.asarray(pdf)
#     pdf /= num_check
#     print(pdf)
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     x = np.arange(0, np.prod(card_parents), 1)
#     datasets = [{"x": x, "y": np.full(len(x), row).astype(int), "z": pdf[row], "colour": (random.uniform(0.2,1),random.uniform(0,0.5),random.uniform(0,1),1)} for row  in range(len(omega))]
#     for dataset in datasets:
#         ax.plot(dataset["x"], dataset["y"], dataset["z"], color=dataset["colour"])
#     ax.set_xlabel('parents state combinations')
#     ax.set_ylabel('child states')
#     ax.set_zlabel('phi states child');
#     ax.set_zlim((0, 1))
#     plt.yticks(np.arange(0, len(omega), step=1))
#     plt.show()
>>>>>>> origin/model_update_bernard
