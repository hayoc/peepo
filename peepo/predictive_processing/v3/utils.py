import itertools

import numpy as np


def get_index_matrix(cardinality):
    """
    Returns the state combinations of the parent nodes given the cardinality of the parents nodes

    :param cardinality: list with the cardinalities of the parent
    :returns: an array with the combination of all possible states
    :type cardinality: list
    :rtype : np.array

    Example
    -------
    >>> cardinality = [2, 3, 2]
    >>> print(get_index_matrix(cardinality))
    [[0 0 0 0 0 0 1 1 1 1 1 1],
     [0 0 1 1 2 2 0 0 1 1 2 2],
     [0 1 0 1 0 1 0 1 0 1 0 1 ]]
    """
    blocks = []
    for i in range(0, len(cardinality)):
        blocks.append([s for s in range(0, cardinality[i])])
    return np.transpose(np.asarray(list(itertools.product(*blocks))))


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



