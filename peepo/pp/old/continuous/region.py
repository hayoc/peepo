import math

import numpy as np
from sklearn.preprocessing import normalize


class Region:
    """
        The Predictive Processing algorithm used here is based on
        M.W. Spratling (2016): Predictive Coding as a Model of Cognition
        For a non-technical intro, see A. Clark (2013):
        Whatever next? Predictive brains, situated agents,
        and the future of cognitive science

        A region is a small part in the whole hiearchy of the neocortex.
        It takes as input a vector of some values, which can come from
        low level input (raw) mechanisms, such as values given by the retina,
        or they can come from another region from a lower or the same level.
        The output is a vector (usually smaller in size, to decrease
        complexity at a higher level, and to allow some form of generalization)

        A region iterates to reach a prediction model with (in theory)
        zero prediction errors. This is achieved through 3 formulas:

            r = Vy
            e = x (/) max(r, p2)
            y = max(y, p1) (*) We

        Where r are the (m by 1) reconstruction neurons, e the (m by 1)
        error neurons, y the (n by 1)prediction neurons, x the (m by 1)
        input values, p1 and p2 some paramater values to avoid division
        by zero errors and a stagnant model, W is a (m by n) matrix,
        representing the synaptic weights, and V the transpose of V
    """

    def __init__(self, inputs, outputsize, synapseweights=None):
        n = len(inputs)
        m = outputsize

        # (n by 1) input vector
        self.x = inputs
        # (n by 1) neuron vector
        self.r = np.zeros(n)
        # (n by 1) neuron vector
        self.e = np.zeros(n)
        # (m by 1) neuron vector
        self.y = np.zeros(m)

        # synapseweights only have one set of free params(W)
        # since V is the transpose of W
        if synapseweights is None:
            synapseweights = np.zeros((m, n))
        self.w = synapseweights
        self.v = normalize(synapseweights.T, axis=0)

        # p1 makes sure prediction neurons don't become unresponsive
        # p2 avoids divde by zero errors and rate of learning
        self.p2 = math.pow(10, -2)
        self.p1 = self.p2 / np.max(self.v.sum(axis=1))

        # Compare with error neurons, all = 1  = no errors
        self.noerrors = np.ones((n, 1))

    def iterate(self, maxiterations):
        """
        Region should stablize within a certain period
        max is failsafe, if errors are 0, which should
        happen once it stablizes, it stops
        """
        for i in range(0, maxiterations):
            # print("====== " + str(i) + " ========")
            self.reconstruct()
            self.compare()
            if np.array_equal(self.e, self.noerrors):
                print("y = " + str(self.y))
                print("No errors left.")
                break
            self.update()

    def reconstruct(self):
        """
        Prediction (reconstruction) = matrix multiplication
        of synapse weights V and model y
        """
        self.r = self.v @ self.y
        # print("r = " + str(self.r))

    def compare(self):
        """
        Prediction error = element wise division
        of input values by predicted (reconstructed) values
        division instead of subtraction to avoid negative values
        """
        self.e = self.x / np.maximum(self.r, self.p2)
        # print("e = " + str(self.e))

    def update(self):
        """
        Representation model gets updated by element multiplication
        of synapse weights W and prediction error vector matrix multiplication
        """
        self.y = np.maximum(self.y, self.p1) * (self.w @ self.e)
        # print("y = " + str(self.y))