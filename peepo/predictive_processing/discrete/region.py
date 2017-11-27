import logging
import uuid
import numpy as np
from scipy.stats import entropy


class Region:

    def __init__(self, lm, hyp=None, th=0.1, name=uuid.uuid4()):
        """
        Predictive Processing Region:

        :param lm: Likelihood matrix containing the likelihoods
         for all predictions given hypotheses.

        :param hyp: Optional list of hypothesis values to start with.
         Length must match column length of param lm.

        :param th: Threshold for prediction error calculation.

        :param name: Unique identifier for Region, for debugging/logging purposes.

        :type lm: numpy.matrix
        :type hyp: numpy.array
        :type th: float
        :type name: str
        """

        self.lm = lm
        self.numprd = lm.shape[0]
        self.numhyp = lm.shape[1]
        self.hyp = np.full(self.numhyp, 0.5) if hyp is None else hyp
        self.th = th
        self.name = name
        self.nonzero = 0.00001
        self.validate()

    def predict(self):
        """
        Prediction Step in Predictive Processing.
         Based on the Likelihood Matrix and the values of the hypotheses,
         a prediction is returned.

        :return: Array of Predictions.
        :rtype: numpy.array
        """
        idx = np.argmax(self.hyp)

        logging.debug('PP [%s] Prediction: %s', self.name, str(self.lm[:, idx].A1.tolist()))

        return self.lm[:, idx].A1

    def error(self, prd, act):
        """
        Prediction Error Calculation Step in Predictive Processing.
         Given the predicted and actual values, the Kullback-Leibler
         Divergence is calculated. If it exceeds a threshold, an
         update to the hypotheses must happen.

        :param prd: Predicted values.

        :param act: Actual values.

        :return: True or False, indicating whether hypotheses should
         be updated or not.

        :type prd: numpy.array
        :type act: numpy.array
        :rtype Boolean
        """

        dkl = entropy(prd, act)

        logging.debug('PP [%s] Error: %s', self.name, str(dkl))

        return dkl > self.th

    def update(self, act):
        """
        Given the actual values (E), the hypotheses (H) of the region (posterior)
         are updated using Bayesian Updating.
         P(H|E) = (P(H) * P(E|H))/P(E)

        :param act: Array of Actual Values (E)

        """

        e = np.argmax(act)

        mrglik = 0
        for idx, hyp in enumerate(self.hyp):
            hyp = self.nonzero if hyp == 0 else hyp
            lik = self.nonzero if self.lm.item((e, idx)) == 0 else self.lm.item((e, idx))
            mrglik += hyp * lik

        logging.debug('PP [%s] Prior: %s', self.name, str(self.hyp.tolist()))

        for idx, hyp in enumerate(self.hyp):
            hyp = self.nonzero if hyp == 0 else hyp
            lik = self.nonzero if self.lm.item((e, idx)) == 0 else self.lm.item((e, idx))
            self.hyp[idx] = (hyp * lik) / mrglik

        logging.debug('PP [%s] Posterior: %s', self.name, str(self.hyp.tolist()))

    def validate(self):
        zeros = []
        ones = []

        for idx, hyp in enumerate(self.hyp):
            if hyp == 0:
                zeros.append(idx)
            elif hyp == 1:
                ones.append(idx)

        if len(zeros) == len(ones):
            for idx in zeros:
                self.hyp[idx] += self.nonzero
            for idx in ones:
                self.hyp[idx] -= self.nonzero
        elif zeros:
            for idx in zeros:
                self.hyp[idx] += self.nonzero
                if idx > 0 and idx - 1 not in zeros:
                    self.hyp[idx - 1] -= self.nonzero
                else:
                    for x in range(0, len(self.hyp)):
                        if idx + x not in zeros:
                            self.hyp[idx + x] -= self.nonzero
                            break

        sum = 0
        for hyp in self.hyp:
            sum += hyp
        if not sum == 1.0:
            logging.error('Hypothesis Distribution must add up to 1.0! Found: ' + str(sum))

    def setHyp(self, hyp):
        self.hyp = np.copy(hyp)

    def getHyp(self):
        return np.copy(self.hyp)

    def __str__(self):
        return 'Region ' + self.name

    def __repr__(self):
        s = 'discrete.region({}, {}, {}, {})'.format(str(self.lm.tolist()), str(self.hyp), self.th, self.name)
        return s
