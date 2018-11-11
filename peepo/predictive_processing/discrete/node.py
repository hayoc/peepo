import logging
import uuid
import numpy as np
from scipy.stats import entropy


class Node:
    def __init__(self, likelihood, motor_command=None, children=[], precision=1.0, confidence=0.1, name=uuid.uuid4(), *args):
        """
        Predictive Processing Node:

        :param likelihood: Likelihood matrix containing the likelihoods
         for all predictions given hypotheses.

        :param motor_command: Motor command. Execute function will be called
         when threshold is not reached for a model update.

        :param children: List of child Nodes.

        :param precision: Precision Threshold to decide between Motor Command and Hypothesis Update.

        :param confidence: Confidence Threshold for prediction error calculation.

        :param name: Unique identifier for Node, for debugging/logging purposes.

        :param *args: Arguments to be passed to Motor Command.

        :type motor_command: motor_command
        :type lm: numpy.matrix
        :type children: list
        :type th: float
        :type name: str
        """

        self.lm = likelihood
        self.mc = motor_command
        self.mcargs = args
        self.children = children
        self.hyps = {}
        self.hyp = None
        self.numprd = likelihood.shape[0]
        self.numhyp = likelihood.shape[1]
        self.prec = precision
        self.conf = confidence
        self.name = name
        self.nonzero = 0.00001

    def predict(self):
        """
        Prediction Step in Predictive Processing.
         Based on the Likelihood Matrix and the values of the hypotheses,
         a prediction is returned.

        :return: Array of Predictions, precision, confidence
        :rtype: numpy.array
        """
        idx = np.argmax(self.hyp)

        logging.debug('PP [%s] Prediction: %s', self.name, str(self.lm[:, idx].A1.tolist()))

        return self.lm[:, idx].A1

    def error(self, prd, act):
        """
        Prediction Error Calculation Step in Predictive Processing.
         Given the predicted and actual values, the Kullback-Leibler
         Divergence is calculated. If it exceeds the confidence threshold,
         an update to the hypotheses must happen.

        :param prd: Predicted values.

        :param act: Actual values (sensory input, or hypotheses from child region).

        :return: True or False, indicating whether hypotheses should
         be updated or not.

        :type prd: numpy.array
        :type act: numpy.array
        :rtype Boolean
        """

        dkl = entropy(np.around(prd, decimals=10), np.around(act, decimals=10))

        logging.debug('PP [%s] Error: %s', self.name, str(dkl))

        return dkl > self.conf

    def update(self, act):
        """
        Depending on the Precision value, either a motor command is triggered,
         or a hypothesis update.
        Given the actual values (E), the hypotheses (H) of the node (posterior)
         are updated using Bayesian Updating.
         P(H|E) = (P(H) * P(E|H))/P(E)

        :param act: Array of Actual Values (E)
        """

        self.motor_command(act) if self.prec < 0.5 else self.hypothesis_update(act)

    def motor_command(self, act):
        """
        Executes motor command provided to the node.
        """
        if self.motor_command is None:
            raise NotImplementedError('Motor Command not specified in Node {} with Precision {}', self.name, self.prec)
        else:
            self.mc(act, self.mcargs)

    def hypothesis_update(self, act):
        """
        Given the actual values (E), the hypotheses (H) of the node (posterior)
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
        """
        Validates the Node Hypotheses.
         Alters the hypothesis values slightly in case one or more
         hypotheses are 0.0 or 1.0.
         Hypothesis values must add op to 1.0.
        """
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

    def setHyp(self, name, hyp):
        """
        Adds an array of hypotheses to the dictionary of hypothesis nodes.
         In case a node has multiple parents, the hypothesis of the current
         node is calculated by averaging over the predictions of all the
         parent nodes.

        :param name: Name of parent node for which to add predictions
        :param hyp: Array of predictions from parent node

        :type name: str
        :type hyp: numpy.array
        """
        self.hyps[name] = hyp
        self.hyp = np.sum(list(self.hyps.values()), axis=0) / len(self.hyps)
        self.validate()

    def getHyp(self):
        return np.copy(self.hyp)

    def __str__(self):
        return 'Node ' + self.name

    def __repr__(self):
        s = 'discrete.node({}, {}, {}, {}, {})'.format(str(self.lm.tolist()), str(self.hyp), self.prec, self.conf, self.name)
        return s