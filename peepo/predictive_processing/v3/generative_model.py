import logging

import numpy as np
import networkx as nx
from pgmpy.base import DirectedGraph
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from scipy.stats import entropy

def default_model():
    """
    Method to create a default initialized Bayesian network.

    :return: Bayesian network with two dependent nodes (A & B) with 50-50 parameters

    :rtype: BayesianModel
    """
    model = BayesianModel([('A', 'B')])

    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.5, 0.5]])
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.5, 0.5],
                                                              [0.5, 0.5]], evidence=['A'], evidence_card=[2])
    model.add_cpds(cpd_a, cpd_b)
    model.check_model()

    return model


class GenerativeModel:
    """
    Predictive Processing Generative Model

    This is a generative model, implemented as a Bayesian Causal Network. The three functions
    prediction, prediction error and prediction error minimization are defined for this model.

    :param sensory_input : Mutable dictionary containing the current sensory inputs
    :param model : Bayesian Causal Network. Causes are hypothesis variables, effects are observational variables.

    :type sensory_input : dict
    :type model : BayesianModel
    """

    def __init__(self, sensory_input, model=default_model()):
        self.sensory_input = sensory_input
        self.model = model
        self.infer = VariableElimination(model)

    def process(self):
        """
        Processes one flow in the predictive processing algorithm:
            1) prediction
            2) prediction error
            3) prediction error minimization
        """
        for node, prediction in self.predict().items():
            pred = prediction.values
            obs = self.sensory_input[node]
            pes = self.error_size(pred, obs)

            # TODO: Precision weighting
            if pes > 0:
                logging.debug("node[%s] with prediction-error-size %s ||| predicted %s -vs- %s observed", node, pes,
                              pred, obs)
                pe = self.error(pred, obs)
                self.error_minimization(node=node, prediction_error_size=pes, prediction_error=pe, prediction=pred)
            else:
                logging.debug("node[%s] no prediction-error", node)

    def predict(self):
        """
        Predicts the leaf nodes (i.e. the observational nodes) based on the parent nodes (i.e. the hypothesis nodes)
        The predicted values of the observations are the maximum a posteriori (MAP) distributions over the
        hypotheses

        :return: predictions for all observation variables

        :rtype: dict
        """
        return self.infer.query(variables=self.model.get_leaves(), evidence=self.get_hypotheses())

    @staticmethod
    def error(pred, obs):
        """
        Calculates the prediction error as the residual of subtracting the predicted inputs from the observed inputs

        :param pred: predicted sensory inputs
        :param obs: observed sensory inputs
        :return: prediction error

        :type pred : np.array
        :type obs : np.array
        :rtype : np.array
        """
        return obs - pred

    @staticmethod
    def error_size(pred, obs):
        """
        Calculates the size of the prediction error as the Kullback-Leibler divergence. This responds the magnitude
        of the prediction error, how wrong the prediction was.

        :param pred: predicted sensory inputs
        :param obs: observed sensory inputs
        :return: prediction error size

        :type pred : np.array
        :type obs : np.array
        :rtype : float
        """
        return entropy(obs, pred)

    def error_minimization(self, node, prediction_error_size, prediction_error, prediction):
        """
        Attempts to minimize the prediction error by one of the possible PEM methods:
            1) Hypothesis Update
            2) Model Update

        :param node: name of the node causing the prediction error
        :param prediction_error_size: size of the prediction error
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error

        :type node : str
        :type prediction_error_size: float
        :type prediction_error: np.array
        :type prediction: np.array
        """
        # TODO: Model update (and possible other PEM methods)
        # PEM 1: Hypothesis Update
        self.hypothesis_update(node, prediction_error, prediction)

    def hypothesis_update(self, node, prediction_error, prediction):
        """
        Updates the hypotheses of the generative model to minimize prediction error

        :param node: name of the node causing the prediction error
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error

        :type node : str
        :type prediction_error: np.array
        :type prediction: np.array
        """
        for hypo in self.model.get_roots():
            result = self.infer.query(variables=[hypo],
                                      evidence={node: np.argmax(
                                          prediction_error if prediction is None else prediction_error + prediction)})
            self.model.get_cpds(hypo).values = result.get(hypo).values
            logging.debug("node[%s] hypothesis-update to %s", hypo, result.get(hypo).values)
        # Should we update hypothesis variables based on only prediction error node?
        # Or all observation nodes in the network???

    def get_hypotheses(self):
        hypos = {}
        for root in self.model.get_roots():
            hypos.update({root: np.argmax(self.model.get_cpds(root).values)})
        return hypos

    def get_observations(self):
        obs = {}
        for leaf in self.model.get_leaves():
            obs.update({leaf: np.argmax(self.model.get_cpds(leaf).values)})
        return obs
