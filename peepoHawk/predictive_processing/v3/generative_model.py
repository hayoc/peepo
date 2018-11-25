#  25/11
import logging
import math
import random

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import entropy

from peepoHawk.visualize.graph import draw_network


class GenerativeModel:
    """
    Predictive Processing Generative Model

    This is a generative model, implemented as a Bayesian Causal Network. The three functions
    prediction, prediction error and prediction error minimization are defined for this model.

    :param sensory_input : Mutable dictionary containing the current sensory inputs
    :param model : Bayesian Causal Network. Causes are hypothesis variables, effects are observational variables.

    :type sensory_input : SensoryInput
    :type model : [BayesianModel]
    """

    MAX_NODES = 10

    def __init__(self, sensory_input, model):
        self.sensory_input = sensory_input
        self.model = model
        #self.atomic_updates = [self.add_node, self.add_edge, self.change_parameters] #TODO: Add change_valency
        draw_network(model)

    def process(self):
        """
        Processes one flow in the predictive processing algorithm:
            1) prediction
            2) prediction error
            3) prediction error minimization
        Returns the total prediction error size observed (for informational purposes...)
        """
        total_pes = 0
        correction = [0.2,0.8]
        for node, prediction in self.predict(self.model).items():
            pred = prediction.values
            if 'Correction' in node:
                #print("Correction case : ", pred )
                correction = pred
        return total_pes,correction

    def predict(self, model):
        """
        Predicts the leaf nodes (i.e. the observational nodes) based on the parent nodes (i.e. the hypothesis nodes)
        The predicted values of the observations are the maximum a posteriori (MAP) distributions over the
        hypotheses

        :return: predictions for all observation variables

        :rtype: dict
        """

        infer = VariableElimination(model)
        return infer.query(variables=model.get_leaves(), evidence=self.get_hypotheses(model))

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

    def error_minimization(self, node, precision, prediction_error, prediction):
        """
        Attempts to minimize the prediction error by one of the possible PEM methods:
            1) Hypothesis Update
            2) Model Update

        :param node: name of the node causing the prediction error
        :param precision: precision of the prediction error
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error

        :type node : str
        :type precision: float
        :type prediction_error: np.array
        :type prediction: np.array
        """
        self.hypothesis_update(node, prediction_error, prediction)
        # TODO: make the choice more sophisticated, with precision, surprise, yada yada yada
        # if precision < 0.5:
        #     self.model_update(node, prediction_error, prediction)
        # else:
        #     self.hypothesis_update(node, prediction_error, prediction)


    @staticmethod
    def get_hypotheses(model):
        hypos = {}
        for root in model.get_roots():
            hypos.update({root: np.argmax(model.get_cpds(root).values)})
        print("Hypos : ", hypos)
        return hypos

    @staticmethod
    def get_observations(model):
        obs = {}
        for leaf in model.get_leaves():
            obs.update({leaf: np.argmax(model.get_cpds(leaf).values)})
        return obs



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
        # Theoretically speaking a hypothesis update should achieve both perceptual and motor update
        # Currently in the implementation we make the difference explicit
        # TODO: Need to have custom implementation of bayesian network, so that prediction errors in proprioceptive
        # TODO: nodes (motor) are resolved by executing the motor action, and not performing hypo update
        infer = VariableElimination(self.model)
        if "motor" in node:
            self.sensory_input.action(node, prediction_error, prediction)
        else:
            for hypo in self.model.get_roots():
                result = infer.query(variables=[hypo],
                                     evidence={node: np.argmax(prediction_error + prediction)})
                before = self.model.get_cpds(hypo).values
                #print("Result ")
                #print(result.get(hypo))
                self.model.get_cpds(hypo).values = result.get(hypo).values
                logging.debug("node[%s] hypothesis-update from %s to %s", hypo, before, result.get(hypo).values)
            # Should we update hypothesis variables based on only prediction error node?
            # Or all observation nodes in the network???

