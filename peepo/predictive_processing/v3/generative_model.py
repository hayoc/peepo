import logging
import random

import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from scipy.stats import entropy


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

    def __init__(self, sensory_input, model):
        self.sensory_input = sensory_input
        self.model = model
        self.infer = VariableElimination(model)
        self.atomic_updates = [self.add_node, self.add_edge, self.change_parameters]

    def process(self):
        """
        Processes one flow in the predictive processing algorithm:
            1) prediction
            2) prediction error
            3) prediction error minimization
        Returns the total prediction error size observed (for informational purposes...)
        """
        total_pes = 0
        for node, prediction in self.predict().items():
            pred = prediction.values
            obs = self.sensory_input.value(node)
            pes = self.error_size(pred, obs)

            # TODO: PEM should only happen if PES is higher than some value, this value
            # TODO: should depend on whatever context the agent finds itself in, and the agent's goal
            precision = entropy(pred, base=2)
            pe = self.error(pred, obs)
            total_pes += pes
            if pes > 0.5:
                logging.debug("node[%s] prediction-error ||| predicted %s -vs- %s observed", node, pred, obs)
                logging.debug("node[%s] PES: %s", node, pes)
            self.error_minimization(node=node, precision=precision, prediction_error=pe, prediction=pred)

        return total_pes

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
        # self.hypothesis_update(node, prediction_error, prediction)
        if precision < 0.5:
            self.model_update(node, prediction_error, prediction)
        else:
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
        # Theoretically speaking a hypothesis update should achieve both perceptual and motor update
        # Currently in the implementation we make the difference explicit
        # TODO: Need to have custom implementation of bayesian network, so that prediction errors in proprioceptive
        # TODO: nodes (motor) are resolved by executing the motor action, and not performing hypo update
        if "motor" in node:
            self.sensory_input.action(node, prediction_error, prediction)
        else:
            for hypo in self.model.get_roots():
                result = self.infer.query(variables=[hypo],
                                          evidence={node: np.argmax(
                                              prediction_error if prediction is None
                                              else prediction_error + prediction)})
                before = self.model.get_cpds(hypo).values
                self.model.get_cpds(hypo).values = result.get(hypo).values
                logging.debug("node[%s] hypothesis-update from %s to %s", hypo, before, result.get(hypo).values)
            # Should we update hypothesis variables based on only prediction error node?
            # Or all observation nodes in the network???

    def model_update(self, node, prediction_error, prediction):
        random.choice(self.atomic_updates)(node)

    def add_node(self, node_in_error):
        pass

    def add_edge(self, node_in_error):
        pass

    def change_parameters(self, node_in_error):
        pass

    def change_valency(self, node_in_error):
        pass

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
