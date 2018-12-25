import logging
import random

import numpy as np
from pgmpy.inference import VariableElimination
from scipy.stats import entropy




class GenerativeModel:
    """
    Predictive Processing Generative Model
    This is a generative model, implemented as a Bayesian Causal Network. The three functions
    prediction, prediction error and prediction error minimization are defined for this model.
    :param sensory_input : Mutable dictionary containing the current sensory inputs
    :param network : Bayesian Causal Network. Causes are hypothesis variables, effects are observational variables.
    :type sensory_input : SensoryInput
    :type network : [BayesianModel]
    TODO: Model Update, e.g. through: self.atomic_updates = [self.add_node, self.add_edge, self.change_parameters]
    TODO: Integrate PRECISION BASED WEIGHTING on prediction errors. E.g. prediction error minimization should only
    happen if the prediction errors have enough weight assigned to them. This can depend on context, the organism's
    goal, or other ways.
    TODO: Implement a custom BayesianNetwork so we can distinguish between action and perception nodes. Instead of
    distinguishing them by checking for 'motor' in the name.
    """

    MAX_NODES = 10

    def __init__(self, sensory_input, network):
        self.sensory_input = sensory_input
        self.network = network.copy()


    def process(self):
        """
        Processes one flow in the predictive processing algorithm:
            1) prediction
            2) prediction error
            3) prediction error minimization (hypothesis or model update)
        Returns the total prediction error size observed (for informational purposes...)
        """
        total_prediction_error_size = 0

        for node in self.network.get_leaves():
            prediction = self.predict(node)
            observation = self.sensory_input.value(node)
            prediction_error_size = self.error_size(prediction, observation)
            prediction_error = self.error(prediction, observation)
            precision = entropy(prediction, base=2)
            total_prediction_error_size += prediction_error_size



        return total_prediction_error_size

    def predict(self, node):
        """
        Predicts the given leaf node (i.e. the observational node) based on the root nodes (i.e. the belief nodes)
        :return: prediction for given observation variable, a prediction is a probability distribution
        :rtype: np.array
        """
        infer = VariableElimination(self.network)
        evidence = self.get_root_nodes(self.network)
        evidence = {k: v for k, v in evidence.items() if k not in [node]}

        return infer.query(variables=[node], evidence=evidence)[node].values

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
        :param precision: precision of the prediction
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error
        :type node : str
        :type precision: float
        :type prediction_error: np.array
        :type prediction: np.array
        """
        self.hypothesis_update(node, prediction_error, prediction)

    def hypothesis_update(self, leaf_node, prediction_error, prediction):
        """
        Updates the hypotheses of the generative model to minimize prediction error
        :param leaf_node: name of the node causing the prediction error
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error
        :type leaf_node : str
        :type prediction_error: np.array
        :type prediction: np.array
        """
        infer = VariableElimination(self.network)
        if "motor" in leaf_node:
            self.sensory_input.action(leaf_node, prediction)
        else:
            result = infer.query(
                variables=self.network.get_roots(),
                evidence={leaf_node: np.argmax(prediction_error + prediction)})

            for root_node, root_cpd in result.items():
                before = self.network.get_cpds(root_node).values
                self.network.get_cpds(root_node).values = root_cpd.values
                logging.debug("node[%s] hypothesis-update from %s to %s", root_node, before, root_cpd.values)

    def model_update(self, node, prediction_error, prediction):
        """
        Updates the generative model by changing its structure (i.e. nodes, edges) or its parameters (i.e. CPDs)
        :param node: name of the node for which the prediction generated a prediction error high/precise enough to
        warrant a model update
        :param prediction_error: the difference between the prediction and the observation
        :param prediction: the prediction of the node - based on the hypothesis nodes in the model
        :return: the update model, hopefully one which generates predictions with less prediction error
        :type node: str
        :type prediction_error: np.array
        :type prediction: np.array
        :rtype BayesianModel
        """
        return self.network

    @staticmethod
    def get_root_nodes(network):
        """
        Returns status of all root nodes.
        :param network: Bayesian Network representing the generative model
        :return: Dictionary containing all root nodes as keys and status as values
        :type network: BayesianModel
        :rtype dict
        """
        roots = {}
        for root in network.get_roots():
            roots.update({root: np.argmax(network.get_cpds(root).values)})
        return roots

    @staticmethod
    def get_observations(model):
        obs = {}
        for leaf in model.get_leaves():
            obs.update({leaf: np.argmax(model.get_cpds(leaf).values)})
        return obs

    @staticmethod
    def get_two_dim(array):
        if len(array.shape) > 1:
            return array
        return array.reshape(array.shape[0], -1)

    @staticmethod
    def reshape_cpd(cpd, variable_card, evidence_card):
        if len(cpd.shape) == 1:
            return cpd.reshape([variable_card, 1])
        return cpd.reshape([variable_card, np.prod(evidence_card)])

    @staticmethod
    def expand_cpd(cpd, evidence_card):
        cpd = np.repeat(cpd, evidence_card, axis=1)
        for x in range(0, cpd.shape[1]):
            perturbation = random.uniform(-0.1, 0.1)
            cpd[0, x] = cpd[0, x] + perturbation  # TODO: Now it only works when variable has 2 values... fix this
            cpd[1, x] = cpd[1, x] - perturbation
        if len(cpd.shape) > 2:
            print('a')
        return cpd