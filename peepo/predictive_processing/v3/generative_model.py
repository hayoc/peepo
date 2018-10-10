import logging
import math

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import entropy

from peepo.visualize.graph import draw_network


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
        self.infer = VariableElimination(model)
        self.atomic_updates = [self.add_node, self.add_edge, self.change_parameters, self.change_valency]
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
        for node, prediction in self.predict(self.model).items():
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

    def predict(self, model):
        """
        Predicts the leaf nodes (i.e. the observational nodes) based on the parent nodes (i.e. the hypothesis nodes)
        The predicted values of the observations are the maximum a posteriori (MAP) distributions over the
        hypotheses

        :return: predictions for all observation variables

        :rtype: dict
        """
        return self.infer.query(variables=model.get_leaves(), evidence=self.get_hypotheses(model))

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
                                          evidence={node: np.argmax(prediction_error + prediction)})
                before = self.model.get_cpds(hypo).values
                self.model.get_cpds(hypo).values = result.get(hypo).values
                logging.debug("node[%s] hypothesis-update from %s to %s", hypo, before, result.get(hypo).values)
            # Should we update hypothesis variables based on only prediction error node?
            # Or all observation nodes in the network???

    def model_update(self, node, prediction_error, prediction):
        lowest_error_size = self.error_size(prediction, prediction_error + prediction)
        best_model = self.model

        for idx, val in enumerate(self.atomic_updates):
            updated_model = val(self.model.copy(), node, prediction, prediction_error + prediction)
            updated_prediction = self.predict(updated_model)[node].values
            updated_error_size = self.error_size(updated_prediction, prediction_error + prediction)
            if updated_error_size < lowest_error_size:
                lowest_error_size = updated_error_size
                best_model = updated_model

        self.model = best_model
        draw_network(self.model)

    def add_node(self, model, node_in_error, original_prediction, observation):
        if len(model) >= GenerativeModel.MAX_NODES:
            return model

        lowest_error = self.error_size(original_prediction, observation)
        best_model = model

        for active_node in model.active_trail_nodes(node_in_error)[node_in_error]:
            new_model = model.copy()
            new_node_name = str(len(model))
            new_model.add_node(new_node_name)
            new_model.add_edge(new_node_name, active_node)

            new_node_cpd = TabularCPD(variable=new_node_name, variable_card=2, values=[[0.1, 0.9]])
            old_cpd = new_model.get_cpds(active_node)

            evidence = old_cpd.get_evidence()
            evidence.append(new_node_name)
            evidence_card = list(old_cpd.get_cardinality(old_cpd.get_evidence()).values())
            evidence_card.append(2)
            values = np.append(old_cpd.values, [[0.5], [0.5]], axis=1)
            new_cpd_for_active_node = TabularCPD(variable=active_node,
                                                 variable_card=old_cpd.variable_card,
                                                 values=values,
                                                 evidence=evidence,
                                                 evidence_card=evidence_card)

            new_model.add_cpds(new_node_cpd, new_cpd_for_active_node)

            new_prediction = self.predict(new_model)[node_in_error].values
            new_error = self.error_size(new_prediction, observation)
            if new_error < lowest_error:
                lowest_error = new_error
                best_model = new_model

        return best_model

    def add_edge(self, model, node_in_error, original_prediction, observation):
        lowest_error = self.error_size(original_prediction, observation)
        best_model = model

        for node in model.get_nodes():
            if node == node_in_error:
                continue

            new_model = model.copy()
            new_model.add_edge(node, node_in_error)

            old_cpd = new_model.get_cpds(node_in_error)
            evidence = old_cpd.get_evidence()
            evidence.append(node)
            evidence_card = list(old_cpd.get_cardinality(old_cpd.get_evidence()).values())
            evidence_card.append(new_model.get_cpds(node).get_cardinality(node))
            values = np.append(old_cpd.values, [[0.5], [0.5]], axis=1)  # TODO: length should be based on cardinality
            new_cpd = TabularCPD(variable=node_in_error,
                                 variable_card=old_cpd.variable_card,
                                 values=values,
                                 evidence=evidence,
                                 evidence_card=evidence_card)

            new_model.add_cpds(new_cpd)

            new_prediction = self.predict(new_model)[node_in_error].values
            new_error = self.error_size(new_prediction, observation)
            if new_error < lowest_error:
                lowest_error = new_error
                best_model = new_model

        return best_model

    def change_parameters(self, model, node_in_error, original_prediction, observation):
        lowest_error = self.error_size(original_prediction, observation)
        best_model = model

        for active_node in model.active_trail_nodes('vision_1')['vision_1'] - set(model.get_roots()):
            vals = model.get_cpds(active_node).values

            for idx_col, col in enumerate(vals.T):
                for idx_row, row in enumerate(col):
                    new_model = model.copy()
                    new_vals = np.copy(vals)

                    to_add = abs(math.log(row, 10))
                    to_subtract = to_add / len(col)

                    new_vals[idx_row, idx_col] = new_vals[idx_row, idx_col] + to_add
                    for idx_row_copy, row_copy in enumerate(col):
                        if idx_row_copy is not idx_row:
                            new_vals[idx_row_copy, idx_col] = new_vals[idx_row_copy, idx_col] - to_subtract

                    new_model.get_cpds(active_node).values = new_vals

                    new_prediction = self.predict(new_model)[node_in_error].values
                    new_error = self.error_size(new_prediction, observation)
                    if new_error < lowest_error:
                        lowest_error = new_error
                        best_model = new_model

        return best_model

    def change_valency(self, model, node_in_error):
        return self.model

    @staticmethod
    def get_hypotheses(model):
        hypos = {}
        for root in model.get_roots():
            hypos.update({root: np.argmax(model.get_cpds(root).values)})
        return hypos

    @staticmethod
    def get_observations(model):
        obs = {}
        for leaf in model.get_leaves():
            obs.update({leaf: np.argmax(model.get_cpds(leaf).values)})
        return obs
