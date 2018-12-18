import logging
import math
import random

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import entropy

from peepo.utilities.bayesian_network import filter_non_observed_nodes
from peepo.visualize.graph import draw_network


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
        self.network = network
        draw_network(network)

    def process(self):
        """
        Processes one flow in the predictive processing algorithm:
            1) prediction
            2) prediction error
            3) prediction error minimization (hypothesis or model update)
        Returns the total prediction error size observed (for informational purposes...)
        """
        total_prediction_error_size = 0
        for node, pred in self.predict(self.network).items():
            prediction = pred.values
            observation = self.sensory_input.value(node)
            prediction_error_size = self.error_size(prediction, observation)
            prediction_error = self.error(prediction, observation)
            precision = entropy(prediction, base=2)
            total_prediction_error_size += prediction_error_size

            # Sometimes numpy entropy calculation returns extremely small numbers when there's no error
            if prediction_error_size > 0.1:
                logging.debug("node[%s] prediction-error ||| predicted %s -vs- %s observed ||| PES %s ||| PRECISION %s",
                              node, prediction, observation, prediction_error_size, precision)
                self.error_minimization(node=node,
                                        precision=precision,
                                        prediction_error=prediction_error,
                                        prediction=prediction)

        return total_prediction_error_size

    def predict(self, network):
        """
        Predicts the leaf nodes (i.e. the observational nodes) based on the root nodes (i.e. the belief nodes)

        :return: predictions for all observation variables, a prediction is a probability distribution

        :rtype: dict
        """
        infer = VariableElimination(network)
        variables = self.get_observed_nodes(network)
        # variables = network.get_leaves()
        evidence = self.get_root_nodes(network)
        evidence = {k: v for k, v in evidence.items() if k not in variables}

        return infer.query(variables=variables, evidence=evidence)

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
                variables=[x for x in self.network.get_roots()
                           if 'obs' not in x and 'motor' not in x],
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
        lowest_error_size = self.error_size(prediction, prediction_error + prediction)
        best_network = self.network
        best_update = 'none'

        for idx, val in enumerate(self.atomic_updates):
            updated_model = val(self.network.copy(), node, prediction, prediction_error + prediction)
            updated_prediction = self.predict(updated_model)[node].values
            updated_error_size = self.error_size(updated_prediction, prediction_error + prediction)
            if updated_error_size < lowest_error_size:
                logging.info('Better update from: ' + val.__name__)
                lowest_error_size = updated_error_size
                best_network = updated_model
                best_update = val.__name__

        self.network = best_network
        logging.info('Best Update: ' + best_update)
        draw_network(self.network)
        return self.network

    def add_node(self, model, node_in_error, original_prediction, observation):
        """
        Updates the generative model by adding a new node, connected to the node which caused the prediction error

        :param model: the generative model to be updated
        :param node_in_error: name of the node which cause the prediction error
        :param original_prediction: the prediction before the model update
        :param observation: the observation which didn't match the prediction
        :return: the updated model

        :type model: BayesianModel
        :type node_in_error: str
        :type original_prediction: np.array
        :type observation: np.array
        :rtype BayesianModel
        """
        if len(model) >= GenerativeModel.MAX_NODES:
            return model

        lowest_error = self.error_size(original_prediction, observation)
        best_model = model

        for active_node in model.active_trail_nodes(node_in_error)[node_in_error]:
            new_model = model.copy()
            new_node_name = str(len(model))
            new_model.add_node(new_node_name)
            new_model.add_edge(new_node_name, active_node)
            new_node_cpd = TabularCPD(variable=new_node_name, variable_card=2, values=[[0.5, 0.5]])
            new_model.add_cpds(new_node_cpd)

            old_cpd = model.get_cpds(active_node)
            variable_card = old_cpd.variable_card
            evidence = old_cpd.get_evidence()
            evidence.append(new_node_name)
            evidence_card = list(old_cpd.get_cardinality(old_cpd.get_evidence()).values())
            old_evidence_card = list(evidence_card)
            evidence_card.append(2)

            values = self.expand_cpd(self.reshape_cpd(old_cpd.values, variable_card, old_evidence_card), 2)

            new_cpd_for_active_node = TabularCPD(variable=active_node,
                                                 variable_card=variable_card,
                                                 values=values,
                                                 evidence=evidence,
                                                 evidence_card=evidence_card)

            new_model.add_cpds(new_cpd_for_active_node)

            new_prediction = self.predict(new_model)[node_in_error].values
            new_error = self.error_size(new_prediction, observation)
            if new_error < lowest_error:
                lowest_error = new_error
                best_model = new_model

        return best_model

    def add_edge(self, model, node_in_error, original_prediction, observation):
        """
        Updates the generative model by adding a new edge, connecting the node which caused the prediction error to
        a random node

        :param model: the generative model to be updated
        :param node_in_error: name of the node which cause the prediction error
        :param original_prediction: the prediction before the model update
        :param observation: the observation which didn't match the prediction
        :return: the updated model

        :type model: BayesianModel
        :type node_in_error: str
        :type original_prediction: np.array
        :type observation: np.array
        :rtype BayesianModel
        """
        lowest_error = self.error_size(original_prediction, observation)
        best_model = model

        for node in model.nodes():
            if node == node_in_error or (node, node_in_error) in model.edges() or 'obs' in node or 'motor' in node:
                continue

            new_model = model.copy()
            new_model.add_edge(node, node_in_error)

            old_cpd = new_model.get_cpds(node_in_error)
            variable_card = old_cpd.variable_card
            evidence = old_cpd.get_evidence()
            evidence.append(node)
            evidence_card = list(old_cpd.get_cardinality(old_cpd.get_evidence()).values())
            old_evidence_card = list(evidence_card)
            evidence_card.append(new_model.get_cpds(node).variable_card)

            values = self.expand_cpd(self.reshape_cpd(old_cpd.values, variable_card, old_evidence_card), 2)

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
        """
        Updates the generative model by changing the parameters of the node which caused the prediction error

        :param model: the generative model to be updated
        :param node_in_error: name of the node which cause the prediction error
        :param original_prediction: the prediction before the model update
        :param observation: the observation which didn't match the prediction
        :return: the updated model

        :type model: BayesianModel
        :type node_in_error: str
        :type original_prediction: np.array
        :type observation: np.array
        :rtype BayesianModel
        """
        lowest_error = self.error_size(original_prediction, observation)
        best_model = model

        for active_node in model.active_trail_nodes(node_in_error)[node_in_error] - set(model.get_roots()):
            cpd = model.get_cpds(active_node)
            vals = self.reshape_cpd(cpd.values, cpd.variable_card,
                                    list(cpd.get_cardinality(cpd.get_evidence()).values()))

            for idx_col, col in enumerate(vals.T):
                for idx_row, row in enumerate(col):
                    new_model = model.copy()
                    new_vals = np.copy(vals)

                    to_add = abs(math.log(row, 100))
                    to_subtract = to_add / (len(col) - 1)

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

    def change_valency(self, model, node_in_error, original_prediction, observation):
        """
        Updates the generative model by changing the valency (i.e. the amount of parameters in the CPD) of the
        node which caused the prediction error

        :param model: the generative model to be updated
        :param node_in_error: name of the node which cause the prediction error
        :param original_prediction: the prediction before the model update
        :param observation: the observation which didn't match the prediction
        :return: the updated model

        :type model: BayesianModel
        :type node_in_error: str
        :type original_prediction: np.array
        :type observation: np.array
        :rtype BayesianModel
        """
        # TODO
        return self.network

    @staticmethod
    def get_observed_nodes(network):
        """
        Returns names of all observed nodes, i.e. nodes containing motor, vision, obs, etc.

        :param network: BayesianModel
        :return: list of observed nodes
        """
        return filter_non_observed_nodes(network.get_leaves())

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
            print()
        return cpd
