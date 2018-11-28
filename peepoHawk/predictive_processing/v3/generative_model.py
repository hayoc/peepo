#28/11
import logging
import math
import random

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from scipy.stats import entropy

from peepoHawk.visualize.graph import draw_network
from peepoHawk.playground.models.CeePeeDees import CPD

class GenerativeModel:
    """
    Predictive Processing Generative Model

    This is a generative model, implemented as a Bayesian Causal Network. The three functions
    prediction, prediction error and prediction error minimization are defined for this model.

    :param sensory_input : Mutable dictionary containing the current sensory inputs
    :param network : Bayesian Causal Network. Causes are hypothesis variables, effects are observational variables.

    :type sensory_input : SensoryInput
    :type network : [BayesianModel]
    """
#TO DO ?: make the code general i.e. not sepcific for actual Raptor-model
    MAX_NODES = 10

    def __init__(self, obj, parentnodes, relevant_parent_nodes):
        self.network = obj.network
        self.raptor = obj
        self.parent_nodes = parentnodes
        self.observations = obj.observations
        self.relevant_parent_nodes  =  relevant_parent_nodes
        self.new_parents_index = np.zeros(np.size(self.relevant_parent_nodes))
        self.atomic_updates = [self.add_node, self.add_edge, self.change_parameters]  # TODO: Add change_valency
        #draw_network(network)

    def process(self):
        """
        Processes one flow in the predictive processing algorithm:
            1) prediction
            2) prediction error
            3) prediction error minimization
        Returns the total prediction error size observed (for informational purposes...)
        """
        total_pes = 0
        pes = 0
        coordinates = self.get_optimal_hypothesis("brute_force")
        '''for node, prediction in self.predict(self.network).items():
            pred = prediction.values
            obs = self.sensory_input.value(node)
            pes = self.error_size(pred, obs)

            # TODO: PEM should only happen if PES is higher than some value, this value
            # TODO: should depend on whatever context the agent finds itself in, and the agent's goal
            precision = entropy(pred, base=2)
            pe = self.error(pred, obs)
            total_pes += pes'''

        if pes > 0.1:  # Sometimes numpy entropy calculation returns extremely small numbers when there's no error
            logging.debug("node[%s] prediction-error ||| predicted %s -vs- %s observed ||| PES %s ||| PRECISION %s",
                            node, pred, obs, pes, precision)
            self.error_minimization(node=node, precision=precision, prediction_error=pe, prediction=pred)

        return self.best_parents(coordinates)

    def best_parents(self,coordinates):
        print("Coordinates : ", coordinates)
        corr = CPD.create_fixed_parent(3, int(coordinates[0]))
        dir = CPD.create_fixed_parent(3, int(coordinates[1]))
        return corr,dir

    def get_optimal_hypothesis(self, method):
        '''This method returns an array with the best relevant parent nodes
        i.e. returns e.g an array [k,l,...,m] for which :
        p(k).parentnode[selectedparentnode[0]] = max
        p(l)..parentnode[selectedparentnode[1]] = max
        .
        .
        p(m)..parentnode[selectedparentnode[n-1]] = max
        with n = number of parent nodes passed.
        The ad-hoc parentnodes will be constructed in the method
                    :predict()'''
        coordinates = np.zeros(np.size(self.relevant_parent_nodes))
        err_0 = float("inf")
        err = 0
        if method == "brute_force":
            deltas = np.zeros(np.size(self.observations))
            cardinality = []
            for i in range(0,np.size(self.relevant_parent_nodes) ):
                cardinality.append(self.network.get_cardinality(self.parent_nodes[self.relevant_parent_nodes[i]]))
            M = CPD.get_index_matrix(cardinality)
            '''print("cardinality")
            print(cardinality)
            print("M")
            print(M)'''
            #combination = np.zeros(np.size(M,1))
            for column  in range(0, np.size(M,1)):
                combination = M[:,column]
                pr = self.predict(combination)
                #print("Prediction for column " , column , ": ", pr)
                #print("Size pr = ", np.size(pr,0), " and observations : ",np.size(self.observations,0) )
                for row in range(0, np.size(pr,0)):
                    #print("pr[",row,"] = ", pr[row],  " and observation = ", self.observations[row])
                    delta  = self.delta(pr[row],self.observations[row], 'dotproduct')
                    deltas[row] = delta
                err = self.epsilon(deltas)
                #print("Calculated error :", err)
                if err <= err_0:
                    err_0 = err
                    coordinates = combination
                    #print("retained combination : ", coordinates)
        return coordinates

    def predict(self, combination):
        evidence = {}
        for i in range(0, np.size(combination)):
            card = self.network.get_cardinality(self.parent_nodes[self.relevant_parent_nodes[i]])
            #print(self.parent_nodes[self.relevant_parent_nodes[i]], " card = " ,card, " combination = ", combination[i])
            self.network.get_cpds(self.parent_nodes[self.relevant_parent_nodes[i]]).values = CPD.create_fixed_parent(card, int(combination[i]))
            evidence.update({self.parent_nodes[self.relevant_parent_nodes[i]]: int(combination[i])})
        infer = VariableElimination(self.network)
        variables = self.network.get_leaves()
        '''print("Variables :", variables)
        print("Evidence : ", evidence)'''
        quer = infer.query(variables=variables, evidence=evidence)
        predictions = []
        for node, prediction in quer.items():
            predictions.append(np.asarray(prediction.values))
        #print("Predictions : "  , predictions)
        return predictions

    def delta(self, predicted, observation, method):
        if method == 'dotproduct':
            return 1 -  np.dot(predicted, observation)
        if method == 'Kullback':
            return 0
        return 0

    def epsilon(self, deltas):
        return np.sum(deltas)

    @staticmethod
    def get_hypotheses(model):
        hypos = {}
        for root in model.get_roots():
            hypos.update({root: np.argmax(model.get_cpds(root).values)})
        return hypos



    '''def predict(self, model):
        """
        Predicts the leaf nodes (i.e. the observational nodes) based on the parent nodes (i.e. the hypothesis nodes)
        The predicted values of the observations are the maximum a posteriori (MAP) distributions over the
        hypotheses

        :return: predictions for all observation variables

        :rtype: dict
        """
        infer = VariableElimination(model)
        variables = model.get_leaves()
        evidence = self.get_hypotheses(model)
        evidence = {k: v for k, v in evidence.items() if k not in variables}
        print("Variables :", variables)
        print("Evidence : " , evidence)

        return infer.query(variables=variables, evidence=evidence)'''

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
        if node in self.network.get_roots() or precision > 0.75:  # TODO: plus add precision
            self.model_update(node, prediction_error, prediction)
        else:
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
        infer = VariableElimination(self.network)
        if "motor" in node:
            self.sensory_input.action(node, prediction_error, prediction)
        else:
            for hypo in self.network.get_roots():
                result = infer.query(
                    variables=[hypo],
                    evidence={node: np.argmax(prediction_error + prediction)}).get(hypo).values

                before = self.network.get_cpds(hypo).values
                self.network.get_cpds(hypo).values = result
                logging.debug("node[%s] hypothesis-update from %s to %s", hypo, before, result)
            # Should we update hypothesis variables based on only prediction error node?
            # Or all observation nodes in the network???

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
        best_model = self.network
        best_update = 'none'

        for idx, val in enumerate(self.atomic_updates):
            updated_model = val(self.network.copy(), node, prediction, prediction_error + prediction)
            updated_prediction = self.predict(updated_model)[node].values
            updated_error_size = self.error_size(updated_prediction, prediction_error + prediction)
            if updated_error_size < lowest_error_size:
                logging.info('Better update from: ' + val.__name__)
                lowest_error_size = updated_error_size
                best_model = updated_model
                best_update = val.__name__

        self.network = best_model
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
            vals = self.reshape_cpd(cpd.values, cpd.variable_card, list(cpd.get_cardinality(cpd.get_evidence()).values()))

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
