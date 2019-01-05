import logging
<<<<<<< HEAD

import numpy as np
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution
from scipy.stats import entropy

=======
import random

import numpy as np
from pgmpy.inference import VariableElimination
from scipy.stats import entropy



>>>>>>> origin/model_update_bernard

class GenerativeModel:
    """
    Predictive Processing Generative Model
    This is a generative model, implemented as a Bayesian Causal Network. The three functions
    prediction, prediction error and prediction error minimization are defined for this model.
<<<<<<< HEAD

    :param bayesian_network : Bayesian Causal Network. Causes are hypothesis variables, effects are observational variables.
    :param sensory_input : Mutable dictionary containing the current sensory inputs
    :param n_jobs : Number of process to spawn for multiprocessing. By default 1 = no additional processes spawned

    :type bayesian_network : [BayesianModel]
    :type sensory_input : SensoryInput
    :type n_jobs : int

=======
    :param sensory_input : Mutable dictionary containing the current sensory inputs
    :param network : Bayesian Causal Network. Causes are hypothesis variables, effects are observational variables.
    :type sensory_input : SensoryInput
    :type network : [BayesianModel]
>>>>>>> origin/model_update_bernard
    TODO: Model Update, e.g. through: self.atomic_updates = [self.add_node, self.add_edge, self.change_parameters]
    TODO: Integrate PRECISION BASED WEIGHTING on prediction errors. E.g. prediction error minimization should only
    happen if the prediction errors have enough weight assigned to them. This can depend on context, the organism's
    goal, or other ways.
    TODO: Implement a custom BayesianNetwork so we can distinguish between action and perception nodes. Instead of
    distinguishing them by checking for 'motor' in the name.
    TODO: Parallelism
    """

    RON = 'RON'
    BEN = 'BEN'
    MEN = 'MEN'
    LAN = 'LAN'
    LEN = 'LEN'

    def __init__(self, peepo_network, sensory_input, n_jobs=1):
        self.bayesian_network = peepo_network.pomegranate_network
        self.sensory_input = sensory_input
<<<<<<< HEAD
        self.n_jobs = n_jobs

        # draw_network(bayesian_network)
=======
        self.network = network.copy()

>>>>>>> origin/model_update_bernard

    def process(self, structure_learning=False):
        """
        Processes one flow in the predictive processing algorithm:
            1) prediction
            2) prediction error
            3) prediction error minimization (hypothesis or model update)
        Returns the total prediction error size observed (for informational purposes...)

        If structure_learning is True, only inference will be done. Hypothesis updates will not happen. This should
        be used for learning the structure of a module, by manually setting the hypothesis and comparing errors of
        different toplogies.
        """
        total_prediction_error_size = 0

<<<<<<< HEAD
        for index, node in enumerate(self.predict()):
            node_name = self.bayesian_network.states[index].name
            if self.is_leaf(index):
                prediction = np.array([x[1] for x in sorted(node.items(), key=lambda tup: tup[0])])
                observation = self.sensory_input.value(node_name)
                prediction_error = self.error(prediction, observation)
                prediction_error_size = self.error_size(prediction, observation)
                precision = self.precision(prediction)
                total_prediction_error_size += prediction_error_size

                # Sometimes numpy entropy calculation returns extremely small numbers when there's no error
                if prediction_error_size > 0.1 and not structure_learning:
                    logging.debug(
                        "node[%s] prediction-error ||| predicted %s -vs- %s observed ||| PES %s ||| PRECISION %s",
                        node_name, prediction, observation, prediction_error_size, precision)
                    self.error_minimization(node_name=node_name,
                                            precision=precision,
                                            prediction_error=prediction_error,
                                            prediction=prediction)
        return total_prediction_error_size

    def predict(self):
        """
        Predicts the leaf nodes (i.e. the observational nodes) based on the root nodes (i.e. the belief nodes)

        :return: prediction for all leaf nodes, a prediction is a probability distribution

        :rtype: list of Distributions

        #TODO: A fundamental problem with PP?: Cannot do prediction>error minimization with one loop per node,
        #TODO: since a sister LEN node which does not yet have the correct input will revert the hypothesis update.
        """
        evidence = self.get_root_values()
        return self.bayesian_network.predict_proba(evidence)
=======
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
>>>>>>> origin/model_update_bernard

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

    @staticmethod
    def precision(pred):
        """
        Calculates the precision of the prediction, indicating the certainty of the prediction.
        Usually calculated as the negative log likelihood (TODO)

        :param pred: Prediction to calculate the precision for
        :return: precision of the prediction

        :type: pred: np.array
        :rtype: float
        """
        return entropy(pred, base=2)

    def error_minimization(self, node_name, precision, prediction_error, prediction):
        """
        Attempts to minimize the prediction error by one of the possible PEM methods:
            1) Hypothesis Update
            2) Model Update
<<<<<<< HEAD

        :param node_name: name of the node causing the prediction error
        :param precision: precision of the prediction
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error

        :type node_name : str
=======
        :param node: name of the node causing the prediction error
        :param precision: precision of the prediction
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error
        :type node : str
>>>>>>> origin/model_update_bernard
        :type precision: float
        :type prediction_error: np.array
        :type prediction: np.array
        """
        self.hypothesis_update(node_name, prediction_error, prediction)

    def hypothesis_update(self, node_name, prediction_error, prediction):
        """
        Updates the hypotheses of the generative model to minimize prediction error
<<<<<<< HEAD

        :param node_name: name of the node causing the prediction error
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error

        :type node_name : str
=======
        :param leaf_node: name of the node causing the prediction error
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error
        :type leaf_node : str
>>>>>>> origin/model_update_bernard
        :type prediction_error: np.array
        :type prediction: np.array
        """
        if "motor" in node_name:
            self.sensory_input.action(node_name, prediction)
        else:
<<<<<<< HEAD
            evidence = {node_name: np.argmax(prediction_error + prediction)}
            result = self.bayesian_network.predict_proba(evidence)

            for root in self.get_roots():
                root_index = self.get_node_index(root.name)

                old_hypo = self.bayesian_network.states[root_index].distribution.items()
                new_hypo = result[root_index].items()

                self.bayesian_network.states[root_index].distribution = DiscreteDistribution(dict(new_hypo))
                logging.debug("node[%s] hypothesis-update from %s to %s", root.name, old_hypo, new_hypo)

    def get_root_values(self):
        return {x.name: x.distribution.mle() for x in self.get_roots()}

    def get_roots(self):
        return [x for x in self.bayesian_network.states if self.RON in x.name]

    def get_leaves(self):
        return [x for x in self.bayesian_network.states if self.LEN in x.name]

    def get_node_index(self, node_name):
        for x, state in enumerate(self.bayesian_network.states):
            if state.name == node_name:
                return x
        raise ValueError('Node %s does not exist in network.', node_name)

    def is_leaf(self, index):
        return not any(index in node_parents for node_parents in self.bayesian_network.structure)

    def is_root(self, index):
        return any(index in node_parents for node_parents in self.bayesian_network.structure)
=======
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
>>>>>>> origin/model_update_bernard
