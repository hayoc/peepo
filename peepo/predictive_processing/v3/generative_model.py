import logging

import numpy as np
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution
from scipy.stats import entropy


class GenerativeModel:
    """
    Predictive Processing Generative Model

    This is a generative model, implemented as a Bayesian Causal Network. The three functions
    prediction, prediction error and prediction error minimization are defined for this model.

    :param bayesian_network : Bayesian Causal Network. Causes are hypothesis variables, effects are observational variables.
    :param sensory_input : Mutable dictionary containing the current sensory inputs
    :param n_jobs : Number of process to spawn for multiprocessing. By default 1 = no additional processes spawned

    :type bayesian_network : [BayesianModel]
    :type sensory_input : SensoryInput
    :type n_jobs : int

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

    def __init__(self, bayesian_network, sensory_input, n_jobs=1):
        self.bayesian_network = bayesian_network.copy()
        self.sensory_input = sensory_input
        self.n_jobs = n_jobs

        # draw_network(bayesian_network)

    def process(self):
        """
        Processes one flow in the predictive processing algorithm:
            1) prediction
            2) prediction error
            3) prediction error minimization (hypothesis or model update)
        Returns the total prediction error size observed (for informational purposes...)
        """
        total_prediction_error_size = 0

        for index, node in enumerate(self.predict()):
            node_name = self.bayesian_network.states[index].name
            if self.LEN in node_name:
                prediction = np.array([x[1] for x in sorted(node.items(), key=lambda tup: tup[0])])
                observation = self.sensory_input.value(node_name)
                prediction_error = self.error(prediction, observation)
                prediction_error_size = self.error_size(prediction, observation)
                precision = self.precision(prediction)
                total_prediction_error_size += prediction_error_size

                # Sometimes numpy entropy calculation returns extremely small numbers when there's no error
                if prediction_error_size > 0.1:
                    logging.debug(
                        "node[%s] prediction-error ||| predicted %s -vs- %s observed ||| PES %s ||| PRECISION %s",
                        node_name, prediction, observation, prediction_error_size, precision)
                    self.error_minimization(node_name=node_name,
                                            precision=precision,
                                            prediction_error=prediction_error,
                                            prediction=prediction)

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

        :param node_name: name of the node causing the prediction error
        :param precision: precision of the prediction
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error

        :type node_name : str
        :type precision: float
        :type prediction_error: np.array
        :type prediction: np.array
        """
        self.hypothesis_update(node_name, prediction_error, prediction)

    def hypothesis_update(self, node_name, prediction_error, prediction):
        """
        Updates the hypotheses of the generative model to minimize prediction error

        :param node_name: name of the node causing the prediction error
        :param prediction_error: the prediction error itself
        :param prediction: prediction causing the prediction error

        :type node_name : str
        :type prediction_error: np.array
        :type prediction: np.array
        """
        if "motor" in node_name:
            self.sensory_input.action(node_name, prediction)
        else:
            evidence = {node_name: str(np.argmax(prediction_error + prediction))}
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
