import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from scipy.stats import entropy


def default_model():
    """
    Method to create a default initialized Bayesian network.

    Returns
    -------
    parameters: BayesianModel
        Bayesian network with two dependent nodes (A & B) with 50-50 parameters
    """
    model = BayesianModel([('A', 'B')])

    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.5, 0.5]])
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.5, 0.5],
                                                              [0.5, 0.5]], evidence=['A'], evidence_card=[2])
    model.add_cpds(cpd_a, cpd_b)
    model.check_model()

    return model


def get_hypotheses(model):
    hypos = {}
    for root in model.get_roots():
        hypos.update({root: np.argmax(model.get_cpds(root).values)})
    return hypos


def get_observations(model):
    obs = {}
    for leaf in model.get_leaves():
        obs.update({leaf: np.argmax(model.get_cpds(leaf).values)})
    return obs


class GenerativeModel:

    def __init__(self, model=default_model()):
        self.model = model
        self.infer = VariableElimination(model)

    def predict(self):
        """
        Predicts the leaf nodes (i.e. the observational nodes) based on the parent nodes (i.e. the hypothesis nodes)
        """
        return self.infer.query(variables=self.model.get_leaves(), evidence=get_hypotheses(self.model))

    def error(self, pred, obs):
        return obs - pred

    def error_size(self, pred, obs):
        return entropy(obs, pred)

    def error_minimization(self, node, prediction_error_size, prediction_error, prediction):
        # PEM 1: Hypothesis Update
        self.hypothesis_update(node, prediction_error, prediction)

    def hypothesis_update(self, node, prediction_error, prediction):
        for hypo in self.model.get_roots():
            result = self.infer.query(variables=[hypo],
                                      evidence={node: np.argmax(
                                          prediction_error if prediction is None else prediction_error + prediction)})
            self.model.get_cpds(hypo).values = result.get(hypo).values
        # Should we update hypothesis variables based on only prediction error node?
        # Or all observation nodes in the network???
