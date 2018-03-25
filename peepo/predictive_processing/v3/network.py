import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from scipy.stats import entropy


def default_network():
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

    model.get_parents()

    return model


def predict(model):
    """
    Predicts the leaf nodes (i.e. the observational nodes) based on the parent nodes (i.e. the hypothesis nodes)

    :param model: BayesianModel to use in prediction
    """
    infer = VariableElimination(model)

    result = []
    evidence = get_hypotheses(model)
    for leaf in model.get_leaves():
        result.append(infer.query([leaf], evidence=evidence)[leaf])
    return result


def prediction_error(pred, obs):
    return obs - pred


def prediction_error_size(pred, obs):
    return entropy(obs, pred)


def prediction_error_minimization(model, node, pes, pe, pred):
    # PEM 1: Hypothesis Update
    evidence = {node: pe + pred}
    for root in model.get_roots():


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
