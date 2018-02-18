import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel


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
    model.get_leaves()
    model.add_cpds(cpd_a, cpd_b)
    model.check_model()
    return model


def predict(model):
    """
    Predicts the leaf nodes (i.e. the observational nodes) based on the parent nodes (i.e. the hypothesis nodes)

    Returns
    -------
    parameters: BayesianModel
        Bayesian network in tree form
    """
    infer = VariableElimination(model)

    result = []
    for leaf in model.get_leaves():
        result.append(infer.query([leaf], evidence=get_evidence(model))[leaf])
    return result


def get_evidence(model):
    evidence = {}
    for root in model.get_roots():
        evidence.update({root: np.argmax(model.get_cpds(root).values)})
    return evidence
