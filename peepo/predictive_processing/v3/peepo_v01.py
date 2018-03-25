import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

from peepo.predictive_processing.v3.network import predict, prediction_error_size, prediction_error, \
    prediction_error_minimization


def network():
    model = BayesianModel([('A', 'B'), ('A', 'C')])

    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.9, 0.1]])
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.9, 0.1],
                                                              [0.1, 0.9]], evidence=['A'], evidence_card=[2])
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9, 0.1],
                                                              [0.1, 0.9]], evidence=['A'], evidence_card=[2])
    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.check_model()
    return model


model = network()
true_input = {'B': np.array([0.9, 0.1]),
              'C': np.array([0.9, 0.1])}

predictions = predict(model)
for distrib in predictions:

    node = distrib.variables[0]
    pred = distrib.values
    obs = true_input[node]
    pes = prediction_error_size(pred, obs)

    if pes > 0:
        pe = prediction_error(pred, obs)

        print("Updating hypotheses")
        model = prediction_error_minimization(model=model, node=node, pes=pes, pe=pe, pred=pred)
    else:
        print("Low prediction error")
