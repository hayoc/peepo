import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.predictive_processing.v3.generative_model import GenerativeModel


def network():
    model = BayesianModel([('A', 'B'), ('A', 'C')])

    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.9, 0.1]])
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.9, 0.1],
                                                              [0.1, 0.9]], evidence=['A'], evidence_card=[2])
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.8, 0.2],
                                                              [0.2, 0.8]], evidence=['A'], evidence_card=[2])
    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.check_model()
    return model


network = network()
true_input = {'B': np.array([0.1, 0.9]),
              'C': np.array([0.8, 0.2])}

model = GenerativeModel(network)
print(model.model.get_cpds('A'))
for node, prediction in model.predict().items():
    pred = prediction.values
    obs = true_input[node]
    pes = model.error_size(pred, obs)

    if pes > 0:
        pe = model.error(pred, obs)

        print("Updating hypotheses")
        model.error_minimization(node=node, prediction_error_size=pes, prediction_error=pe, prediction=pred)
    else:
        print("Low prediction error")

print(model.model.get_cpds('A'))
