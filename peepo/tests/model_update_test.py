import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput
from peepo.visualize.graph import draw_network


def create_network():
    network = BayesianModel([('A', 'B')])

    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.1, 0.9]])
    cpd_b = TabularCPD(variable='B', variable_card=2,
                       values=[[0.1, 0.9],
                               [0.9, 0.1]],
                       evidence=['A'],
                       evidence_card=[2])
    network.add_cpds(cpd_a, cpd_b)
    network.check_model()

    return GenerativeModel(TestSensoryInput(), network)


class TestSensoryInput(SensoryInput):

    def action(self, node, prediction_error, prediction):
        pass

    def value(self, name):
        return np.array([0.1, 0.9])


gen_model = create_network()

prediction = gen_model.predict(gen_model.model)['B'].values
observation = gen_model.sensory_input.value('B')
prediction_error = gen_model.error(prediction, observation)

new_model = gen_model.add_node(gen_model.model, 'B', prediction, observation)
draw_network(new_model, True)
