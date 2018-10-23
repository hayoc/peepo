import logging

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


def create_network_for_add_edge():
    """ Note: adding an edge where the values are duplicated from the existing connecting nodes won't change
        predictions at all. Since it will just take on the value of the pre-existing state, given that we duplicate
        the values... Maybe go back to 0.5 0.5 defaults?
    """
    network = BayesianModel([('A', 'B'), ('C', 'D')])

    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.4, 0.6]])
    cpd_b = TabularCPD(variable='B', variable_card=2,
                       values=[[0.1, 0.9],
                               [0.9, 0.1]],
                       evidence=['A'],
                       evidence_card=[2])
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.6, 0.4]])
    cpd_d = TabularCPD(variable='D', variable_card=2,
                       values=[[0.8, 0.9],
                               [0.2, 0.1]],
                       evidence=['C'],
                       evidence_card=[2])
    network.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)
    network.check_model()

    return GenerativeModel(TestSensoryInput(), network)


class TestSensoryInput(SensoryInput):

    def action(self, node, prediction_error, prediction):
        pass

    def value(self, name):
        return np.array([0.1, 0.9])


def test_add_node():
    gen_model = create_network()

    prediction = gen_model.predict(gen_model.model)['B'].values
    observation = gen_model.sensory_input.value('B')
    prediction_error = gen_model.error(prediction, observation)

    # time.sleep(20)
    logging.warning('Adding node now...')

    new_model = gen_model.add_node(gen_model.model, 'B', prediction, observation)
    draw_network(new_model, True)


def test_add_edge():
    gen_model = create_network_for_add_edge()

    prediction = gen_model.predict(gen_model.model)['B'].values
    observation = gen_model.sensory_input.value('B')
    prediction_error = gen_model.error(prediction, observation)

    # time.sleep(20)
    logging.warning('Adding edge now...')

    new_model = gen_model.add_edge(gen_model.model, 'B', prediction, observation)
    draw_network(new_model, True)


# TODO: Current implementation does not technically work - since any new model will have same prediction error size
test_add_edge()
