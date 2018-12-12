from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.utilities.bayesian_network import bayesian_network_to_json

VISION1 = 'vision_1'
VISION2 = 'vision_2'
VISION3 = 'vision_3'
VISION4 = 'vision_4'
VISION5 = 'vision_5'
VISION6 = 'vision_6'
MOTORLEFT = 'motor_left'
MOTORRIGHT = 'motor_right'


def generate_blank_bn():
    network = BayesianModel()

    network.add_nodes_from([VISION1, VISION2, VISION3, VISION4, VISION5, VISION6, MOTORLEFT, MOTORRIGHT])

    cpd1 = TabularCPD(variable=VISION1, variable_card=2, values=[[0.5, 0.5]])
    cpd2 = TabularCPD(variable=VISION2, variable_card=2, values=[[0.5, 0.5]])
    cpd3 = TabularCPD(variable=VISION3, variable_card=2, values=[[0.5, 0.5]])
    cpd4 = TabularCPD(variable=VISION4, variable_card=2, values=[[0.5, 0.5]])
    cpd5 = TabularCPD(variable=VISION5, variable_card=2, values=[[0.5, 0.5]])
    cpd6 = TabularCPD(variable=VISION6, variable_card=2, values=[[0.5, 0.5]])
    cpd7 = TabularCPD(variable=MOTORLEFT, variable_card=2, values=[[0.5, 0.5]])
    cpd8 = TabularCPD(variable=MOTORRIGHT, variable_card=2, values=[[0.5, 0.5]])

    network.add_cpds(cpd1, cpd2, cpd3, cpd4, cpd5, cpd6, cpd7, cpd8)
    network.check_model()

    return network


bayesian_network_to_json(generate_blank_bn(), 'gen0', 'id0')
