import logging

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.predictive_processing.v3.generative_model import GenerativeModel

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def network():
    model = BayesianModel([('hypo', 'infrared'), ('hypo', 'infrared-supporting'), ('hypo', 'motor')])

    cpd_a = TabularCPD(variable='hypo', variable_card=2, values=[[0.9, 0.1]])
    cpd_b = TabularCPD(variable='infrared', variable_card=2, values=[[0.9, 0.1],
                                                                     [0.1, 0.9]], evidence=['hypo'], evidence_card=[2])
    cpd_d = TabularCPD(variable='infrared-supporting', variable_card=2, values=[[0.9, 0.1],
                                                                                [0.1, 0.9]], evidence=['hypo'],
                       evidence_card=[2])
    cpd_c = TabularCPD(variable='motor', variable_card=2, values=[[0.6, 0.2],
                                                                  [0.4, 0.8]], evidence=['hypo'], evidence_card=[2])
    model.add_cpds(cpd_a, cpd_b, cpd_d, cpd_c)
    model.check_model()
    return model


network = network()
true_input = {'infrared': np.array([0.1, 0.9]),
              'infrared-supporting': np.array([0.1, 0.9]),
              'motor': np.array([0.6, 0.4])}

model = GenerativeModel(true_input, network)
model.process()
