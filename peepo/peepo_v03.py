import logging

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.bot.peepo_bot import PeepoBot
from peepo.predictive_processing.v3.generative_model import GenerativeModel

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

network = BayesianModel([('hypo', 'infrared'), ('hypo', 'motor')])

cpd_a = TabularCPD(variable='hypo', variable_card=2, values=[[0.7, 0.3]])
cpd_b = TabularCPD(variable='infrared', variable_card=2, values=[[0.9, 0.1],
                                                                 [0.1, 0.9]], evidence=['hypo'], evidence_card=[2])
cpd_c = TabularCPD(variable='motor', variable_card=2, values=[[0.6, 0.2],
                                                              [0.4, 0.8]], evidence=['hypo'], evidence_card=[2])
network.add_cpds(cpd_a, cpd_b, cpd_c)
network.check_model()

bot = PeepoBot()

sensory_input = {'infrared': np.array([0.1, 0.9] if bot.vision() > 60 else np.array([0.9, 0.1])),
                 'motor': np.array([0.1, 0.9]) if bot.is_driving_backward() else np.array([0.9, 0.1])}

model = GenerativeModel(sensory_input, network)

while True:
    model.process()
    sensory_input['infrared'] = np.array([0.1, 0.9] if bot.vision() > 60 else np.array([0.9, 0.1]))
    sensory_input['motor'] = np.array([0.1, 0.9]) if bot.is_driving_backward() else np.array([0.9, 0.1])
