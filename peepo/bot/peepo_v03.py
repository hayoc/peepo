import logging

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.bot.peepo_bot import PeepoBot
from peepo.bot.sensory_input_bot import SensoryInput
from peepo.predictive_processing.v3.generative_model import GenerativeModel

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

logging.info('logging initialized')
bot = PeepoBot()
logging.info('bot initialized')

network = BayesianModel([('hypo', 'infrared'), ('hypo', 'motor')])

cpd_a = TabularCPD(variable='hypo', variable_card=2, values=[[0.7, 0.3]])
cpd_b = TabularCPD(variable='infrared', variable_card=2, values=[[0.9, 0.1],
                                                                 [0.1, 0.9]], evidence=['hypo'], evidence_card=[2])
cpd_c = TabularCPD(variable='motor', variable_card=2, values=[[0.9, 0.1],
                                                              [0.1, 0.9]], evidence=['hypo'], evidence_card=[2])
network.add_cpds(cpd_a, cpd_b, cpd_c)
network.check_model()

model = GenerativeModel(SensoryInput(bot), network)

logging.info('starting predictive processing')

while True:
    model.process()
