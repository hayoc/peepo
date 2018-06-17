import logging
import random

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.playground.sensory_input_coin import SensoryInputCoin
from peepo.predictive_processing.v3.generative_model import GenerativeModel

import matplotlib.pyplot as plt


def paired_set(size):
    return [1 if i % 2 == 0 else 0 for i in range(size)]


def sorted_set(size):
    return [1 if i < size / 2 else 0 for i in range(size)]


def random_set(size):
    return [random.choice([0, 1]) for _ in range(size)]


def process(coin_set, ax, title):
    network = BayesianModel([('hypo', 'coin')])

    cpd_a = TabularCPD(variable='hypo', variable_card=2, values=[[0.5, 0.5]])
    cpd_b = TabularCPD(variable='coin', variable_card=2, values=[[0.99, 0.01],
                                                                 [0.01, 0.99]], evidence=['hypo'],
                       evidence_card=[2])
    network.add_cpds(cpd_a, cpd_b)
    network.check_model()

    model = GenerativeModel(SensoryInputCoin(coin_set), network)

    pes_list = list()
    for _ in coin_set:
        pes_list.append(model.process())

    ax.plot(pes_list)
    ax.set_title(title + ' - Total Error: ' + str(sum(pes_list)))


def model_for_paired(coin_set):
    network = BayesianModel([('previous', 'current')])
    cpd_a = TabularCPD(variable='previous', variable_card=2, values=[[0.99, 0.01]])
    cpd_b = TabularCPD(variable='current', variable_card=2, values=[[0.01, 0.99],
                                                                    [0.99, 0.01]], evidence=['previous'],
                       evidence_card=[2])

    network.add_cpds(cpd_a, cpd_b)
    network.check_model()

    model = GenerativeModel(SensoryInputCoin(coin_set), network)
    return model


def process_for_paired(coin_set, model, ax, title):
    pes_list = list()
    for _ in coin_set:
        pes_list.append(model.process())

    ax.plot(pes_list)
    ax.set_title(title + ' - Total Error: ' + str(sum(pes_list)))


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

size = 100

f, (ax1) = plt.subplots(1, sharex='all')
coin_set = paired_set(size)
model = model_for_paired(coin_set)
process_for_paired(coin_set, model, ax1, 'Paired')

# f, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
# logging.info("========================================================================================================")
# logging.info("================================================= PAIRED ===============================================")
# logging.info("========================================================================================================")
# process(paired_set(size), ax1, 'Paired')
# logging.info("========================================================================================================")
# logging.info("================================================== SORTED ==============================================")
# logging.info("========================================================================================================")
# process(sorted_set(size), ax2, 'Sorted')
# logging.info("========================================================================================================")
# logging.info("=================================================== RANDOM =============================================")
# logging.info("========================================================================================================")
# process(random_set(size), ax3, 'Random')

logging.debug("showingbefore")
plt.show()
logging.debug("showing")
