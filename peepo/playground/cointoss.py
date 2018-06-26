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


def default_model(coin_set):
    network = BayesianModel([('hypo', 'coin')])

    cpd_a = TabularCPD(variable='hypo', variable_card=2, values=[[0.5, 0.5]])
    cpd_b = TabularCPD(variable='coin', variable_card=2, values=[[0.99, 0.01],
                                                                 [0.01, 0.99]], evidence=['hypo'],
                       evidence_card=[2])
    network.add_cpds(cpd_a, cpd_b)
    network.check_model()

    model = GenerativeModel(SensoryInputCoin(coin_set), network)
    return model


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


def model_for_sorted(coin_set):
    network = BayesianModel([('previous', 'current')])
    cpd_a = TabularCPD(variable='previous', variable_card=2, values=[[0.99, 0.01]])
    cpd_b = TabularCPD(variable='current', variable_card=2, values=[[0.99, 0.01],
                                                                    [0.01, 0.99]], evidence=['previous'],
                       evidence_card=[2])

    network.add_cpds(cpd_a, cpd_b)
    network.check_model()

    model = GenerativeModel(SensoryInputCoin(coin_set), network)
    return model


def plot_result(model, coin_set, ax, title):
    pes_list = list()
    for _ in coin_set:
        pes_list.append(model.process())

    ax.plot(pes_list)
    ax.set_title(title + ' - Total Error: ' + str(sum(pes_list)))


logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

size = 100

f, (ax1, ax2) = plt.subplots(2, sharex='all', sharey='all')

logging.info("========================================================================================================")
logging.info("================================================== PAIRED ==============================================")
logging.info("========================================================================================================")

paired_coin_set = paired_set(size)
paired_model = model_for_paired(paired_coin_set)
# paired_model = default_model(paired_coin_set)
plot_result(paired_model, paired_coin_set, ax1, 'Paired')

logging.info("========================================================================================================")
logging.info("================================================== SORTED ==============================================")
logging.info("========================================================================================================")

sorted_coin_set = sorted_set(size)
sorted_model = model_for_sorted(sorted_coin_set)
# sorted_model = default_model(sorted_coin_set)
plot_result(sorted_model, sorted_coin_set, ax2, 'Sorted')

plt.show()
logging.info("================================================== DONE ================================================")
