import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.predictive_processing.v3.sensory_input import SensoryInput


class SensoryInputCoin(SensoryInput):

    def __init__(self, coin_set):
        super().__init__()
        self.coin_set = coin_set
        self.index = 0

    def action(self, node, prediction_error, prediction):
        pass

    def value(self, name):
        # 0 = heads, 1 = tails
        val = np.array([1, 0]) if self.coin_set[self.index] == 0 else np.array([0, 1])
        self.index += 1
        return val


def paired_set(size):
    return [1 if i % 2 == 0 else 0 for i in range(size)]


def sorted_set(size):
    return [1 if i < size / 2 else 0 for i in range(size)]


def random_set(size):
    return [random.choice([0, 1]) for _ in range(size)]


def heads_heads_tails_set(size):
    return [1 if i % 3 == 0 else 0 for i in range(size)]


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
    cpd_a = TabularCPD(variable='previous', variable_card=2, values=[[0.9, 0.1]])
    cpd_b = TabularCPD(variable='current', variable_card=2, values=[[0.01, 0.99],
                                                                    [0.99, 0.01]], evidence=['previous'],
                       evidence_card=[2])

    network.add_cpds(cpd_a, cpd_b)
    network.check_model()

    model = GenerativeModel(SensoryInputCoin(coin_set), network)
    return model


def model_for_sorted(coin_set):
    network = BayesianModel([('previous', 'current')])
    cpd_a = TabularCPD(variable='previous', variable_card=2, values=[[0.9, 0.1]])
    cpd_b = TabularCPD(variable='current', variable_card=2, values=[[0.99, 0.01],
                                                                    [0.01, 0.99]], evidence=['previous'],
                       evidence_card=[2])

    network.add_cpds(cpd_a, cpd_b)
    network.check_model()

    model = GenerativeModel(SensoryInputCoin(coin_set), network)
    return model


def generic_model(coin_set):
    network = BayesianModel([('A', 'D'), ('B', 'D'), ('C', 'D')])
    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.9, 0.1]])
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.1, 0.9]])
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.1, 0.9]])
    cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9],
                                                              [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1]],
                       evidence=['A', 'B', 'C'], evidence_card=[2, 2, 2])

    network.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)
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

f, (ax1) = plt.subplots(1, sharex='all', sharey='all')

coin_set = heads_heads_tails_set(100)
model = generic_model(coin_set)
plot_result(model, coin_set, ax1, 'Generic')

plt.show()
logging.info("done")

# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
#
# size = 100
#
# f, (ax1, ax2) = plt.subplots(2, sharex='all', sharey='all')
#
# logging.info("========================================================================================================")
# logging.info("================================================== PAIRED ==============================================")
# logging.info("========================================================================================================")
#
# paired_coin_set = paired_set(size)
# paired_model = model_for_paired(paired_coin_set)
# # paired_model = default_model(paired_coin_set)
# plot_result(paired_model, paired_coin_set, ax1, 'Paired')
#
# logging.info("========================================================================================================")
# logging.info("================================================== SORTED ==============================================")
# logging.info("========================================================================================================")
#
# sorted_coin_set = sorted_set(size)
# sorted_model = model_for_sorted(sorted_coin_set)
# # sorted_model = default_model(sorted_coin_set)
# plot_result(sorted_model, sorted_coin_set, ax2, 'Sorted')
#
# plt.show()
# logging.info("================================================== DONE ================================================")
