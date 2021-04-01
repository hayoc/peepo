import logging
import random

import matplotlib.pyplot as plt
import numpy as np

from peepo.pp.generative_model import GenerativeModel
from peepo.pp.peepo import Peepo
from peepo.pp.peepo_network import PeepoNetwork


class CointossPeepo(Peepo):

    def __init__(self, coin_set, model_type):
        super().__init__(self.create_model(model_type))
        self.coin_set = coin_set
        self.index = 0
        self.generative_model = GenerativeModel(self, n_jobs=1)

    def update(self):
        return self.generative_model.process()

    def action(self, node, prediction):
        pass

    def observation(self, name):
        val = np.array([1, 0]) if self.coin_set[self.index] == 0 else np.array([0, 1])
        self.index += 1
        return val

    @staticmethod
    def create_model(model_type):
        a = 'A'
        b = 'B'
        c = 'C'
        d = 'D'

        if model_type == "paired":
            pp_network = PeepoNetwork(
                ron_nodes=[
                    {'name': a, 'card': 2},
                ],
                ext_nodes=[
                    {'name': d, 'card': 2},
                ],
                pro_nodes=[
                ],
                edges=[
                    (a, d),
                ],
                cpds={
                    a: [0.5, 0.5],
                    d: [[0.99, 0.01],
                        [0.01, 0.99]]
                })
        elif model_type == "sorted":
            pp_network = PeepoNetwork(
                ron_nodes=[
                    {'name': a, 'card': 2},
                ],
                ext_nodes=[
                    {'name': d, 'card': 2},
                ],
                pro_nodes=[
                ],
                edges=[
                    (a, d),
                ],
                cpds={
                    a: [0.9, 0.1],
                    d: [[0.01, 0.99],
                        [0.99, 0.01]]
                })
        elif model_type == "default":
            pp_network = PeepoNetwork(
                ron_nodes=[
                    {'name': a, 'card': 2},
                ],
                ext_nodes=[
                    {'name': d, 'card': 2},
                ],
                pro_nodes=[
                ],
                edges=[
                    (a, d),
                ],
                cpds={
                    a: [0.5, 0.5],
                    d: [[0.99, 0.01],
                        [0.01, 0.99]]
                })
        else:
            pp_network = PeepoNetwork(
                ron_nodes=[
                    {'name': a, 'card': 2},
                    {'name': b, 'card': 2},
                    {'name': c, 'card': 2}
                ],
                ext_nodes=[
                    {'name': d, 'card': 2},
                ],
                pro_nodes=[
                ],
                edges=[
                    (a, d),
                    (b, d),
                    (c, d),
                ],
                cpds={
                    a: [0.9, 0.1],
                    b: [0.1, 0.9],
                    c: [0.1, 0.9],
                    d: [[0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9],
                        [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1]]
                })

        pp_network.assemble()

        return pp_network


def paired_set(size):
    return [1 if i % 2 == 0 else 0 for i in range(size)]


def sorted_set(size):
    return [1 if i < size / 2 else 0 for i in range(size)]


def random_set(size):
    return [random.choice([0, 1]) for _ in range(size)]


def heads_heads_tails_set(size):
    return [1 if i % 15 == 0 else 0 for i in range(size)]


def plot_result(model, coin_set, ax, title):
    pes_list = list()
    for _ in coin_set:
        pes_list.append(model.update())

    ax.plot(pes_list)
    ax.set_title(title + ' - Total Error: ' + str(sum(pes_list)))


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    coin_set_size = 100
    paired_coin_set = paired_set(coin_set_size)
    sorted_coin_set = sorted_set(coin_set_size)
    random_coin_set = random_set(coin_set_size)
    hht_coin_set = heads_heads_tails_set(coin_set_size)

    print(paired_coin_set)
    print(sorted_coin_set)
    print(random_coin_set)
    print(hht_coin_set)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='all', sharey='all')

    plot_result(CointossPeepo(paired_coin_set, "paired"), paired_coin_set, ax1, 'Paired')
    # plot_result(CointossPeepo(sorted_coin_set, "default"), sorted_coin_set, ax2, 'Sorted')
    # plot_result(CointossPeepo(random_coin_set, "default"), random_coin_set, ax3, 'Random')
    # plot_result(CointossPeepo(hht_coin_set, "default"), hht_coin_set, ax4, 'HHT')

    plt.show()
