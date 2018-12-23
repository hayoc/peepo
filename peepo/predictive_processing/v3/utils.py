import json
import os

from config import ROOT_DIR
from peepo.predictive_processing.v3.peepo_network import PeepoNetwork


def write_to_file(name, peepo_network):
    directory = ROOT_DIR + '/resources/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + '/' + str(name) + '.json', 'w') as outfile:
        json.dump(peepo_network.to_json(), outfile, default=str)


def read_from_file(name):
    with open(ROOT_DIR + '/resources/' + str(name) + '.json') as json_data:
        return PeepoNetwork().from_json(json.load(json_data))


# TEMPORARY
if __name__ == "__main__":
    data = {'BENS_0': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'BENS_1': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            'BENS_2': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            'BENS_3': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            'WORLD_0': [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            'WORLD_1': [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            'WORLD_2': [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]}

    pp_net = PeepoNetwork(bel_nodes=[{'name': 'BENS_0', 'card': 2},
                                     {'name': 'BENS_1', 'card': 2},
                                     {'name': 'BENS_2', 'card': 2},
                                     {'name': 'BENS_3', 'card': 2}],
                          ext_nodes=[{'name': 'WORLD_0', 'card': 2},
                                     {'name': 'WORLD_1', 'card': 2},
                                     {'name': 'WORLD_2', 'card': 2}],
                          edges=[('BENS_1', 'WORLD_0'),
                                 ('BENS_1', 'WORLD_1'),
                                 ('BENS_1', 'WORLD_2'),
                                 ('BENS_2', 'WORLD_0'),
                                 ('BENS_2', 'WORLD_1'),
                                 ('BENS_2', 'WORLD_2'),
                                 ('BENS_3', 'WORLD_0'),
                                 ('BENS_3', 'WORLD_1'),
                                 ('BENS_3', 'WORLD_2')],
                          train_data=data)
    pp_net.assemble()

    write_to_file('color_recognition', pp_net)

    pp_net = read_from_file('color_recognition')
    pp_net.train_data = data

    pm_net = pp_net.to_pomegranate()
    print('')
