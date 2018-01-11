#!/usr/bin/env python3

from multiprocessing import Process

import numpy as np
import logging

from peepo.bot.peepo_bot import Peepo
from peepo.bot.peepo_virtual import PeepoVirtual
from peepo.predictive_processing.discrete.level import Level
from peepo.predictive_processing.discrete.module import Module
from peepo.predictive_processing.discrete.node import Node

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

# peepo_bot = PeepoVirtual()
peepo_bot = Peepo()


def drive(act, peepo):
    peepo[0].drive(0.9)


def steer(act, peepo):
    peepo[0].steer(0.9)


"""
===========================================
========= NO OBSTACLES MODULE =============
===========================================
"""
node_infrared = Node(np.matrix([[0.9, 0.1], [0.1, 0.9]]), steer, [], 0.4, 0.1, 'infrared', peepo_bot)
node_obstacles = Node(np.matrix([[0.9, 0.1], [0.1, 0.9]]), children=[node_infrared], name='obstacles')
node_obstacles.setHyp('unknown_higher', np.array([0.1, 0.9]))

levels_obs = [
    Level(0, [node_obstacles]),
    Level(1, [node_infrared])
]

# ao = np.array([0.9, 0.1])
# actuals_obs = {
#     'infrared': ao
# }
# peepo_bot.set_infrared(ao)

mod_obstacles = Module(levels_obs, peepo_bot.vision)

"""
===========================================
========= ALWAYS MOVING MODULE ============
===========================================
"""

node_drive = Node(np.matrix([[0.9, 0.1], [0.1, 0.9]]), drive, [], 0.4, 0.1, 'drive', peepo_bot)
node_moving = Node(np.matrix([[0.9, 0.1], [0.1, 0.9]]), children=[node_drive], name='moving')
node_moving.setHyp('unknown_higher', np.array([0.1, 0.9]))

levels_mov = [
    Level(0, [node_moving]),
    Level(1, [node_drive])
]

# am = np.array([0.9, 0.1])
# actuals_mov = {
#     'drive': am
# }
# peepo_bot.set_moving(am)


mod_moving = Module(levels_mov, peepo_bot.movement)

"""
===========================================
================= START ===================
===========================================
"""

if __name__ == '__main__':
    p2 = Process(target=mod_moving.run)
    p1 = Process(target=mod_obstacles.run)
    logging.info("Starting process 1...")
    p1.start()
    logging.info("Starting process 2...")
    p2.start()
