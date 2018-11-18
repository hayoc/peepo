"""
Baby in crib experiment.

See: Conjugate Reinforcement of Infant Exploratory Behavior - http://dx.doi.org/10.1016/0022-0965(69)90025-3

A baby is placed in a crib with a mobile. One of the baby's limbs (e.g. left foot) is attached to the crib with
a ribbon. Due to motor babbling the baby comes to know the causal effect of moving its left foot which results in
the mobile moving. Once the ribbon is cut and tied to another limb (e.g. right foot), the baby will display a short
period of time of frantic movement of the left food (i.e. the first limb connected) but after a while realize that
the causal structure of the world has shifted and that the mobile is now connected to its right foot.

In terms of predictive processing we expect first little prediction error as the baby is merely exploring. Once it
expects the mobile to be attached to its left foot prediction error will increase as it tries to intentionally move
the mobile, until it has sufficiently learned the causal model and can consistently move the mobile. Once the ribbon
is moved to the other limb - a large increase in prediction error should be witnessed, until this again lowers as the
baby learns the new causal model.
"""
import logging

import peepo.playground.baby_in_crib.model as md
from peepo.playground.baby_in_crib.crib import Crib
from peepo.playground.baby_in_crib.sensory_input import SensoryInputCribBaby
from peepo.predictive_processing.v3.generative_model import GenerativeModel
from peepo.visualize.graph import draw_network

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class Baby:

    def __init__(self):
        self.limbs = {
            md.LEFT_ARM: False,
            md.RIGHT_ARM: False,
            md.LEFT_FOOT: False,
            md.RIGHT_FOOT: False
        }


baby = Baby()
crib = Crib()
network = md.baby_model()

model = GenerativeModel(SensoryInputCribBaby(baby, crib), network)

i = 0
while True:
    model.process()
    i += 1
    if i > 1000:
        draw_network(model.network)
        i = 0
