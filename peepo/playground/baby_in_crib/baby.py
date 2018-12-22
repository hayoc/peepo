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


model = GenerativeModel(md.baby_model(), SensoryInputCribBaby(Baby(), Crib()))

while True:
    model.process()

# def run():
#     model = GenerativeModel(SensoryInputCribBaby(Baby(), Crib()), md.fully_connected_model())
#     # model.process()
#     original_model = model.network.copy()
#     edges = original_model.edges()
#
#     result = {}
#
#     for x in range(0, len(edges) + 1):
#         subresult = []
#         for cmb in itertools.combinations(edges, x):
#             copy = original_model.copy()
#
#             for cpd in copy.get_cpds():
#                 copy.remove_cpds(cpd)
#
#             for edge in cmb:
#                 copy.remove_edge(edge[0], edge[1])
#
#             copy.fit(md.TRAINING_DATA, estimator=BayesianEstimator, prior_type="BDeu")
#             model = GenerativeModel(SensoryInputCribBaby(Baby(), Crib()), copy)
#
#             loops = -1
#             for l in range(0, 20):
#                 total_error = model.process()
#                 if total_error < 0.1:
#                     loops = l
#                     break
#
#             subresult.append({
#                 "score": loops,
#                 "edges": copy.edges()})
#
#             logging.info('---------' + str(x) + '-------------')
#             logging.info('SCORE: ' + str(loops))
#             logging.info(copy.edges())
#             logging.info('--------------------------')
#         result[str(x)] = subresult
#
#     print('======================================')
#     print('======================================')
#     print('======================================')
#     print('======================================')
#     print('======================================')
#     print('======================================')
#     print('======================================')
#
#     print(result)
#     # with open('lolk.json', 'w') as outfile:
#     #     json.dump(result, outfile)
#     #
#     # # import json
#     # #
#     # # dfs = []
#     # # with open('lolk.json') as json_data:
#     # #     hahajk = json.load(json_data)
#     # #     for x in range(0, 6):
#     # #         durp = pd.DataFrame(hahajk[str(x)])
#     # #         dfs.append(durp['score'])
#     # #
#     # # prottyke = pd.concat(dfs)
#     # # prottyke.plot(kind='bar')
#
#
# if __name__ == "__main__":
#     logging.basicConfig()
#     logging.getLogger().setLevel(logging.DEBUG)
#     run()
