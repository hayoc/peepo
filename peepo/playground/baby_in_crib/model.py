import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianModel
from pomegranate.BayesianNetwork import BayesianNetwork
from pomegranate.base import State
from pomegranate.distributions.ConditionalProbabilityTable import ConditionalProbabilityTable
from pomegranate.distributions.DiscreteDistribution import DiscreteDistribution

MOBILE = 'LEN_mobile'
LEFT_ARM = 'LEN_motor_left_arm'
RIGHT_ARM = 'LEN_motor_right_arm'
LEFT_FOOT = 'LEN_motor_left_foot'
RIGHT_FOOT = 'LEN_motor_right_foot'

DESIRE = 'LEN_desire'
BOREDOM = 'RON_BOREDOM'
MOTOR_HYPO = 'RON_MOVEMENT'


def baby_model():
    d1 = DiscreteDistribution({'0': 0.6, '1': 0.4})
    d2 = DiscreteDistribution({'0': 0.6, '1': 0.4})
    d3 = ConditionalProbabilityTable(
        [['1', '1', 0.1],
         ['1', '0', 0.9],
         ['0', '1', 0.9],
         ['0', '0', 0.1]], [d1])
    d4 = ConditionalProbabilityTable(
        [['1', '1', '1', 0.1],
         ['1', '1', '0', 0.9],
         ['1', '0', '1', 0.1],
         ['1', '0', '0', 0.9],
         ['0', '1', '1', 0.9],
         ['0', '1', '0', 0.1],
         ['0', '0', '1', 0.9],
         ['0', '0', '0', 0.1]], [d1, d2])
    d5 = ConditionalProbabilityTable(
        [['1', '1', 0.1],
         ['1', '0', 0.9],
         ['0', '1', 0.9],
         ['0', '0', 0.1]], [d2])

    s1 = State(d1, name=BOREDOM)
    s2 = State(d2, name=MOTOR_HYPO)
    s3 = State(d3, name=DESIRE)
    s4 = State(d4, name=MOBILE)
    s5 = State(d5, name=LEFT_ARM)

    model = BayesianNetwork()
    model.add_states(s1, s2, s3, s4, s5)
    model.add_edge(s1, s3)
    model.add_edge(s1, s4)
    model.add_edge(s2, s4)
    model.add_edge(s2, s5)
    model.bake()

    return model


TRAINING_DATA = pd.DataFrame(data={
    BOREDOM: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    DESIRE: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    MOBILE: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    MOTOR_HYPO: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    LEFT_ARM: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
})


def fully_connected_model(nodes=None):
    if not nodes:
        nodes = [BOREDOM, DESIRE, MOBILE, MOTOR_HYPO, LEFT_ARM]
    network = BayesianModel()
    network.add_nodes_from(nodes)

    for hypo in nodes:
        if 'hypo' in hypo:
            for obs in nodes:
                if 'obs' in obs or 'motor' in obs:
                    network.add_edge(u=hypo, v=obs)

    network.fit(TRAINING_DATA, estimator=BayesianEstimator, prior_type="BDeu")

    return network



