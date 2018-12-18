import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

MOBILE = 'obs_mobile'
LEFT_ARM = 'motor_left_arm'
RIGHT_ARM = 'motor_right_arm'
LEFT_FOOT = 'motor_left_foot'
RIGHT_FOOT = 'motor_right_foot'

DESIRE = 'obs_desire'
BOREDOM = 'hypo_boredom'
MOTOR_HYPO = 'hypo_movement'


def random_binary_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.5, 0.5]])


def inactivity_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.9, 0.1]])


def baby_model():
    model = fully_connected_model()
    return model
    # network = BayesianModel()
    #
    # network.add_nodes_from([BOREDOM, DESIRE, MOBILE, LEFT_ARM])
    # network.add_edges_from([(BOREDOM, DESIRE), (BOREDOM, MOBILE), (MOTOR_HYPO, MOBILE), (MOTOR_HYPO, LEFT_ARM)])
    #
    # cpd1 = TabularCPD(variable=DESIRE, variable_card=2, values=[[0.1, 0.9],
    #                                                             [0.9, 0.1]],
    #                   evidence=[BOREDOM],
    #                   evidence_card=[2])
    # cpd2 = TabularCPD(variable=LEFT_ARM, variable_card=2, values=[[0.1, 0.9],
    #                                                               [0.9, 0.1]],
    #                   evidence=[MOTOR_HYPO],
    #                   evidence_card=[2])
    # cpd3 = TabularCPD(variable=MOBILE, variable_card=2, values=[[0.1, 0.1, 0.9, 0.9],
    #                                                             [0.9, 0.9, 0.1, 0.1]],
    #                   evidence=[BOREDOM, MOTOR_HYPO],
    #                   evidence_card=[2, 2])
    # cpd4 = TabularCPD(variable=BOREDOM, variable_card=2, values=[[0.6, 0.4]])
    # cpd5 = TabularCPD(variable=MOTOR_HYPO, variable_card=2, values=[[0.6, 0.4]])
    #
    # network.add_cpds(cpd1, cpd2, cpd3, cpd4, cpd5)
    # network.check_model()
    #
    # return network


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



