from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

MOBILE = 'obs_mobile'
LEFT_ARM = 'motor_left_arm'
RIGHT_ARM = 'motor_right_arm'
LEFT_FOOT = 'motor_left_foot'
RIGHT_FOOT = 'motor_right_foot'

DESIRE = 'obs_desire'
BOREDOM = 'hypo_boredom'
MOTOR_HYPO = 'hypo_motor'


def random_binary_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.5, 0.5]])


def inactivity_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.9, 0.1]])


def baby_model():
    network = BayesianModel()

    network.add_nodes_from([BOREDOM, DESIRE, MOBILE, LEFT_ARM])
    network.add_edges_from([(BOREDOM, DESIRE), (BOREDOM, MOBILE)])

    cpd1 = TabularCPD(variable=DESIRE, variable_card=2, values=[[0.1, 0.9],
                                                                [0.9, 0.1]],
                      evidence=[BOREDOM],
                      evidence_card=[2])
    # cpd2 = TabularCPD(variable=LEFT_ARM, variable_card=2, values=[[0.1, 0.9],
    #                                                               [0.9, 0.1]],
    #                   evidence=[MOTOR_HYPO],
    #                   evidence_card=[2])
    # cpd3 = TabularCPD(variable=MOBILE, variable_card=2, values=[[0.1, 0.1, 0.9, 0.9],
    #                                                             [0.9, 0.9, 0.1, 0.1]],
    #                   evidence=[BOREDOM, MOTOR_HYPO],
    #                   evidence_card=[2, 2])

    cpd3 = TabularCPD(variable=MOBILE, variable_card=2, values=[[0.1, 0.9],
                                                                [0.9, 0.1]],
                      evidence=[BOREDOM],
                      evidence_card=[2])

    cpd2 = TabularCPD(variable=LEFT_ARM, variable_card=2, values=[[0.9, 0.1]])
    cpd4 = TabularCPD(variable=BOREDOM, variable_card=2, values=[[0.6, 0.4]])
    cpd5 = TabularCPD(variable=MOTOR_HYPO, variable_card=2, values=[[0.6, 0.4]])

    network.add_cpds(cpd1, cpd2, cpd3, cpd4)
    network.check_model()

    return network
