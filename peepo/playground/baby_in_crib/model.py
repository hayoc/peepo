from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

MOBILE = 'obs_mobile'
BOREDOM = 'obs_boredom'
LEFT_ARM = 'motor_left_arm'
RIGHT_ARM = 'motor_right_arm'
LEFT_FOOT = 'motor_left_foot'
RIGHT_FOOT = 'motor_right_foot'


def random_binary_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.5, 0.5]])


def inactivity_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.9, 0.1]])


def baby_model():
    network = BayesianModel()

    network.add_nodes_from([MOBILE, BOREDOM, LEFT_ARM, RIGHT_ARM, LEFT_FOOT, RIGHT_FOOT])

    network.add_cpds(random_binary_cpd(MOBILE),
                     random_binary_cpd(BOREDOM),
                     inactivity_cpd(LEFT_ARM),
                     inactivity_cpd(RIGHT_ARM),
                     inactivity_cpd(LEFT_FOOT),
                     inactivity_cpd(RIGHT_FOOT))
    network.check_model()

    return network
