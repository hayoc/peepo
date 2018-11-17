from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel

MOBILE = 'mobile'
LEFT_ARM = 'limb_left_arm'
RIGHT_ARM = 'limb_right_arm'
LEFT_FOOT = 'limb_left_foot'
RIGHT_FOOT = 'limb_right_foot'


def random_binary_cpd(var):
    return TabularCPD(variable=var, variable_card=2, values=[[0.5, 0.5]])


def baby_model():
    baby_model = BayesianModel()

    baby_model.add_nodes_from([MOBILE, LEFT_ARM, RIGHT_ARM, LEFT_FOOT, RIGHT_FOOT])

    baby_model.add_cpds(random_binary_cpd(MOBILE),
                        random_binary_cpd(LEFT_ARM),
                        random_binary_cpd(RIGHT_ARM),
                        random_binary_cpd(LEFT_FOOT),
                        random_binary_cpd(RIGHT_FOOT))
    baby_model.check_model()

    return baby_model
