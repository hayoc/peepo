class SensoryInput:
    """
    SensoryInput represents the actual observed input an intelligent organism receives through it's sensors. These
    can be exteroceptive, proprioceptive or interoceptive. Both PERCEPTION and ACTION are represented here in the
    SensoryInput class.
    When a LEAF node contains 'motor' in the name, the action method will be executed to minimize prediction error.
    Otherwise the LEAF node will minimize the prediction error by fetching the OBSERVED value and performing a
    hypothesis update.
    """

    def __init__(self):
        pass

    def action(self, node, prediction):
        pass

    def value(self, name):
        pass
