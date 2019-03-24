class Peepo:
    """
    Abstract class used in generative model.
    Generative Model will call the action and observation functions in order to do active inference and hypothesis
    update respectively.

    When a LEAF node contains 'motor' in the name, the action method will be executed to minimize prediction error.
    Otherwise the LEAF node will minimize the prediction error by fetching the OBSERVED value and performing a
    hypothesis update.
    """

    def __init__(self, network):
        self.network = network
        pass

    def action(self, node, prediction):
        pass

    def observation(self, name):
        pass
