from peepo.predictive_processing.v3.network import default_network, predict


class Module:

    def __init__(self):
        self.network = default_network()


m = Module()
print(m.network.get_cpds('B'))
print(predict(m.network)[0].values)
