class Level:
    def __init__(self, index, nodes):
        """
        Predictive Processing Level:

        :param index: Index number of level in Module Hierarchy.

        :param nodes: List of Node objects.

        :type index: int
        :type nodes: list
        """
        self.index = index
        self.nodes = nodes

    def __str__(self):
        return 'Level ' + str(self.index)

    def __repr__(self):
        return 'discrete.level({}, {})'.format(str(self.index), str(self.nodes))
