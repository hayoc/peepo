class Level:
    def __init__(self, index, regions):
        """
        Predictive Processing Level:

        :param index: Index number of level in Module Hierarchy.

        :param regions: List of Region objects.

        :type index: int
        :type regions: list
        """
        self.index = index
        self.regions = regions

    def __str__(self):
        return 'Level ' + str(self.index)

    def __repr__(self):
        return 'discrete.level({}, {})'.format(str(self.index), str(self.regions))
