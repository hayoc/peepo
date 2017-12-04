class Module:

    def __init__(self, levels):
        self.levels = levels

    def flow(self):
        for level in self.levels:
            for region in level.regions:
                pred = region.predict()


    def resolve(self):
        pass