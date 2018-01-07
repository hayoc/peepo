import logging


class Module:
    def __init__(self, levels, actual):
        self.lvls = levels
        self.act = actual
        self.iter = 0
        self.lvls.sort(key=lambda x: x.index)

    def run(self):
        while True:
            self.predict_flow()

    def predict_flow(self):
        if self.iter < 10:
            self.iter += 1
        else:
            logging.error('Max iterations reached. Aborting')
            return

        logging.debug("---------- PREDICT FLOW --------------")

        for level in self.lvls:
            for node in level.nodes:
                pred = node.predict()
                for child in node.children:
                    child.setHyp(node.name, pred)

        self.error_flow()

    def error_flow(self):
        logging.debug("---------- ERROR FLOW --------------")

        lowest = True
        error = False

        for level in reversed(self.lvls):
            if not lowest and not error:
                return
            for node in level.nodes:
                if lowest:
                    if node.error(node.predict(), self.act[node.name]):
                        error = True
                        node.update(self.act)
                else:
                    for child in node.children:
                        if node.error(node.predict(), child.hyp):
                            error = True
                            node.update(child.hyp)
            lowest = False
        self.predict_flow()
