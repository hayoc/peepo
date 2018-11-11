import numpy as np

from peepo.predictive_processing.continuous import dataload
from peepo.predictive_processing.continuous.region import Region


class Hierarchy:

    def __init__(self):
        pass

    def test(self):
        weights = dataload.load("/home/data/usps")
        testdata = dataload.load("/home/data/usps.t")

        i, c = 0, 0

        for row in testdata:
            inputlabels = list(row[-10:])
            region = Region(row, len(weights), weights)
            region.iterate(75)
            outputlabels = region.r[-10:]

            inputlabel = np.array(inputlabels).argmax()
            outputlabel = outputlabels.argmax()

            print("======== " + str(i) + " =========")
            if inputlabel == outputlabel:
                print("CORRECT " + str(inputlabel) + " - " + str(outputlabel))
                c += 1
            else:
                print("INCORECT " + str(inputlabel) + " - " + str(outputlabel))
            i += 1

        print("========================")
        print("PERCENTAGE CORRECT: " + str(i / c))


h = Hierarchy()
h.test()