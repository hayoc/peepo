from sklearn.datasets import load_svmlight_file
import numpy as np


def load(path):
    X, y = load_svmlight_file(path)
    X = X.toarray()
    # Normalize - no negative values!
    X = (X - X.min(0)) / X.ptp(0)

    data = []

    # Weights are feature values + class labels
    for i in range(0, len(y)):
        label = int(y[i])
        weights = X[i]
        labels = np.zeros(int(y.max()))
        labels[label - 1] = 1.
        weights = np.append(weights, labels)
        data.append(weights)

    return np.array(data)