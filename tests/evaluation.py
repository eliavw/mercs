import numpy as np


def accuracy(y, y_hat):
    # Calculate classification accuracy of model
    return (y.astype(int) == y_hat.astype(int)).sum() / y.size


def rmse(y, y_hat):
    # Calculate root squared mean error of model
    return np.sqrt(((y - y_hat) ** 2).mean())


def compute_scores(y, y_hat, classification_targets):
    scores = np.zeros(y.shape[1])
    # Calculate the classification and regression accuracy of the model
    for i in range(y.shape[1]):
        if i in classification_targets:
            scores[i] = accuracy(y[:, i], y_hat[:, i])
        else:
            scores[i] = rmse(y[:, i], y_hat[:, i])

    return scores
