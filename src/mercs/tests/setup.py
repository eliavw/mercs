import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def default_dataset(random_state=997, n_samples=10 ** 3, n_features=7, df=False):
    """
    Generate a dataset to be used in tests.

    Returns:

    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_repeated=0,
        n_redundant=0,
        n_clusters_per_class=2,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    if df:
        train = pd.DataFrame(X_train)
        train = train.assign(y=y_train)

        test = pd.DataFrame(X_test)
        test = test.assign(y=y_test)

        return train, test
    else:
        return np.c_[X_train, y_train], np.c_[X_test, y_test]
