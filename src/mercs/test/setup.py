import os
from os.path import dirname

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def data_directory():
    test_dir = dirname(__file__)
    data_dir = os.path.join(test_dir, "data")  # Slight HARDCODE!
    return data_dir


def filename_data(dataset="iris", kind="train", separator="-", extension="csv"):

    data_dir = data_directory()
    filename = separator.join([dataset, kind]) + ".{}".format(extension)
    return os.path.join(data_dir, filename)


def _detect_nominal(df):
    nominal_columns = set(df.select_dtypes(exclude=["float"]).columns)

    nominal_ids = {i for i, c in enumerate(df.columns) if c in nominal_columns}

    return nominal_ids


def load_iris(df=False, nominal_ids=True):

    fn_train = filename_data(dataset="iris", kind="train")
    fn_test = filename_data(dataset="iris", kind="test")

    df_train = pd.read_csv(fn_train, header=None)
    df_test = pd.read_csv(fn_test, header=None)

    result = []
    result.append(df_train if df else df_train.values)
    result.append(df_test if df else df_test.values)

    if nominal_ids:
        result.append(_detect_nominal(df_train))

    return result


def default_dataset(
    random_state=RANDOM_STATE,
    n_samples=10 ** 3,
    n_features=7,
    n_redundant=0,
    n_repeated=0,
    test_size=0.2,
    n_clusters_per_class=2,
    df=False,
    **kwargs
):
    """
    Generate a dataset to be used in tests.

    Returns:

    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_repeated=n_repeated,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state,
        **kwargs
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if df:
        train = pd.DataFrame(X_train)
        train = train.assign(y=y_train)

        test = pd.DataFrame(X_test)
        test = test.assign(y=y_test)

        return train, test
    else:
        return np.c_[X_train, y_train], np.c_[X_test, y_test]
