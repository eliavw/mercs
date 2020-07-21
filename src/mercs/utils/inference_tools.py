import numpy as np


# Helpers - Data Handling
def dummy_array(nb_rows):
    a = np.empty((nb_rows, 1))
    a.fill(np.nan)
    return a


def pad_proba(classes, all_classes):
    idx = _map_classes(classes, all_classes)

    def pad(X):
        R = np.zeros((X.shape[0], len(all_classes)))
        R[:, idx] = X
        return R

    return pad


def _map_classes(classes, all_classes):
    sorted_idx = np.argsort(all_classes)
    matches = np.searchsorted(all_classes[sorted_idx], classes)
    return sorted_idx[matches]


def select_numeric(idx):
    def select(X):
        if X.ndim == 2:
            return X.take(idx, axis=1)
        else:
            return X

    return select


def select_nominal(idx):
    def select(X, model_type):
        if isinstance(X, list):
            return X[idx]
        elif model_type == "mixed":
            return X[:, idx].tolist()
        elif isinstance(X, np.ndarray):
            return X

    return select
