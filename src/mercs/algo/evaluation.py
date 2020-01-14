import numpy as np
from functools import partial
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from ..utils.inference_tools import _dummy_array
from ..utils import DESC_ENCODING, TARG_ENCODING, MISS_ENCODING


def dummy_evaluation(data, m_codes, m_list, i_list, random_state=42, test_size=0.2):
    m_score = np.zeros(m_codes.shape)
    m_score[np.where(m_codes==TARG_ENCODING)] = 1
    return m_score


def base_evaluation(data, m_codes, m_list, i_list, random_state=42, test_size=0.2):

    # Init
    m_score = np.zeros(m_codes.shape)

    # Data
    _, X = train_test_split(data, test_size=test_size, random_state=random_state)    

    # Evaluate imputers
    i_score = np.zeros(m_codes.shape[0])
    for i_idx, imp in enumerate(i_list):
        o = X[:, i_idx]

        y_true = o
        y_pred = imp.transform(_dummy_array(len(o))).ravel()

        if imp.out_kind in {"nominal"}:
            metric = partial(f1_score, average="macro")
        else:
            metric = r2_score

        imp.score = _calc_performance(y_true, y_pred, metric)
        i_score[i_idx] = imp.score

    # Evaluate models
    for m_idx, mod in enumerate(m_list):
        i, o = X[:, mod.desc_ids], X[:, mod.targ_ids]

        multi_target = o.shape[1] != 1
        if not multi_target:
            # If output is single variable, we need 1D matrix
            o = o.ravel()

        y_true = o
        y_pred = mod.predict(i)

        if mod.out_kind in {"nominal"}:
            metric = partial(f1_score, average="macro")
        else:
            metric = r2_score

        mod.score = _calc_performance(y_true, y_pred, metric, multi_target)
        m_score[m_idx, mod.targ_ids] = (mod.score - i_score[mod.targ_ids])/mod.score
    return np.clip(m_score, 0, 1)


def _calc_performance(y_true, y_pred, metric, multi_target=False):

    if multi_target:
        performance = [
            metric(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
        ]
    else:
        performance = metric(y_true, y_pred)
    return performance
