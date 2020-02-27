import numpy as np
from functools import partial
from sklearn.preprocessing import maxabs_scale, minmax_scale
from sklearn.metrics import f1_score, r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from ..utils.inference_tools import _dummy_array
from ..utils import DESC_ENCODING, TARG_ENCODING, MISS_ENCODING, get_i_o


def dummy_evaluation(
    data, m_codes, m_list, i_list, random_state=42, test_size=0.2, **kwargs
):
    return _dummy_evaluation(m_codes)


def base_evaluation(
    data,
    m_codes,
    m_list,
    i_list,
    random_state=42,
    test_size=0.2,
    per_attribute_normalization=False,
    consider_imputations=False,
    **kwargs
):

    # Data
    _, X = train_test_split(data, test_size=test_size, random_state=random_state)

    m_score = _model_evaluation(X, m_list, m_codes)

    if consider_imputations:
        i_score = _imputer_evaluation(X, i_list)
        m_score = normalize_m_score_relative_to_imputation(
            m_score, i_score, per_attribute_normalization=per_attribute_normalization
        )
    else:
        dummy_evaluation = _dummy_evaluation(m_codes)
        m_score = normalize_m_score(
            m_score,
            dummy_evaluation,
            per_attribute_normalization=per_attribute_normalization,
        )

    return m_score


def normalize_m_score_relative_to_imputation(
    m_score, i_score, per_attribute_normalization=False
):
    diff = m_score - i_score

    # All the models that do worse than the baseline
    diff[np.where(diff < 0)] = 0

    # Per attribute relative scaling.
    if per_attribute_normalization:
        norm_m_score = maxabs_scale(diff, axis=0)
    else:
        norm_m_score = maxabs_scale(diff.flat, axis=0).reshape(diff.shape)

    assert np.min(norm_m_score) >= 0
    return norm_m_score


def normalize_m_score(m_score, dummy_evaluation, per_attribute_normalization=False):
    # Per attribute relative scaling.
    if per_attribute_normalization:
        norm_m_score = minmax_scale(m_score, axis=0)
    else:
        norm_m_score = minmax_scale(m_score.flat, axis=0).reshape(m_score.shape)

    # Set zeroes again
    norm_m_score[np.where(dummy_evaluation != 1)] = 0.0
    norm_m_score[np.where(dummy_evaluation == 1)] = (
        norm_m_score[np.where(dummy_evaluation == 1)]
        + (1 - norm_m_score[np.where(dummy_evaluation == 1)]) * 0.01
    )

    assert np.min(norm_m_score) >= 0
    return norm_m_score


def _model_evaluation(X, m_list, m_codes):
    m_score = np.zeros(m_codes.shape)
    for m_idx, mod in enumerate(m_list):
        i, o = get_i_o(X, mod.desc_ids, mod.targ_ids, filter_nan=True)

        # i, o = X[:, mod.desc_ids], X[:, mod.targ_ids]

        multi_target = o.shape[1] != 1
        if not multi_target:
            # If output is single variable, we need 1D matrix
            o = o.ravel()

        y_true = o
        y_pred = mod.predict(i)

        metric = _select_metric(mod)
        mod.score = _calc_performance(y_true, y_pred, metric, multi_target)
        m_score[m_idx, mod.targ_ids] = mod.score

        # m_score[m_idx, mod.targ_ids] = (mod.score - i_score[mod.targ_ids])/mod.score
    return m_score


def _imputer_evaluation(X, i_list):
    # Evaluate imputers
    i_score = np.zeros(len(i_list))
    for i_idx, imp in enumerate(i_list):
        _, o = get_i_o(X, [], [i_idx], filter_nan=True)

        # o = X[:, i_idx]

        y_true = o
        y_pred = imp.transform(_dummy_array(len(o))).ravel()

        metric = _select_metric(imp)
        imp.score = _calc_performance(y_true, y_pred, metric)
        i_score[i_idx] = imp.score
    return i_score


def _dummy_evaluation(m_codes):
    m_score = np.zeros(m_codes.shape)
    m_score[np.where(m_codes == TARG_ENCODING)] = 1
    return m_score


def _select_metric(model):
    if model.out_kind in {"nominal"}:
        metric = accuracy_score
        #metric = partial(f1_score, average="macro")
    else:
        metric = normalized_root_mean_squared_error
        # metric = r2_score
    return metric


def _calc_performance(y_true, y_pred, metric, multi_target=False):

    if multi_target:
        performance = [
            metric(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
        ]
    else:
        performance = metric(y_true, y_pred)
    return performance


def normalized_root_mean_squared_error(y_true, y_pred):
    """Compute normalized mean squared error
    Cf. missForest
    
    Returns:
        NMRSE -- Result
    """
    mse = mean_squared_error(y_true, y_pred)
    var = np.var(y_true)

    nrmse = np.sqrt(mse / var)
    return 1 - nrmse
