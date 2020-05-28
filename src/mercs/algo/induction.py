import warnings
from functools import partial, wraps
import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from joblib import Parallel, delayed
from ..utils.decoration import decorate_tree

from sklearn.preprocessing import normalize
import shap


try:
    from xgboost import XGBClassifier as XGBC
    from xgboost import XGBRegressor as XGBR
except:
    XGBC, XGBR = None, None

try:
    from lightgbm import LGBMClassifier as LGBMC
    from lightgbm import LGBMRegressor as LGBMR
except:
    LGBMC, LGBMR = None, None

try:
    from catboost import CatBoostClassifier as CBC
    from catboost import CatBoostRegressor as CBR
except:
    CBC, CBR = None, None

try:
    from wekalearn import RandomForestClassifier as WLC
    from wekalearn import RandomForestRegressor as WLR
except:
    WLC, WLR = None, None

from ..composition.CanonicalModel import CanonicalModel
from ..utils import code_to_query, debug_print, get_att, get_i_o


def base_induction_algorithm(
    data,
    m_codes,
    metadata,
    classifier,
    regressor,
    classifier_kwargs,
    regressor_kwargs,
    random_state=997,
    calculation_method_feature_importances="default",
    min_nb_samples=10,
    n_jobs=1,
    verbose=0,
):
    """Basic induction algorithm. Models according to the m_codes it receives.
    
    Arguments:
        data {np.ndarray} -- Input data
        m_codes {np.ndarray} -- Model codes
        metadata {dict} -- Metadata of MERCS
        classifier {Supported ML learner} -- Supported ML learner
        regressor {Supported ML learner} -- Supported ML learner
        classifier_kwargs {dict} -- Kwargs for classifier
        regressor_kwargs {dict} -- Kwargs for regressor
    
    Keyword Arguments:
        random_state {int} -- Seed for random numbers (default: {997})
        n_jobs {int} -- Joblib can be used in training. (default: {1})
        verbose {int} -- Verbosity level for Joblib. (default: {0})
    
    Raises:
        ValueError: When trying to learn a model with both nominal and numeric outputs.
    
    Returns:
        list -- List of learned ML-models
    """

    assert isinstance(data, np.ndarray)

    # Init
    if classifier_kwargs is None:
        classifier_kwargs = dict()
    if regressor_kwargs is None:
        regressor_kwargs = dict()

    n_rows, n_cols = data.shape
    attributes = set(range(n_cols))
    nominal_attributes = metadata.get("nominal_attributes")
    numeric_attributes = metadata.get("numeric_attributes")
    msg = """Not all attributes of m_codes are accounted for in metadata"""
    assert not (attributes - nominal_attributes - numeric_attributes), msg

    # Codes to queries
    ids = [
        (d, t)
        for d, t, _ in [code_to_query(m_code, return_list=True) for m_code in m_codes]
    ]

    # Generate a list of random seeds.
    np.random.seed(random_state)
    random_states = np.random.randint(10 ** 3, size=(len(ids)), dtype=int)

    parameters = _build_parameters(
        ids,
        nominal_attributes,
        classifier,
        classifier_kwargs,
        numeric_attributes,
        regressor,
        regressor_kwargs,
        random_states,
        calculation_method_feature_importances,
        min_nb_samples,
        data,
    )

    # Learn Models
    m_list = _build_models(parameters, n_jobs=n_jobs, verbose=verbose)

    return m_list


def expand_induction_algorithm(
    data,
    m_codes,
    metadata,
    classifier,
    regressor,
    classifier_kwargs,
    regressor_kwargs,
    random_state=997,
    n_jobs=1,
    verbose=0,
):
    """Basic induction algorithm. Models according to the m_codes it receives.
    
    Arguments:
        data {np.ndarray} -- Input data
        m_codes {np.ndarray} -- Model codes
        metadata {dict} -- Metadata of MERCS
        classifier {Supported ML learner} -- Supported ML learner
        regressor {Supported ML learner} -- Supported ML learner
        classifier_kwargs {dict} -- Kwargs for classifier
        regressor_kwargs {dict} -- Kwargs for regressor
    
    Keyword Arguments:
        random_state {int} -- Seed for random numbers (default: {997})
        n_jobs {int} -- Joblib can be used in training. (default: {1})
        verbose {int} -- Verbosity level for Joblib. (default: {0})
    
    Raises:
        ValueError: When trying to learn a model with both nominal and numeric outputs.
    
    Returns:
        list -- List of learned ML-models
    """
    m_list = base_induction_algorithm(
        data,
        m_codes,
        metadata,
        classifier,
        regressor,
        classifier_kwargs,
        regressor_kwargs,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    m_list = _expand_m_list(m_list)

    return m_list


def _expand_m_list(m_list):
    return list(itertools.chain.from_iterable(m_list))


def _build_models(parameters, n_jobs=1, verbose=0):
    m_list = []
    if n_jobs < 2:
        m_list = [_learn_model(*a, **k) for a, k in parameters]
    else:
        msg = """Training is being parallellized using Joblib. Number of jobs = {}""".format(
            n_jobs
        )
        warnings.warn(msg)

        m_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_learn_model)(*a, **k) for a, k in parameters
        )
    return m_list


def _build_parameters(
    ids,
    nominal_attributes,
    classifier,
    classifier_kwargs,
    numeric_attributes,
    regressor,
    regressor_kwargs,
    random_states,
    calculation_method_feature_importances,
    min_nb_samples,
    data,
):
    parameters = []
    for idx, (desc_ids, targ_ids) in enumerate(ids):

        if set(targ_ids).issubset(nominal_attributes):
            learner = classifier
            out_kind = "nominal"
            metric = partial(f1_score, average="macro")
            kwargs = classifier_kwargs.copy()  # Copy is essential
        elif set(targ_ids).issubset(numeric_attributes):
            learner = regressor
            out_kind = "numeric"
            metric = r2_score
            kwargs = regressor_kwargs.copy()  # Copy is essential
        else:
            msg = """
            Cannot learn mixed (i.e. nominal+numeric) models
            """
            raise ValueError(msg)

        kwargs["random_state"] = random_states[idx]
        kwargs[
            "calculation_method_feature_importances"
        ] = calculation_method_feature_importances
        kwargs["min_nb_samples"] = min_nb_samples

        kwargs = _add_categorical_features_to_kwargs(
            learner, desc_ids, nominal_attributes, kwargs
        )  # This is learner-specific

        # Learn a model for current desc_ids-targ_ids combo
        args = (data, desc_ids, targ_ids, learner, out_kind, metric)
        parameters.append((args, kwargs))

    return parameters


def _learn_model(
    data,
    desc_ids,
    targ_ids,
    learner,
    out_kind,
    metric,
    filter_nan=True,
    min_nb_samples=10,
    calculation_method_feature_importances="default",
    **kwargs
):
    """
    Learn a model from the data.

    The arguments of this function determine specifics on which task,
    which learner etc.

    Model is a machine learning method that has a .fit() method.
    """
    assert learner is not None

    i, o = get_i_o(data, desc_ids, targ_ids, filter_nan=filter_nan)

    if i.shape[0] < min_nb_samples:
        msg = """
        Only {} samples available for training.
        min_nb_samples is set to {}.
        Therefore no training occured.
        """.format(
            i.shape[0], min_nb_samples
        )
        warnings.warn(msg)
        return None
    else:
        # Pre-processing
        if i.ndim == 1:
            # We always want 2D inputs
            i = i.reshape(-1, 1)

        multi_target = o.shape[1] != 1
        if not multi_target:
            # If output is single variable, we need 1D matrix
            o = o.ravel()

        if learner in {LGBMC, LGBMR}:
            categorical_feature = kwargs.pop("categorical_feature")
            model = learner(**kwargs)
            model.fit(i, o, categorical_feature=categorical_feature)
        else:
            model = learner(**kwargs)
            model.fit(i, o)

        performance = 1.0

        if calculation_method_feature_importances in {"shap"}:
            model.shap_values_ = _calculate_shap_values(model, i)

        # Bookkeeping
        model = CanonicalModel(model, desc_ids, targ_ids, out_kind, performance)

        return model


# Helpers
def _calculate_shap_values(model, X):
    shap_values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(shap_values, list):
        r = _summarize_shaps(shap_values[0])
    else:
        r = _summarize_shaps(shap_values)
    return r


def _summarize_shaps(shap_values):
    avgs_values = np.mean(np.abs(shap_values), axis=0)
    return np.squeeze(normalize(avgs_values.reshape(1, -1), norm="l1"))


def _add_categorical_features_to_kwargs(learner, desc_ids, nominal_attributes, kwargs):
    assert learner is not None
    cat_features = _get_cat_features(desc_ids, nominal_attributes)

    if learner in {CBC, CBR}:
        kwargs["cat_features"] = cat_features
    elif learner in {WLC, WLR}:
        kwargs["cat_features"] = cat_features
    elif learner in {LGBMC, LGBMR}:
        kwargs["categorical_feature"] = cat_features

    return kwargs


def _score_model(model, metric, y_test, y_pred, multi_target):
    try:
        performance = _calc_performance(y_test, y_pred, metric, multi_target)
    except ValueError as e:
        mean = np.nanmean(y_pred)
        if not np.isfinite(mean):
            warnings.warn(
                """We have a model that cannot solve anything. Something shady might be going on."""
            )
            performance = 0
        else:
            y_pred[np.isnan(y_pred)] = mean
            assert np.all(np.isfinite(y_pred))
            performance = _calc_performance(y_test, y_pred, metric, multi_target)

    return performance


def _calc_performance(y_test, y_pred, metric, multi_target):
    if multi_target:
        performance = [
            metric(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])
        ]
    else:
        performance = metric(y_test, y_pred)
    return performance


def _get_cat_features(desc_ids, nominal_ids):
    cat_features = np.where(
        np.in1d(np.array(list(desc_ids)), np.array(list(nominal_ids)))
    )[0]
    return [int(c) for c in cat_features]
