import itertools
import warnings

import numpy as np
import shap
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

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

from mercs.composition.CanonicalModel import CanonicalModel
from mercs.utils import code_to_query, get_i_o


def base_induction_algorithm(
        data,
        m_codes,
        metadata,
        classifier,
        regressor,
        mixed,
        classifier_kwargs,
        regressor_kwargs,
        mixed_kwargs,
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
        mixed {Supported ML mixed learner} -- Supported ML mixed learner
        classifier_kwargs {dict} -- Kwargs for classifier
        regressor_kwargs {dict} -- Kwargs for regressor
        mixed_kwargs {dict} -- kwargs for mixed learner
    
    Keyword Arguments:
        random_state {int} -- Seed for random numbers (default: {997})
        n_jobs {int} -- Joblib can be used in training. (default: {1})
        verbose {int} -- Verbosity level for Joblib. (default: {0})
    
    Raises:
        ValueError: When trying to learn a model with both nominal and numeric outputs.
    
    Returns:
        m_list -- List of learned ML-models
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
        mixed,
        mixed_kwargs,
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
        m_list -- List of learned ML-models
    """
    m_list = base_induction_algorithm(
        data,
        m_codes,
        metadata,
        classifier,
        regressor,
        None,
        classifier_kwargs,
        regressor_kwargs,
        None,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    m_list = _expand_m_list(m_list)

    return m_list


def _expand_m_list(m_list):
    return list(itertools.chain.from_iterable(m_list))


def _build_models(parameters, n_jobs=1, verbose=0):
    """ Method in charge of learning the models based on the given parameters. It can be done is parallel by specifying
    the number of jobs.

    Args:
        parameters: configuration parameters of the models
        n_jobs: number of parallel jobs
        verbose: verbosity level for multi-core learning

    Returns:
        m_list: list of trained models

    """

    if n_jobs < 2:
        if n_jobs < 1:
            msg = """Number of jobs needs to be at least 1. Assuming 1 job."""
            warnings.warn(msg)
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
        mixed,
        mixed_kwargs,
        random_states,
        calculation_method_feature_importances,
        min_nb_samples,
        data,
):
    """ Creates the "parameters" object used by the models

    Args:
        ids: query ids in the form of [(descriptive, target), (descriptive, target), ...]
        nominal_attributes: idxs of the nominal attributes
        classifier: classifier algorithm
        classifier_kwargs: classifier configuration
        numeric_attributes: idxs of the numeric attributes
        regressor: regressor algorithm
        regressor_kwargs: regressor configuration
        mixed: mixed learning algorithm
        mixed_kwargs: mixed learner configuration
        random_states:
        calculation_method_feature_importances:
        min_nb_samples: minimum number of samples
        data: training data

    Returns:
        parameters: list of the model parameters
            - data: training data
            - desc_ids: descriptive parameters
            - targ_ids: target parameters
            - learner: learner algorithm
            - out_kind: output data type(nominal, numeric, mixed)
            - kwargs: model kwargs

    """
    parameters = []
    for idx, (desc_ids, targ_ids) in enumerate(ids):
        if set(targ_ids).issubset(nominal_attributes):
            learner = classifier
            out_kind = "nominal"
            kwargs = classifier_kwargs.copy()  # Copy is essential
            kwargs["random_state"] = random_states[idx]
        elif set(targ_ids).issubset(numeric_attributes):
            learner = regressor
            out_kind = "numeric"
            kwargs = regressor_kwargs.copy()  # Copy is essential
            kwargs["random_state"] = random_states[idx]
        else:
            # Case when target ids contain both numerical and nominal data
            learner = mixed
            out_kind = "mixed"
            kwargs = mixed_kwargs.copy()
            kwargs["classification_targets"] = np.where(
                np.array(list(nominal_attributes)) == np.array(targ_ids)
            )[0].tolist()

        kwargs["calculation_method_feature_importances"] = calculation_method_feature_importances
        kwargs["min_nb_samples"] = min_nb_samples

        kwargs = _add_categorical_features_to_kwargs(
            learner, desc_ids, nominal_attributes, kwargs
        )  # This is learner-specific

        # Learn a model for current desc_ids-targ_ids combo
        args = (data, desc_ids, targ_ids, learner, out_kind)
        parameters.append((args, kwargs))

    return parameters


def _learn_model(
        data,
        desc_ids,
        targ_ids,
        learner,
        out_kind,
        filter_nan=True,
        min_nb_samples=10,
        calculation_method_feature_importances="default",
        **kwargs
):
    """Learn a single model from the data.

    The arguments of this function determine specifics on which task,
    which learner etc.

    Model is a machine learning method that has a .fit() method.

    Args:
        data: training data
        desc_ids: ids of the descriptive attributes
        targ_ids: ids of the target attributes
        learner: learning algorithm
        out_kind: type of the ouput data (numeric, nominal or mixed)
        filter_nan: indicates if NaN values should be filtered
        min_nb_samples: minimum number of samples
        calculation_method_feature_importances:
        **kwargs: keyword argument

    Returns:
        model: the learned model
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


def _get_cat_features(desc_ids, nominal_ids):
    cat_features = np.where(
        np.in1d(np.array(list(desc_ids)), np.array(list(nominal_ids)))
    )[0]
    return [int(c) for c in cat_features]
