from functools import wraps

import numpy as np

import warnings
from ..composition.CanonicalModel import CanonicalModel
from ..utils import code_to_query, debug_print, get_att
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, r2_score

from joblib import Parallel, delayed
from functools import partial

VERBOSITY = 0


def base_induction_algorithm(
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
    """
    Basic induction algorithm. Models according to the m_codes it receives.

    Parameters
    ----------
    data:                   np.ndarray
    m_codes:                np.ndarray
    metadata:               dict
    classifier:
    regressor:
    classifier_kwargs:      dict
    regressor_kwargs:       dict

    Returns
    -------

    """
    assert isinstance(data, np.ndarray)

    # Init
    if classifier_kwargs is None:
        classifier_kwargs = dict()
    if regressor_kwargs is None:
        regressor_kwargs = dict()

    _, n_cols = data.shape
    m_list = []

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

    np.random.seed(random_state)
    random_states = np.random.randint(10 ** 4, size=(len(ids)))

    parameters = []
    for idx, (desc_ids, targ_ids) in enumerate(ids):

        if set(targ_ids).issubset(nominal_attributes):
            kwargs = classifier_kwargs
            learner = classifier
            out_kind = "nominal"
            metric = partial(f1_score, average="macro")
        elif set(targ_ids).issubset(numeric_attributes):
            kwargs = regressor_kwargs
            learner = regressor
            out_kind = "numeric"
            metric =  r2_score

        else:
            msg = """
            Cannot learn mixed (nominal/numeric) models
            """
            raise ValueError(msg)

        kwargs["random_state"] = random_states[idx]

        # Learn a model for current desc_ids-targ_ids combo
        tup = (data, desc_ids, targ_ids, learner, out_kind, metric)
        parameters.append((tup, kwargs))

    if n_jobs > 1:
        msg = """
        Training is being parallellized using Joblib. Number of jobs = {}
        """.format(
            n_jobs
        )
        warnings.warn(msg)
        m_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_learn_model)(*t, **k) for t, k in parameters
        )
    else:
        m_list = [_learn_model(*t, **k) for t, k in parameters]

    return m_list


from sklearn.model_selection import train_test_split


def _learn_model(data, desc_ids, targ_ids, learner, out_kind, metric, **kwargs):
    """
    Learn a model from the data.

    The arguments of this function determine specifics on which task,
    which learner etc.

    Model is a machine learning method that has a .fit() method.
    """

    i, o = data[:, desc_ids], data[:, targ_ids]

    if i.ndim == 1:
        # We always want 2D inputs
        i = i.reshape(-1, 1)
    if o.shape[1] == 1:
        # If output is single variable, we need 1D matrix
        o = o.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        i, o, test_size=0.10, random_state=42
    )

    try:
        model = learner(**kwargs)
        model.fit(X_train, y_train)

        performance = 1
        #y_pred = model.predict(X_test)
        #performance = metric(y_test, y_pred)

    except ValueError as e:
        raise ValueError(out_kind, metric, e)

    # Bookkeeping
    model = CanonicalModel(model, desc_ids, targ_ids, out_kind, performance)

    # model.desc_ids = desc_ids
    # model.targ_ids = targ_ids
    # model.out_kind = out_kind

    return model
