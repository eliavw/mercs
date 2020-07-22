import numpy as np
import pytest

from src.mercs.core import Mercs
from tests import default_dataset


@pytest.fixture
def mercs():
    mercs = Mercs(
        selection_algorithm="base",
        inference_algorithm="base",
        prediction_algorithm="it",
        mixed_algorithm="morfist",
        max_depth=4,
        nb_targets=2,
        nb_iterations=2,
        n_jobs=1,
        verbose=1,
        max_steps=8,
    )
    train, test = default_dataset(n_features=3)

    # ids of the nominal variables
    nominal_ids = {train.shape[1] - 1}

    # fit the model
    mercs.fit(train, nominal_attributes=nominal_ids)

    return mercs


def test_mixed_mode(mercs):
    assert mercs.mixed_algorithm is not None


def test_codes(mercs):
    assert [0, 0, 1, 1] in mercs.m_codes


def test_prediction(mercs):
    _, test = default_dataset(n_features=3)
    q_code = np.array([0, 0, 0, 1])

    # predict value of query code for test data
    y_pred = mercs.predict(test, q_code=q_code)

    assert y_pred[0] == 0 and y_pred[9] == 1
