from mercs import Mercs
from mercs.utils import default_dataset
import numpy as np


def test_init():

    # load default dataset and print head
    train, test = default_dataset(n_features=3)

    # initialise MERCS model
    # the nb_targets defines the number of targets to use while fitting the model
    # but it is unrelated to the number of targets to use while predicting
    clf = Mercs(
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

    # ids of the nominal variables
    nominal_ids = {train.shape[1]-1}

    # fit the model
    clf.fit(train, nominal_attributes=nominal_ids)

    for idx, m in enumerate(clf.m_list):
        msg = """
        Model with index: {}
        {}
        """.format(idx, m.model)
        print(msg)

    for m_idx, m in enumerate(clf.m_list):
        msg = """
        Tree with id:          {}
        has source attributes: {}
        has target attributes: {},
        and predicts {} attributes
        """.format(m_idx, m.desc_ids, m.targ_ids, m.out_kind)
        print(msg)

    # Query code is [0 0 0 1] where 0 = descriptive and 1 = target
    q_code = np.array([0, 0, 0, 1])
    print("Query code is:", q_code)

    # predict value of query code for test data
    y_pred = clf.predict(test, q_code=q_code)
    # print the first 10 predictions
    print("Predictions:", y_pred[:10])


test_init()
