from src.mercs.core import Mercs
from src.mercs.tests import default_dataset
import pandas as pd
import numpy as np


def test_init():
    train, test = default_dataset(n_features=3)

    df = pd.DataFrame(train)
    df.head()

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

    nominal_ids = {train.shape[1]-1}

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

    # Single target
    q_code = np.zeros(clf.m_codes[0].shape[0], dtype=int)
    q_code[-1:] = 1
    print("Query code is: {}".format(q_code))

    y_pred = clf.predict(test, q_code=q_code)
    print(y_pred[:10])


test_init()