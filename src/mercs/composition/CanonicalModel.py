from functools import wraps

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class CanonicalModel(object):
    """
    Canonical Attributes:
        - desc_ids, {1,2}
        - targ_ids, {3}
        - out_kind, {'numerical', 'nominal', 'mix'}
        - feature_importances
        - classes_,  list of arrays of shape n_classes
        - n_classes_, list of ints

    Canonical Methods:
        - predict, np.ndarray, shape = nb_instances X nb_desc_ids
        - predict_proba, list, each entry is np.ndarray of shape = nb_instances X nb_classes
       
    """

    def __init__(self, model, desc_ids, targ_ids, out_kind):

        self.model = model

        # Bookkeeping
        self.desc_ids = desc_ids
        self.targ_ids = targ_ids
        self.out_kind = out_kind

        self.feature_importances_ =  self.model.feature_importances_

        # Canonization of prediction-related stuff
        if self.out_kind in {"nominal", "mix"}:
            self.predict = self.model.predict
            self.predict_proba = self.model.predict_proba
            self.classes_ = self.model.classes_
            self.n_classes_ = self.model.n_classes_

            if single_target_sklearn_classifier(model):
                self.predict = canonical_predict(self.model.predict)
                self.predict_proba = canonical_predict_proba(self.model.predict_proba)
                self.classes_ = [self.model.classes_]
                self.n_classes_ = [self.model.n_classes_]

        elif self.out_kind in {"numeric"}:
            self.predict = self.model.predict

            if single_target_sklearn_regressor(model):
                self.predict = canonical_predict(self.model.predict)

        else:
            raise NotImplementedError(
                "I do not recognize this kind of model: {}".format(out_kind)
            )

        return


# Wrappers
def canonical_predict(f):
    @wraps(f)
    def predict(*args, **kwargs):
        return f(*args, **kwargs).reshape(-1, 1)

    return predict


def canonical_predict_proba(f):
    @wraps(f)
    def predict_proba(*args, **kwargs):
        return [f(*args, **kwargs)]

    return predict_proba


# Helpers
def single_target_sklearn_regressor(model):
    return (
        isinstance(model, (DecisionTreeRegressor, RandomForestRegressor))
        and model.n_outputs_ == 1
    )


def single_target_sklearn_classifier(model):
    return (
        isinstance(model, (DecisionTreeClassifier, RandomForestClassifier))
        and model.n_outputs_ == 1
    )
