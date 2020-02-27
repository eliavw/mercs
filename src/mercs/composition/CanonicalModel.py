from functools import wraps
import warnings

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from catboost import CatBoostClassifier as CBC
    from catboost import CatBoostRegressor as CBR
except:
    CBC, CBR = None, None


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

    def __init__(self, model, desc_ids, targ_ids, out_kind, performance=1.0):

        self.model = model

        # Bookkeeping
        self.desc_ids = desc_ids
        self.targ_ids = targ_ids
        self.out_kind = out_kind
        self.score = performance

        if hasattr(model, 'shap_values_'):
            self.feature_importances_ = self.model.shap_values_
        else:
            self.feature_importances_ = self.model.feature_importances_

        # Canonization of prediction-related stuff
        if self.out_kind in {"nominal", "mix"}:
            self.predict = self.model.predict
            self.predict_proba = self.model.predict_proba
            self.classes_ = self.model.classes_

            if isinstance(model, type(CBC)):
                self.n_classes_ = len(self.classes_)
                self.classes_ = [int(c) for c in self.classes_]
                self.predict = catboost_predict(self.model.predict)
            else:
                self.n_classes_ = self.model.n_classes_

            if len(targ_ids) == 1:
                self.classes_ = [self.classes_]

            if single_target_sklearn_classifier(model):
                pass
                # self.predict = canonical_predict(self.model.predict)
                # self.predict_proba = canonical_predict_proba(self.model.predict_proba)
                # self.classes_ = [self.model.classes_]
                # self.n_classes_ = [self.model.n_classes_]

        elif self.out_kind in {"numeric"}:
            self.predict = self.model.predict

            # if single_target_sklearn_regressor(model):
            #    self.predict = canonical_predict(self.model.predict)

        else:
            raise NotImplementedError(
                "I do not recognize this kind of model: {}".format(out_kind)
            )

        return

    def __len__(self):
        """Returns the number of estimators in the ensemble."""
        try:
            return len(self.model)
        except TypeError:
            # The underlying model does not allow expansion.
            return 1

    def __getitem__(self, index):
        """Returns the index'th estimator in the ensemble."""
        try:
            return CanonicalModel(
                self.model[index],
                self.desc_ids,
                self.targ_ids,
                self.out_kind,
                self.score,
            )
        except TypeError:
            # The underlying model does not allow expansion, only index 0 makes sense.
            assert (
                index == 0
            ), "You are not an ensemble model, so there can be only one index: 0"
            return self

    def __iter__(self):
        """Returns iterator over estimators in the ensemble."""
        return (self[i] for i in range(len(self)))


# Wrappers
def catboost_predict(f):
    @wraps(f)
    def predict(*args, **kwargs):
        y_pred = f(*args, **kwargs)
        return y_pred.astype(int)

    return predict


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
